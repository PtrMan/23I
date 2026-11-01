"""
titansA1_fixed.py  —  fixed version of the Titans LMM implementation.

Bug-fix log (relative to original titansA0):
  BUG 1 [CRITICAL]  alpha_gate_proj: sigmoid(0) = 0.5 per step.
                    Over seq_len=64:  (0.5)^64 ≈ 5e-20  →  gradient to M_0 ≈ 0  →  M_0 never updates.
                    FIX: initialise bias to −4.0  →  sigmoid(−4) ≈ 0.018 per step,
                         giving (0.982)^64 ≈ 0.32 gradient at the first position  →  usable BPTT.

  BUG 2 [CRITICAL]  Internal MSE  reduction='mean'  divides by  batch_size × value_dim  (16×64 = 1024).
                    grad_surprise is 1024× too small  →  memory update ≈ 0  →  retrieval stays random
                    →  loss stays near log(30522) ≈ 10.3.
                    FIX: use  reduction='sum'  (sum only over value_dim, averaged over batch at the
                         tensor-function level is more natural, but sum lets theta_t control the scale).
                         Also: clip grad_surprise by global norm (see BUG 4).

  BUG 3 [CRITICAL]  eta_gate_proj: sigmoid(0) = 0.5 momentum decay per step.
                    Momentum evaporates in ~3 steps; update accumulation is negligible.
                    FIX: initialise bias to +2.0  →  sigmoid(2) ≈ 0.88 per step  →  useful momentum.

  BUG 4 [HIGH]      With reduction='sum' the raw grad_surprise can be huge when value_dim is large,
                    causing memory parameters to explode.
                    FIX: clip each grad tensor so the global L2 norm across all params ≤ max_surprise_norm.

  BUG 5 [HIGH]      No residual / bypass connection in LMModelWithLMM.
                    At the start of training M_0 is random  →  retrieved_memory is noise  →  the
                    output_layer receives noise  →  all layers upstream get useless gradients.
                    FIX: project the embedded input to value_dim and ADD it to the LMM output
                         (a learned residual).  This gives the model a direct bypass path immediately.

  BUG 6 [MEDIUM]    No LayerNorm on LMM output  →  unbounded scale entering the output_layer  →
                    logits that are too large/small, slow convergence.
                    FIX: add nn.LayerNorm(lmm_value_dim) before the output_layer.

  BUG 7 [MINOR]     Dead variable  x_perm = x.permute(1, 2, 0)  computed but never used.
                    FIX: removed.
"""

import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Functional MLP helper
# ─────────────────────────────────────────────────────────────────────────────

def functional_mlp_forward(x, params, activation_fn):
    """Applies an MLP functionally using a flat list [w0, b0, w1, b1, ...]."""
    if not params:
        return x
    if len(params) % 2 != 0:
        raise ValueError(f"Functional MLP requires an even number of params. Got {len(params)}.")
    current_x = x
    num_layers = len(params) // 2
    for i in range(num_layers):
        weight = params[i * 2]
        bias   = params[i * 2 + 1]
        if current_x.shape[-1] != weight.shape[1]:
            raise RuntimeError(
                f"Shape mismatch in functional MLP layer {i}: "
                f"input {current_x.shape}, weight expects dim {weight.shape[1]}"
            )
        current_x = F.linear(current_x, weight, bias)
        if i < num_layers - 1:
            current_x = activation_fn(current_x)
    return current_x


def _build_mlp(input_dim, hidden_dims, output_dim, activation=nn.SiLU):
    layers = []
    current_dim = input_dim
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    elif hidden_dims is None:
        hidden_dims = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Long-Term Memory Module
# ─────────────────────────────────────────────────────────────────────────────

class LongTermMemoryModule(nn.Module):
    """
    Neural Long-Term Memory Module (LMM) from the Titans paper.

    Memory update rule (per time step t):
        grad_t  = ∇_M  ||M_{t-1}(k_t) − v_t||²   (surprise = gradient of associative loss)
        S_t     = η_t · S_{t-1} − θ_t · grad_t     (momentum in descent direction)
        M_t     = (1 − α_t) · M_{t-1} + S_t        (update with weight decay / forget gate)
        out_t   = M_t(q_t)                          (retrieval)

    Learnable initial state M_0 is trained by the outer optimiser via BPTT.
    """

    def __init__(self,
                 input_dim:          int,
                 key_dim:            int,
                 value_dim:          int,
                 query_dim:          int,
                 memory_hidden_dims: List[int],
                 memory_depth:       int   = 2,
                 use_convolution:    bool  = True,
                 conv_kernel_size:   int   = 3,
                 normalize_qk:       bool  = True,
                 learn_gates:        bool  = True,
                 fixed_alpha:        float = 0.01,
                 fixed_eta:          float = 0.9,
                 fixed_theta:        float = 0.01,
                 max_surprise_norm:  float = 1.0):   # BUG 4 fix: gradient clipping budget
        super().__init__()
        if key_dim != query_dim:
            raise ValueError(f"LMM requires key_dim ({key_dim}) == query_dim ({query_dim})")

        self.input_dim        = input_dim
        self.key_dim          = key_dim
        self.value_dim        = value_dim
        self.query_dim        = query_dim
        self.normalize_qk     = normalize_qk
        self.learn_gates      = learn_gates
        self.fixed_alpha      = fixed_alpha
        self.fixed_eta        = fixed_eta
        self.fixed_theta      = fixed_theta
        self.max_surprise_norm = max_surprise_norm

        # Projections
        self.Wk = nn.Linear(input_dim, key_dim)
        self.Wv = nn.Linear(input_dim, value_dim)
        self.Wq = nn.Linear(input_dim, query_dim)

        # Optional convolution
        self.use_convolution = use_convolution
        if use_convolution:
            pad = (conv_kernel_size - 1) // 2
            self.conv_k = nn.Conv1d(key_dim,   key_dim,   conv_kernel_size, padding=pad, groups=key_dim)
            self.conv_v = nn.Conv1d(value_dim, value_dim, conv_kernel_size, padding=pad, groups=value_dim)
            self.conv_q = nn.Conv1d(query_dim, query_dim, conv_kernel_size, padding=pad, groups=query_dim)

        # Memory network — holds trainable initial parameters M_0
        self.memory_net_initial = _build_mlp(key_dim, memory_hidden_dims, value_dim, activation=nn.SiLU)
        self.memory_activation  = nn.SiLU()

        # Gates
        if learn_gates:
            self.alpha_gate_proj = nn.Linear(input_dim, 1)
            self.eta_gate_proj   = nn.Linear(input_dim, 1)
            self.theta_gate_proj = nn.Linear(input_dim, 1)
            # ── BUG 1 FIX ──────────────────────────────────────────────────
            # Default bias=0 → sigmoid(0)=0.5 per step → (0.5)^64 ≈ 5e-20
            # gradient to M_0. Bias toward 0 weight-decay so BPTT flows.
            nn.init.constant_(self.alpha_gate_proj.bias, -4.0)   # sigmoid(-4) ≈ 0.018
            # ── BUG 3 FIX ──────────────────────────────────────────────────
            # Default bias=0 → eta=0.5. Strong momentum needs eta close to 1.
            nn.init.constant_(self.eta_gate_proj.bias, 2.0)      # sigmoid(2) ≈ 0.88
            # theta: softplus(0)=0.693 is a reasonable starting learning rate;
            # leave it, but its scale is now meaningful because of the sum fix.

        # Momentum-state zero template (used for device-aware initialisation)
        self._momentum_template = [
            torch.zeros_like(p, requires_grad=False)
            for p in self.memory_net_initial.parameters()
        ]

    def _init_momentum(self, device):
        return [torch.zeros_like(t, device=device, requires_grad=False)
                for t in self._momentum_template]

    def _initial_params(self):
        return list(self.memory_net_initial.parameters())

    # ── BUG 4 FIX: clip the surprise gradient tensors by global L2 norm ──────
    @staticmethod
    def _clip_grads(grads: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
        total_norm = torch.stack([g.norm() for g in grads]).norm()
        clip_coef  = max_norm / (total_norm + 1e-6)
        clip_coef  = torch.clamp(clip_coef, max=1.0)
        return [g * clip_coef for g in grads]

    def forward(self,
                x:             torch.Tensor,
                initial_state: Optional[Tuple[List[torch.Tensor],
                                               List[torch.Tensor]]] = None
                ) -> Tuple[torch.Tensor,
                           Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        x : [SeqLen, BatchSize, InputDim]
        Returns (output_sequence [SeqLen, BatchSize, ValueDim], final_state_detached)
        """
        seq_len, batch_size, _ = x.shape
        device = x.device
        grad_outer = torch.is_grad_enabled()

        # Initialise state
        if initial_state is not None:
            M = [p.to(device).detach().requires_grad_(grad_outer and p.is_floating_point())
                 for p in initial_state[0]]
            S = [s.to(device).detach().requires_grad_(grad_outer and s.is_floating_point())
                 for s in initial_state[1]]
        else:
            M = [p.clone() for p in self._initial_params()]
            S = self._init_momentum(device)
            S = [s.requires_grad_(grad_outer and s.is_floating_point()) for s in S]

        # Project all time steps at once
        k_proj = self.Wk(x).permute(1, 2, 0)   # [B, key_dim, T]
        v_proj = self.Wv(x).permute(1, 2, 0)   # [B, value_dim, T]
        q_proj = self.Wq(x).permute(1, 2, 0)   # [B, query_dim, T]

        # BUG 7 FIX: removed unused `x_perm` variable

        if self.use_convolution:
            k_proj = self.conv_k(k_proj)
            v_proj = self.conv_v(v_proj)
            q_proj = self.conv_q(q_proj)

        keys    = k_proj.permute(2, 0, 1)   # [T, B, key_dim]
        values  = v_proj.permute(2, 0, 1)   # [T, B, value_dim]
        queries = q_proj.permute(2, 0, 1)   # [T, B, query_dim]

        outputs = []

        for t in range(seq_len):
            xt = x[t]           # [B, input_dim]
            kt = keys[t]        # [B, key_dim]
            vt = values[t]      # [B, value_dim]
            qt = queries[t]     # [B, query_dim]

            if self.normalize_qk:
                kt = F.normalize(kt, p=2, dim=-1)
                qt = F.normalize(qt, p=2, dim=-1)

            # Gates
            if self.learn_gates:
                alpha_t = torch.sigmoid(self.alpha_gate_proj(xt)).mean()   # scalar
                eta_t   = torch.sigmoid(self.eta_gate_proj(xt)).mean()     # scalar
                theta_t = F.softplus(self.theta_gate_proj(xt)).mean()      # scalar
            else:
                alpha_t = torch.tensor(self.fixed_alpha, device=device)
                eta_t   = torch.tensor(self.fixed_eta,   device=device)
                theta_t = torch.tensor(self.fixed_theta, device=device)

            # ── Compute surprise (internal gradient) ─────────────────────────
            kt_in  = kt.detach().clone()
            vt_tgt = vt.detach().clone()

            with torch.enable_grad():
                tmp_params = [p.clone().detach().requires_grad_(True) for p in M]
                pred_v     = functional_mlp_forward(kt_in, tmp_params, self.memory_activation)
                # BUG 2 FIX: use sum (not mean) so grad_surprise reflects the
                # actual gradient magnitude, not 1/(batch*value_dim) of it.
                loss_int   = F.mse_loss(pred_v, vt_tgt, reduction='sum')
                if loss_int.requires_grad:
                    raw_grads = torch.autograd.grad(
                        loss_int, tmp_params,
                        create_graph=False,
                        allow_unused=True
                    )
                    grads = [
                        g if g is not None else torch.zeros_like(p, device=device)
                        for g, p in zip(raw_grads, tmp_params)
                    ]
                else:
                    grads = [torch.zeros_like(p, device=device) for p in tmp_params]

            # BUG 4 FIX: clip surprise gradients so memory update stays bounded
            grads = self._clip_grads(grads, self.max_surprise_norm)

            # ── State update (part of outer computation graph) ───────────────
            S_new = [eta_t * s - theta_t * g for s, g in zip(S, grads)]
            M_new = [(1.0 - alpha_t) * m + s for m, s in zip(M, S_new)]

            # ── Retrieval from updated M_t ────────────────────────────────────
            out = functional_mlp_forward(qt, M_new, self.memory_activation)
            outputs.append(out)

            M, S = M_new, S_new

        output_seq   = torch.stack(outputs, dim=0)                          # [T, B, value_dim]
        final_state  = ([p.detach() for p in M], [s.detach() for s in S])
        return output_seq, final_state


# ─────────────────────────────────────────────────────────────────────────────
# Language model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LMModelWithLMM(nn.Module):
    """
    LM wrapper around LongTermMemoryModule.

    Fixes applied:
      BUG 5 – residual projection: embed_dim → value_dim added to LMM output.
      BUG 6 – LayerNorm on LMM output before the vocab head.
    """

    def __init__(self,
                 vocab_size:            int,
                 embed_dim:             int,
                 lmm_key_dim:           int,
                 lmm_value_dim:         int,
                 lmm_query_dim:         int,
                 lmm_memory_hidden_dims: List[int],
                 lmm_memory_depth:      int  = 2,
                 lmm_use_convolution:   bool = True,
                 lmm_learn_gates:       bool = True,
                 max_seq_len:           int  = 512,
                 max_surprise_norm:     float = 1.0):
        super().__init__()
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.max_seq_len = max_seq_len

        self.embedding   = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        self.lmm = LongTermMemoryModule(
            input_dim          = embed_dim,
            key_dim            = lmm_key_dim,
            value_dim          = lmm_value_dim,
            query_dim          = lmm_query_dim,
            memory_hidden_dims = lmm_memory_hidden_dims,
            memory_depth       = lmm_memory_depth,
            use_convolution    = lmm_use_convolution,
            learn_gates        = lmm_learn_gates,
            max_surprise_norm  = max_surprise_norm,
        )

        # BUG 5 FIX: learned bypass from input embeddings → value_dim space.
        # Without this, if LMM output is initially random/small, the output layer
        # receives no useful signal and all upstream gradients are useless.
        self.residual_proj = nn.Linear(embed_dim, lmm_value_dim, bias=False)

        # BUG 6 FIX: normalise LMM+residual output before the vocabulary head.
        self.out_norm    = nn.LayerNorm(lmm_value_dim)

        self.output_layer = nn.Linear(lmm_value_dim, vocab_size)

    def forward(self,
                input_ids:         torch.Tensor,
                initial_lmm_state: Optional[Tuple[List[torch.Tensor],
                                                   List[torch.Tensor]]] = None
                ) -> Tuple[torch.Tensor,
                           Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        input_ids : [BatchSize, SeqLen]
        Returns (logits [BatchSize, SeqLen, VocabSize], final_lmm_state)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Embedding + positional encoding
        embeds      = self.embedding(input_ids)                         # [B, T, D]
        pos_enc_len = min(seq_len, self.max_seq_len)
        pos_enc     = self.pos_encoder[:, :pos_enc_len, :].to(device)
        x           = embeds[:, :pos_enc_len, :] + pos_enc
        if seq_len > pos_enc_len:
            x = torch.cat([x, embeds[:, pos_enc_len:, :]], dim=1)

        # 2. LMM expects [SeqLen, BatchSize, InputDim]
        x_lmm = x.permute(1, 0, 2)                                      # [T, B, D]

        # 3. LMM forward
        lmm_out, final_state = self.lmm(x_lmm, initial_state=initial_lmm_state)
        lmm_out = lmm_out.permute(1, 0, 2)                              # [B, T, value_dim]

        # BUG 5 FIX: add residual from input embedding
        residual = self.residual_proj(x)                                 # [B, T, value_dim]
        combined = lmm_out + residual

        # BUG 6 FIX: layer norm before head
        normed  = self.out_norm(combined)                                # [B, T, value_dim]
        logits  = self.output_layer(normed)                              # [B, T, vocab_size]

        return logits, final_state


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        print(f"Tokenizing {len(texts)} texts with max_length {max_length}...")
        count = 0
        for text in texts:
            tokenized = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            for i in range(0, len(tokenized) - 1, max_length):
                chunk = tokenized[i: i + max_length + 1]
                if len(chunk) < 2:
                    continue
                input_ids  = chunk[:-1]
                target_ids = chunk[1:]
                ip = max_length - len(input_ids)
                tp = max_length - len(target_ids)
                input_ids  = input_ids  + [tokenizer.pad_token_id] * ip
                target_ids = target_ids + [-100] * tp
                self.examples.append({
                    "input_ids":  torch.tensor(input_ids[:max_length],  dtype=torch.long),
                    "target_ids": torch.tensor(target_ids[:max_length], dtype=torch.long),
                })
                count += 1
        print(f"Created {count} training examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    PATH = "titansA1.pth"

    # ── Training-data search paths ────────────────────────────────────────────
    # Add as many directories as you like.  Each is searched for *.txt files.
    # Set recursive=True to also descend into sub-directories.
    CORPUS_DIRECTORIES = [
        "/notebooks/trainingdata__text/",   # original path
        # "D:/mydata/text/",
        # "C:/Users/me/Documents/corpus/",
    ]
    CORPUS_RECURSIVE = False   # set True to search subdirectories too

    # Hyperparameters
    VOCAB_SIZE      = 30522
    EMBED_DIM       = 128
    LMM_KEY_DIM     = 64
    LMM_VALUE_DIM   = 64
    LMM_QUERY_DIM   = LMM_KEY_DIM          # must match
    LMM_MEM_HIDDEN  = [256]
    LMM_MEM_DEPTH   = len(LMM_MEM_HIDDEN) + 1
    LMM_USE_CONV    = False
    LMM_LEARN_GATES = True
    MAX_SEQ_LEN     = 64
    BATCH_SIZE      = 16
    LEARNING_RATE   = 1e-4
    NUM_EPOCHS        = 350
    SAVE_EVERY_EPOCHS = 10     # save checkpoint every N epochs (not every epoch)
    DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_EVAL_TOKENS     = 30
    MAX_SURPRISE_NORM = 1.0    # BUG 4 FIX: clip budget for grad_surprise

    print(f"Using device: {DEVICE}")

    # Tokenizer
    tokenizer_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    VOCAB_SIZE = tokenizer.vocab_size
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        VOCAB_SIZE += 1
    print(f"Tokenizer: {tokenizer_name}  vocab={VOCAB_SIZE}  pad_id={tokenizer.pad_token_id}")

    # Load corpus from every directory in CORPUS_DIRECTORIES
    all_texts   = []
    total_files = 0
    pattern     = '**/*.txt' if CORPUS_RECURSIVE else '*.txt'
    for directory in CORPUS_DIRECTORIES:
        if not os.path.isdir(directory):
            print(f"[data] '{directory}' not found — skipped.")
            continue
        found = glob.glob(os.path.join(directory, pattern), recursive=CORPUS_RECURSIVE)
        if not found:
            print(f"[data] '{directory}' — no .txt files found.")
            continue
        for fp in found:
            try:
                with open(fp, encoding='utf-8') as f:
                    lines = [l.strip() for l in f if l.strip()]
                all_texts.extend(lines)
                total_files += 1
            except Exception as e:
                print(f"[data] skip {fp}: {e}")
        print(f"[data] '{directory}' — {len(found)} files loaded.")
    if all_texts:
        print(f"[data] Total: {len(all_texts):,} lines from {total_files} files.")
    else:
        print("[data] No text found in any directory. Using inline fallback data.")
        all_texts = [
            "The study of long-term memory involves understanding how information is encoded, stored, and retrieved.",
            "Neural networks provide a powerful framework for modeling cognitive processes including memory.",
            "Recurrent neural networks were early models for sequence processing but suffer from vanishing gradients.",
            "Transformers with their attention mechanisms have become state-of-the-art for language processing.",
            "The Titans paper introduces a novel long-term memory module updated during the forward pass.",
            "This online learning mechanism allows the memory to adapt to incoming data at inference time.",
            "The memory update rule is inspired by gradient descent with momentum and weight decay.",
            "Keys, values, and queries are projected from the input sequence similar to attention mechanisms.",
            "An internal memory network maps keys to predicted values like an associative store.",
            "The difference between predicted and actual values drives the surprise-based memory update.",
            "Momentum helps smooth updates and accelerate learning in consistent directions.",
            "Weight decay controlled by the alpha gate acts as a forgetting mechanism.",
            "Data-dependent gates allow the update dynamics to adapt based on the input context.",
            "This recurrent implementation focuses on clarity and demonstrates the core update logic.",
            "Training involves learning the initial parameters of the projections, gates, and memory network.",
            "During inference the learned initial parameters are dynamically modified by the internal update rule.",
            "The retrieved memory output can be used by subsequent layers in a larger architecture.",
            "Potential applications include lifelong learning, continual adaptation, and very long sequences.",
            "Evaluating such models requires tasks that test long-range dependencies and adaptation.",
            "Tokenization converts raw text into numerical IDs that the model can process.",
            "Embeddings map discrete token IDs into continuous vector representations.",
            "Positional encodings provide the model with information about the order of tokens.",
            "The final layer maps the internal representation to logits over the vocabulary.",
            "Cross-entropy loss trains language models by comparing predicted logits to target token IDs.",
            "AdamW adjusts the model parameters to minimize the loss during training.",
            "Gradient clipping prevents exploding gradients and stabilizes training.",
            "Perplexity measures how well the model predicts a held-out sample of text.",
            "Generating text involves iteratively predicting the next token and updating the context.",
            "Long context processing is a key challenge for modern language models.",
            "Memory modules can help retain and retrieve information across very long sequences.",
        ] * 10

    train_dataset = TextDataset(all_texts, tokenizer, max_length=MAX_SEQ_LEN)
    if len(train_dataset) == 0:
        raise ValueError("Dataset is empty. Check CORPUS_DIRECTORY or fallback data.")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset: {len(train_dataset)} examples  |  DataLoader ready.")

    # Model
    loadModel = False   # set True to resume from titansA1.pth

    model = LMModelWithLMM(
        vocab_size            = VOCAB_SIZE,
        embed_dim             = EMBED_DIM,
        lmm_key_dim           = LMM_KEY_DIM,
        lmm_value_dim         = LMM_VALUE_DIM,
        lmm_query_dim         = LMM_QUERY_DIM,
        lmm_memory_hidden_dims = LMM_MEM_HIDDEN,
        lmm_memory_depth      = LMM_MEM_DEPTH,
        lmm_use_convolution   = LMM_USE_CONV,
        lmm_learn_gates       = LMM_LEARN_GATES,
        max_seq_len           = MAX_SEQ_LEN,
        max_surprise_norm     = MAX_SURPRISE_NORM,
    ).to(DEVICE)

    if loadModel and os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH, map_location=DEVICE))
        print(f"Loaded model from {PATH}")
    else:
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.apply(init_weights)
        # Re-apply gate biases AFTER init_weights (which zeros all biases)
        if LMM_LEARN_GATES:
            nn.init.constant_(model.lmm.alpha_gate_proj.bias, -4.0)   # BUG 1 FIX
            nn.init.constant_(model.lmm.eta_gate_proj.bias,   2.0)    # BUG 3 FIX
        print(f"Model initialised: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    model.train()
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        total_loss, num_batches = 0.0, 0
        for i, batch in enumerate(train_dataloader):
            input_ids  = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)

            logits, _ = model(input_ids, initial_lmm_state=None)
            loss = criterion(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

            if (i + 1) % 20 == 0 or i == len(train_dataloader) - 1:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  "
                      f"Batch [{i+1}/{len(train_dataloader)}]  "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, num_batches)
        print(f"--- Epoch {epoch+1} done --- avg loss: {avg_loss:.4f} ---")
        # Save every SAVE_EVERY_EPOCHS epochs (and always on the very last epoch).
        # Saving on EVERY epoch causes the visible pause between epochs because
        # torch.save serialises the full state dict to disk synchronously.
        if (epoch + 1) % SAVE_EVERY_EPOCHS == 0 or epoch == NUM_EPOCHS - 1:
            torch.save(model.state_dict(), PATH)
            print(f"    [ckpt saved → {PATH}]")

    print("--- Training finished. ---")

    # Generation
    model.eval()
    prompt = 'A frog is'
    print(f"\n--- Generating {N_EVAL_TOKENS} tokens ---")
    print(f"Prompt: '{prompt}'")
    input_ids      = tokenizer.encode(prompt, add_special_tokens=False)
    generated_ids  = list(input_ids)
    lmm_state      = None
    cur_input      = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        for i in range(N_EVAL_TOKENS):
            logits, lmm_state = model(cur_input, initial_lmm_state=lmm_state)
            next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated_ids.append(next_id)
            cur_input = torch.cat([cur_input,
                                   torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)],
                                  dim=1)
            if cur_input.shape[1] > MAX_SEQ_LEN:
                cur_input = cur_input[:, -MAX_SEQ_LEN:]
            if tokenizer.eos_token_id and next_id == tokenizer.eos_token_id:
                break
            if tokenizer.sep_token_id and next_id == tokenizer.sep_token_id:
                break

    print("\n--- Generated Text ---")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
