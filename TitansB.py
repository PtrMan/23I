import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Dict
import math
import copy
#import os # For checking dummy file existence


import os
import glob # Import the glob module to find files matching a pattern


# --- Enable Anomaly Detection (Crucial for Debugging This!) ---
# Can be disabled after debugging if it causes performance issues
#torch.autograd.set_detect_anomaly(True)

# --- Functional MLP Helper ---
def functional_mlp_forward(x, params, activation_fn):
    """Applies an MLP functionally using a list of parameters."""
    if not params: # Handle empty MLP case (e.g., depth 0)
        return x
    if len(params) % 2 != 0:
        raise ValueError(f"Functional MLP requires an even number of parameters (weights and biases). Got {len(params)}.")
    current_x = x
    num_layers = len(params) // 2
    for i in range(num_layers):
        weight = params[i * 2]
        bias = params[i * 2 + 1]
        # Ensure dimensions match
        if current_x.shape[-1] != weight.shape[1]:
             raise RuntimeError(f"Shape mismatch in functional MLP layer {i}: input has shape {current_x.shape}, "
                                f"but weight expects dimension {weight.shape[1]} (weight shape: {weight.shape})")
        current_x = F.linear(current_x, weight, bias)
        if i < num_layers - 1: # Apply activation to hidden layers only
            current_x = activation_fn(current_x)
    return current_x

# Helper function to build the MLP structure (used for parameter initialization)
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


class LongTermMemoryModule(nn.Module):
    """
    Implements the Neural Long-Term Memory Module (LMM) from the Titans paper.
    This version uses a functional recurrent update mechanism suitable for training.
    It learns to memorize key-value associations at test/inference time.
    """
    def __init__(self,
                 input_dim: int,
                 key_dim: int,
                 value_dim: int,
                 query_dim: int, # MUST == key_dim for this functional approach
                 memory_hidden_dims: List[int] or int,
                 memory_depth: int = 2, # L_M >= 1
                 use_convolution: bool = True,
                 conv_kernel_size: int = 3,
                 normalize_qk: bool = True,
                 learn_gates: bool = True,
                 fixed_alpha: float = 0.01,
                 fixed_eta: float = 0.9,
                 fixed_theta: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_dim = query_dim
        self.normalize_qk = normalize_qk
        self.learn_gates = learn_gates
        self.fixed_alpha = fixed_alpha
        self.fixed_eta = fixed_eta
        self.fixed_theta = fixed_theta

        if key_dim != query_dim:
             # This constraint is necessary because the same functional MLP M
             # is applied to both k_t (for loss) and q_t (for retrieval).
             raise ValueError(f"Functional LMM requires key_dim ({key_dim}) == query_dim ({query_dim})")

        # --- Projections ---
        self.Wk = nn.Linear(input_dim, key_dim)
        self.Wv = nn.Linear(input_dim, value_dim)
        self.Wq = nn.Linear(input_dim, query_dim)

        # --- Optional Convolution ---
        self.use_convolution = use_convolution
        if use_convolution:
            padding = (conv_kernel_size - 1) // 2
            self.conv_k = nn.Conv1d(key_dim, key_dim, kernel_size=conv_kernel_size, padding=padding, groups=key_dim)
            self.conv_v = nn.Conv1d(value_dim, value_dim, kernel_size=conv_kernel_size, padding=padding, groups=value_dim)
            self.conv_q = nn.Conv1d(query_dim, query_dim, kernel_size=conv_kernel_size, padding=padding, groups=query_dim)

        # --- Memory Network (MLP) - Holds ONLY the trainable initial parameters M_0 ---
        # We build it to easily get the parameters, but won't call its forward method directly in the loop.
        self.memory_net_initial = _build_mlp(key_dim, memory_hidden_dims, value_dim, activation=nn.SiLU)
        self.memory_activation = nn.SiLU() # Store activation function instance

        # --- Gates ---
        if learn_gates:
            self.alpha_gate_proj = nn.Linear(input_dim, 1)
            self.eta_gate_proj = nn.Linear(input_dim, 1)
            self.theta_gate_proj = nn.Linear(input_dim, 1)

        # --- State Initialization Structure ---
        self._initial_momentum_state_structure = [torch.zeros_like(p, requires_grad=False)
                                                  for p in self.memory_net_initial.parameters()]

    def _initialize_momentum_state(self, device: torch.device) -> List[torch.Tensor]:
        """Initializes the momentum buffer S with zeros on the correct device."""
        return [torch.zeros_like(p_struct, device=device, requires_grad=False)
                for p_struct in self._initial_momentum_state_structure]

    def _get_initial_memory_params(self) -> List[nn.Parameter]:
        """Gets the trainable initial parameters M_0 of the memory network."""
        return list(self.memory_net_initial.parameters())

    def forward(self,
                x: torch.Tensor,
                initial_state: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None
               ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Processes the input sequence token by token functionally.

        Args:
            x: Input sequence tensor of shape [SeqLen, BatchSize, InputDim].
            initial_state: Optional tuple containing (initial_memory_params M_{t-1}, initial_momentum_state S_{t-1}).
                           If None, uses the network's initial parameters M_0 and zero momentum S_0.

        Returns:
            A tuple containing:
            - output_sequence: Tensor of shape [SeqLen, BatchSize, QueryDim] (retrieved memory).
            - final_state: Tuple containing (final_memory_params M_N, final_momentum_state S_N), detached.
        """
        seq_len, batch_size, _ = x.shape
        device = x.device
        grad_outer_enabled = torch.is_grad_enabled() # Check if gradients are needed for the outer loop

        # --- Initialize State ---
        # State tensors (M_t, S_t) will carry gradients if grad_outer_enabled=True
        if initial_state:
            # Use provided state. Ensure requires_grad matches outer context.
            # Detach first, then set requires_grad based on outer context.
            M_t_minus_1 = [p.to(device).detach().requires_grad_(grad_outer_enabled and p.is_floating_point()) for p in initial_state[0]]
            S_t_minus_1 = [s.to(device).detach().requires_grad_(grad_outer_enabled and s.is_floating_point()) for s in initial_state[1]]
        else:
            # Use trainable initial parameters M_0. Clone them to start the state sequence.
            # The gradient will flow back from M_t to M_0 (self.memory_net_initial.parameters)
            initial_params_M0 = self._get_initial_memory_params()
            # Clone M_0; requires_grad is implicitly handled by cloning nn.Parameter if grad_outer_enabled
            M_t_minus_1 = [p.clone() for p in initial_params_M0]
            S_t_minus_1 = self._initialize_momentum_state(device=device)
            # Ensure requires_grad is set correctly on initial momentum state if needed (usually False)
            S_t_minus_1 = [s.requires_grad_(grad_outer_enabled and s.is_floating_point()) for s in S_t_minus_1]


        outputs = []
        # --- Projections and Convolution ---
        x_perm = x.permute(1, 2, 0)
        k_proj = self.Wk(x).permute(1, 2, 0)
        v_proj = self.Wv(x).permute(1, 2, 0)
        q_proj = self.Wq(x).permute(1, 2, 0)
        if self.use_convolution:
            k_proj = self.conv_k(k_proj)
            v_proj = self.conv_v(v_proj)
            q_proj = self.conv_q(q_proj)
        keys = k_proj.permute(2, 0, 1)
        values = v_proj.permute(2, 0, 1)
        queries = q_proj.permute(2, 0, 1)

        # --- Recurrent Processing (Token by Token - Functional State Update) ---
        for t in range(seq_len):
            xt = x[t]
            kt = keys[t]
            vt = values[t]
            qt = queries[t]

            if self.normalize_qk:
                kt = F.normalize(kt, p=2, dim=-1)
                qt = F.normalize(qt, p=2, dim=-1)

            # --- Gates ---
            # These calculations need to be part of the graph if gates are learned
            if self.learn_gates:
                alpha_t = torch.sigmoid(self.alpha_gate_proj(xt)).mean().squeeze()
                eta_t = torch.sigmoid(self.eta_gate_proj(xt)).mean().squeeze()
                theta_t = F.softplus(self.theta_gate_proj(xt)).mean().squeeze()
            else:
                # Use fixed hyperparameters (as detached tensors)
                alpha_t = torch.tensor(self.fixed_alpha, device=device, requires_grad=False)
                eta_t = torch.tensor(self.fixed_eta, device=device, requires_grad=False)
                theta_t = torch.tensor(self.fixed_theta, device=device, requires_grad=False)

            # --- Internal Loss & Gradient Calculation ---
            kt_input = kt.detach().clone() # Input to internal loss should not track gradient
            vt_target = vt.detach().clone() # Target for internal loss should not track gradient

            # *** Temporarily enable gradient calculation for the internal update mechanism ***
            with torch.enable_grad():
                # We need M_{t-1} with gradients enabled *only* for the internal grad calculation
                # These params are temporary and local to this 'enable_grad' block
                params_for_internal_grad = [p.clone().detach().requires_grad_(True) for p in M_t_minus_1]

                # Calculate internal loss using these temporary parameters
                # This calculation now happens *with* gradient tracking enabled
                predicted_v_internal = functional_mlp_forward(kt_input, params_for_internal_grad, self.memory_activation)
                loss_internal = F.mse_loss(predicted_v_internal, vt_target, reduction='mean') # loss_internal will have grad_fn

                # Calculate gradient w.r.t temporary M_{t-1}, *without* creating graph for outer loop
                # Check requires_grad just in case, though enable_grad should ensure it
                if loss_internal.requires_grad:
                    grads_internal = torch.autograd.grad(loss_internal, params_for_internal_grad,
                                                         create_graph=False, # Do NOT backprop through this grad calc
                                                         allow_unused=True)
                    momentary_surprise = [g if g is not None else torch.zeros_like(p, device=device)
                                          for g, p in zip(grads_internal, params_for_internal_grad)]
                else:
                     # This case should not happen if M_t_minus_1 has params and enable_grad is on
                     print("Warning: Internal loss does not require grad. Momentary surprise set to zeros.")
                     momentary_surprise = [torch.zeros_like(p, device=device) for p in params_for_internal_grad]

            # *** Gradient calculation is disabled again here if outer context is no_grad() ***
            # momentary_surprise is now a list of detached gradient tensors

            # --- State Update Calculations (Part of the Outer Graph if grad_outer_enabled) ---
            # These calculations happen outside the enable_grad block.
            # Their gradient tracking depends on grad_outer_enabled and whether inputs (M_t_minus_1, S_t_minus_1, gates) require grad.
            # During inference (no_grad), these calculations will produce detached M_t, S_t.
            # During training (grad enabled), they will produce M_t, S_t that track gradients.

            # 1. Calculate S_t
            S_t = []
            for s_prev, grad_surprise in zip(S_t_minus_1, momentary_surprise):
                # grad_surprise is detached.
                # eta_t/theta_t track grads if gates learned. s_prev tracks if outer enabled.
                s_new = eta_t * s_prev - theta_t * grad_surprise
                S_t.append(s_new)

            # 2. Calculate M_t
            M_t = []
            for m_prev, s_new in zip(M_t_minus_1, S_t):
                 # alpha_t tracks grads if gates learned. m_prev tracks if outer enabled. s_new tracks.
                 m_new = (1.0 - alpha_t) * m_prev + s_new
                 M_t.append(m_new)

            # --- Memory Retrieval Step ---
            # Use the *newly calculated* M_t parameters functionally.
            # This step's gradient tracking depends on grad_outer_enabled.
            # qt might track gradients back to Wq/input x. M_t tracks back to M_0/gates/S_t.
            retrieved_memory = functional_mlp_forward(qt, M_t, self.memory_activation)
            outputs.append(retrieved_memory)

            # --- Update state for next iteration ---
            # The new state (M_t, S_t) carries the necessary gradient history
            M_t_minus_1 = M_t
            S_t_minus_1 = S_t

        # --- End of Loop ---
        output_sequence = torch.stack(outputs, dim=0)

        # Final state should be detached for returning, but M_t/S_t calculated above hold the graph history
        final_state_detached = ([p.detach() for p in M_t], [s.detach() for s in S_t])

        return output_sequence, final_state_detached




# --- Simple Model using LMM for Language Modeling ---

class LMModelWithLMM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 lmm_key_dim: int,
                 lmm_value_dim: int,
                 lmm_query_dim: int, # Should match lmm_key_dim now
                 lmm_memory_hidden_dims: List[int] or int,
                 lmm_memory_depth: int = 2,
                 lmm_use_convolution: bool = True,
                 lmm_learn_gates: bool = True,
                 max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        self.lmm = LongTermMemoryModule(
            input_dim=embed_dim,
            key_dim=lmm_key_dim,
            value_dim=lmm_value_dim,
            query_dim=lmm_query_dim, # Passed correctly
            memory_hidden_dims=lmm_memory_hidden_dims,
            memory_depth=lmm_memory_depth,
            use_convolution=lmm_use_convolution,
            learn_gates=lmm_learn_gates
        )

        # Output layer maps LMM output (query_dim) to vocab logits
        self.output_layer = nn.Linear(lmm_query_dim, vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                initial_lmm_state: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None
               ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Args:
            input_ids: Tensor of shape [BatchSize, SeqLen]
            initial_lmm_state: Optional initial state for the LMM.

        Returns:
            A tuple containing:
            - logits: Tensor of shape [BatchSize, SeqLen, VocabSize]
            - final_lmm_state: The final state of the LMM after processing the sequence (detached).
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Embedding + Positional Encoding
        embeds = self.embedding(input_ids)
        pos_enc_len = min(seq_len, self.max_seq_len)
        pos_enc = self.pos_encoder[:, :pos_enc_len, :].to(device)
        x = embeds[:, :pos_enc_len, :] + pos_enc
        if seq_len > pos_enc_len:
             x = torch.cat([x, embeds[:, pos_enc_len:, :]], dim=1)

        # 2. Reshape for LMM (expects [SeqLen, BatchSize, InputDim])
        x = x.permute(1, 0, 2) # [SeqLen, BatchSize, EmbedDim]

        # 3. Process with LMM, passing and receiving state
        # LMM processes the sequence functionally, updating state internally
        lmm_output_sequence, final_lmm_state = self.lmm(x, initial_state=initial_lmm_state)
        # lmm_output_sequence tracks gradients back through M_t, S_t to M_0 etc.

        # 4. Reshape back and predict logits
        lmm_output_sequence = lmm_output_sequence.permute(1, 0, 2) # [BatchSize, SeqLen, QueryDim]
        logits = self.output_layer(lmm_output_sequence) # [BatchSize, SeqLen, VocabSize]

        # final_lmm_state is detached, suitable for passing to the next call
        return logits, final_lmm_state

    
    
    
    
    
    
    
    
    
    
    

# --- Tokenization and Dataset ---

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        print(f"Tokenizing {len(texts)} texts with max_length {max_length}...")
        count = 0
        for text in texts:
            # Encode without special tokens for LM task where context flows
            tokenized = tokenizer.encode(text, add_special_tokens=False, truncation=False) # No truncation here

            # Chunk into sequences of max_length + 1
            for i in range(0, len(tokenized) - 1, max_length): # Ensure we have target
                chunk = tokenized[i : i + max_length + 1]
                if len(chunk) < 2: continue # Need at least input and target

                input_ids = chunk[:-1]
                target_ids = chunk[1:]

                # Pad the last chunk if necessary
                input_padding_length = max_length - len(input_ids)
                target_padding_length = max_length - len(target_ids)

                input_ids = input_ids + [tokenizer.pad_token_id] * input_padding_length
                target_ids = target_ids + [-100] * target_padding_length # Pad targets with -100

                self.examples.append({
                    "input_ids": torch.tensor(input_ids[:max_length], dtype=torch.long), # Ensure max_length
                    "target_ids": torch.tensor(target_ids[:max_length], dtype=torch.long) # Ensure max_length
                })
                count += 1
        print(f"Created {count} training examples.")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

if __name__ == '__main__':
    PATH = "titansA0.pth"
        
    # --- Configuration ---
    # Define the directory containing your text files
    CORPUS_DIRECTORY = "/notebooks/trainingdata__text/" # <--- CHANGE THIS to your actual directory path

    
    # --- Hyperparameters ---
    VOCAB_SIZE = 30522 # Placeholder, will be updated by tokenizer
    EMBED_DIM = 128
    LMM_KEY_DIM = 64
    LMM_VALUE_DIM = 64
    LMM_QUERY_DIM = LMM_KEY_DIM # *** ENSURE THIS MATCHES ***
    LMM_MEM_HIDDEN = [256]
    LMM_MEM_DEPTH = len(LMM_MEM_HIDDEN) + 1 if LMM_MEM_HIDDEN else 0 # Depth is layers in MLP
    LMM_USE_CONV = False
    LMM_LEARN_GATES = True
    MAX_SEQ_LEN = 64
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 350
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_EVALUATION_PREDICTED_TOKENS = 30
    DUMMY_CORPUS_FILE = "dummy_text_corpus.txt" # File for dummy data

    print(f"Using device: {DEVICE}")
    print(f"LMM_QUERY_DIM set to {LMM_QUERY_DIM}")
    print(f"Anomaly Detection Enabled: {torch.is_anomaly_enabled()}") # Confirm it's on

    # --- Tokenizer ---
    tokenizer_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    VOCAB_SIZE = tokenizer.vocab_size
    if tokenizer.pad_token is None:
        print("Adding PAD token to tokenizer.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        VOCAB_SIZE += 1
    print(f"Tokenizer loaded: {tokenizer_name}, Vocab size: {VOCAB_SIZE}")
    print(f"Pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    
    '''
    # --- Dummy Data ---
    if os.path.exists(DUMMY_CORPUS_FILE):
        try:
            with open(DUMMY_CORPUS_FILE, "r", encoding="utf-8") as f:
                 dummy_texts = f.readlines()
            print(f"Loaded {len(dummy_texts)} lines from {DUMMY_CORPUS_FILE}")
            if len(dummy_texts) < 50:
                 print("Warning: Dummy corpus is very small.")
        except Exception as e:
            print(f"Error loading {DUMMY_CORPUS_FILE}: {e}. Using inline data.")
            dummy_texts = [] # Fallback to inline if error
    else:
        print(f"{DUMMY_CORPUS_FILE} not found, using inline dummy data.")
        dummy_texts = []

    if not dummy_texts: # Use inline data if file not found or empty/error
        dummy_texts = [
            "The study of long-term memory involves understanding how information is encoded, stored, and retrieved over extended periods.",
            "Neural networks provide a powerful framework for modeling cognitive processes, including memory.",
            "Recurrent neural networks (RNNs) were early models used for sequence processing, but suffer from vanishing gradients.",
            "Transformers, with their attention mechanisms, have become state-of-the-art for many natural language processing tasks.",
            "The Titans paper introduces a novel long-term memory module designed to be updated during the forward pass.",
            "This online learning mechanism allows the memory to adapt to incoming data streams at inference time.",
            "The memory update rule is inspired by gradient descent with momentum and weight decay on an associative loss.",
            "Keys, values, and queries are projected from the input sequence, similar to attention mechanisms.",
            "An internal memory network, often an MLP, maps keys to predicted values.",
            "The difference between the predicted value and the actual value drives the memory update.",
            "Momentum helps to smooth the updates and accelerate learning in consistent directions.",
            "Weight decay, controlled by the alpha gate, acts as a forgetting mechanism, preventing memory saturation.",
            "Data-dependent gates allow the update dynamics (learning rate, momentum decay, forgetting) to adapt based on the input context.",
            "Implementing the parallel scan version of the update rule can significantly improve performance but requires specialized techniques.",
            "This recurrent implementation focuses on clarity and demonstrates the core update logic step-by-step.",
            "Training involves learning the initial parameters of the projections, gates, and the memory network.",
            "During inference, these learned initial parameters are dynamically modified by the LMM's internal update rule.",
            "The retrieved memory output can then be used by subsequent layers in a larger architecture.",
            "Potential applications include lifelong learning, continual adaptation, and processing very long sequences.",
            "Evaluating such models requires tasks that specifically test long-range dependencies and adaptation.",
            "Tokenization converts raw text into a sequence of numerical IDs that the model can process.",
            "Embeddings map these discrete token IDs into continuous vector representations.",
            "Positional encodings are added to provide the model with information about the order of tokens in the sequence.",
            "The final layer typically maps the model's internal representation back to logits over the vocabulary.",
            "Cross-entropy loss is commonly used for training language models by comparing predicted logits to target token IDs.",
            "Optimization algorithms like AdamW adjust the model's trainable parameters to minimize the loss.",
            "Hyperparameter tuning, including learning rate, batch size, and model dimensions, is crucial for good performance.",
            "Gradient clipping can help stabilize training by preventing exploding gradients.",
            "Evaluation metrics for language models often include perplexity, which measures how well the model predicts a sample of text.",
            "Generating text involves iteratively predicting the next token based on the preceding sequence and the model's internal state."
        ] * 5 # Repeat inline data if used

    # --- Dataset and DataLoader ---
    train_dataset = TextDataset(dummy_texts, tokenizer, max_length=MAX_SEQ_LEN)
    # Handle potential empty dataset
    if len(train_dataset) == 0:
        raise ValueError("Created dataset is empty. Check dummy data source and tokenization.")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    '''
    
    
    
    
    
    
    

    # --- Load Data from Directory ---
    all_texts = []
    if os.path.isdir(CORPUS_DIRECTORY):
        print(f"Attempting to load text files from directory: {CORPUS_DIRECTORY}")
        # Find all files ending with .txt in the specified directory
        # Use recursive=True if you want to search subdirectories as well
        # text_files = glob.glob(os.path.join(CORPUS_DIRECTORY, '**/*.txt'), recursive=True) # For recursive search
        text_files = glob.glob(os.path.join(CORPUS_DIRECTORY, '*.txt')) # Non-recursive search

        if not text_files:
            print(f"Warning: No '.txt' files found in {CORPUS_DIRECTORY}.")
        else:
            print(f"Found {len(text_files)} '.txt' files.")
            for file_path in text_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        # Read lines from each file and add them to the list
                        # Use readlines() if your TextDataset expects a list of lines
                        # Use f.read() if your TextDataset expects one large string per file
                        #   and handles splitting/chunking internally.
                        lines = f.readlines()
                        # Optional: Strip leading/trailing whitespace from each line
                        lines = [line.strip() for line in lines if line.strip()]
                        all_texts.extend(lines)
                    # print(f"Successfully read {len(lines)} lines from {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}. Skipping.")

            if all_texts:
                print(f"Loaded a total of {len(all_texts)} lines from {len(text_files)} files.")
                if len(all_texts) < 100: # Adjust threshold as needed
                     print("Warning: Combined corpus from directory is very small.")
            else:
                 print(f"Warning: Found {len(text_files)} files, but couldn't load any valid text lines.")

    else:
        print(f"Directory '{CORPUS_DIRECTORY}' not found.")



    # --- Dataset and DataLoader (using the loaded or fallback data) ---
    # Make sure TextDataset, tokenizer, MAX_SEQ_LEN, DataLoader, BATCH_SIZE are defined elsewhere
    # Example placeholder definitions if needed:
    # class TextDataset: ...
    # tokenizer = ...
    # MAX_SEQ_LEN = 512
    # BATCH_SIZE = 4
    # from torch.utils.data import DataLoader, Dataset # Assuming PyTorch

    train_dataset = TextDataset(all_texts, tokenizer, max_length=MAX_SEQ_LEN)

    # Handle potential empty dataset AFTER attempting to load/fallback
    if len(train_dataset) == 0:
        raise ValueError("Created dataset is empty. Check data sources (directory/fallback) and tokenization.")

    print(f"Successfully created dataset with {len(train_dataset)} items.")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("DataLoader created.")

    
    loadModel = True
    
    
    # --- Model Instantiation ---
    model = LMModelWithLMM(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            lmm_key_dim=LMM_KEY_DIM,
            lmm_value_dim=LMM_VALUE_DIM,
            lmm_query_dim=LMM_QUERY_DIM, # Ensure this matches key_dim
            lmm_memory_hidden_dims=LMM_MEM_HIDDEN,
            lmm_memory_depth=LMM_MEM_DEPTH,
            lmm_use_convolution=LMM_USE_CONV,
            lmm_learn_gates=LMM_LEARN_GATES,
            max_seq_len=MAX_SEQ_LEN
        ).to(DEVICE)
    
    
    if loadModel:
        
        model.load_state_dict(torch.load(PATH))  # Load the state dictionary
    
    
    else:
        

        # Resize embeddings if needed (after moving model to device)
        if tokenizer.pad_token_id >= model.embedding.num_embeddings:
             print("Resizing model embeddings and output layer for added pad token.")
             model.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=tokenizer.pad_token_id).to(DEVICE)
             # Also adjust output layer if tied or needs resizing
             model.output_layer = nn.Linear(LMM_QUERY_DIM, VOCAB_SIZE).to(DEVICE)

        # Initialize weights (optional but good practice)
        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                 torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu') # Or appropriate init
                 if module.bias is not None:
                     torch.nn.init.zeros_(module.bias)

        model.apply(init_weights)
        print("Model weights initialized.")


        print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # --- Optimizer and Loss ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignore padding in targets

    # --- Training Loop ---
    model.train()
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        num_batches = 0
        # Get initial M_0 parameters sum for comparison (these are the ones optimized)
        initial_lmm_params_sum = sum(p.clone().sum().item() for p in model.lmm.memory_net_initial.parameters())

        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)

            # --- Forward Pass ---
            # State is handled functionally within LMM forward
            logits, _ = model(input_ids, initial_lmm_state=None) # Ignore final state during training batches

            # --- Loss Calculation ---
            loss = criterion(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))

            # --- Backward Pass and Optimization ---
            optimizer.zero_grad()
            # Use anomaly detection during backward (already enabled globally)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # Updates initial LMM params (M_0), projections, gates, embeddings, output layer

            total_loss += loss.item()
            num_batches += 1

            if (i + 1) % 20 == 0 or i == len(train_dataloader) - 1: # Print progress and on last batch
                 print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        # Get final M_0 parameters sum
        final_lmm_params_sum = sum(p.clone().sum().item() for p in model.lmm.memory_net_initial.parameters())
        print(f"--- Epoch {epoch+1} Finished --- Average Loss: {avg_loss:.4f} ---")
        print(f"Initial LMM params (M_0) sum changed from {initial_lmm_params_sum:.4f} to {final_lmm_params_sum:.4f}")
        # This change reflects the optimizer's updates to the *initial* memory state

        
        # store
        torch.save(model.state_dict(), PATH)
        print(f"Model state_dict saved to {PATH}")

    print("--- Training finished. ---")
    # Disable anomaly detection for generation if desired (can slow things down)
    # torch.autograd.set_detect_anomaly(False)
    # print(f"Anomaly Detection Disabled: {not torch.is_anomaly_enabled()}")

    # --- Generation / Evaluation ---
    model.eval()
    prompt = 'A frog is'
    print(f"\n--- Generating {N_EVALUATION_PREDICTED_TOKENS} tokens ---")
    print(f"Prompt: '{prompt}'")

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated_ids = list(input_ids)
    current_lmm_state = None # Start with M_0 and zero momentum for generation
    current_input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad(): # Disable gradients for the overall generation process
        for i in range(N_EVALUATION_PREDICTED_TOKENS):
            # The LMM's internal update still runs with enable_grad() context
            logits, current_lmm_state = model(current_input_tensor, initial_lmm_state=current_lmm_state)

            # Get prediction for the *next* token
            next_token_logits = logits[:, -1, :]

            # Sample the next token (Greedy decoding)
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            # Append prediction
            generated_ids.append(next_token_id)

            # Prepare input for the next step
            current_input_tensor = torch.cat([
                current_input_tensor,
                torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)
            ], dim=1)

            # Optional: Handle max sequence length context window
            current_seq_len = current_input_tensor.shape[1]
            if current_seq_len > MAX_SEQ_LEN:
                 print(f"Note: Truncating generation context from {current_seq_len} to {MAX_SEQ_LEN}")
                 # Keep only the last MAX_SEQ_LEN tokens as input context
                 current_input_tensor = current_input_tensor[:, -MAX_SEQ_LEN:]
                 # The LMM state (current_lmm_state) is carried over, allowing it
                 # to potentially retain information from before the truncation.

            # Optional: Stop if an EOS/SEP token is generated
            if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                 print(f"\nEOS token ({next_token_id}) generated at step {i+1}, stopping generation.")
                 break
            if tokenizer.sep_token_id is not None and next_token_id == tokenizer.sep_token_id:
                 print(f"\nSEP token ({next_token_id}) generated at step {i+1}, stopping generation.")
                 break

    # Decode the final sequence
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\n--- Generated Text ---")
    print(generated_text)

    # Optional: Inspect final state
    if current_lmm_state:
         final_mem_params, final_mom_state = current_lmm_state
         print(f"\nLMM final memory state after generation has {len(final_mem_params)} tensors.")
         # print(f"Example final memory param sum: {final_mem_params[0].sum().item():.4f}") # Example inspection
    else:
         print("\nLMM state was not tracked (generation might have been too short or failed).")
