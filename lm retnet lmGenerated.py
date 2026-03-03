import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random

from toolingConvTxtToTokens2 import TextTrainingCollector, tokenizeText, convTokensToText


# =============================================================================
# 1. RETNET ARCHITECTURE (from previous response)
# =============================================================================

class RetNetConfig:
    def __init__(self, d_model=512, n_layers=4, n_heads=8, vocab_size=10000, ffn_size=None):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        # Paper suggests FFN size is 2 * d_model for RetNet
        self.ffn_size = ffn_size if ffn_size is not None else d_model * 2
        
        # Paper: Head dim for V is 2x head dim for Q/K
        self.qk_dim = d_model // n_heads
        self.v_dim = d_model // n_heads * 2 

def get_retention_decay(n_heads, device):
    """
    Section 3.1: Gamma decay rates.
    gamma = 1 - e^(linspace(log(1/32), log(1/512), h))
    """
    # Log space between 1/32 and 1/512
    s = torch.log(torch.tensor(1/32.0))
    e = torch.log(torch.tensor(1/512.0))
    steps = torch.linspace(s, e, n_heads, device=device)
    gammas = 1 - torch.exp(steps)
    return gammas

class MultiScaleRetention(nn.Module):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.qk_dim = config.qk_dim
        self.v_dim = config.v_dim
        
        # Projections
        self.w_q = nn.Linear(self.d_model, self.qk_dim * self.n_heads, bias=False)
        self.w_k = nn.Linear(self.d_model, self.qk_dim * self.n_heads, bias=False)
        self.w_v = nn.Linear(self.d_model, self.v_dim * self.n_heads, bias=False)
        self.w_g = nn.Linear(self.d_model, self.v_dim * self.n_heads, bias=False)
        self.w_o = nn.Linear(self.v_dim * self.n_heads, self.d_model, bias=False)
        
        # GroupNorm applied to the concatenated head outputs
        self.group_norm = nn.GroupNorm(self.n_heads, self.v_dim * self.n_heads)
        
        # Fixed gammas for decay
        # Register as buffer, but ensure it's on the correct device later
        self.register_buffer("gammas", get_retention_decay(self.n_heads, torch.device('cpu')))

    def get_rotary_embedding(self, seq_len, device):
        """
        Standard RoPE-style complex rotation e^(i * n * theta)
        """
        theta = 10000.0 ** (-2 * (torch.arange(0, self.qk_dim, 2).float() / self.qk_dim))
        theta = theta.to(device)
        position = torch.arange(seq_len, device=device).float()
        angles = torch.outer(position, theta)
        # Shape: (seq_len, qk_dim // 2) -> Complex numbers
        angles = torch.polar(torch.ones_like(angles), angles)
        return angles

    def parallel_retention(self, q, k, v, decay_mask):
        """
        Equation 5: Parallel representation for training.
        """
        # q, k, v shape: (batch, heads, seq_len, dim)
        
        # 1. Calculate Retention Score: R = Q @ K.T
        # (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
        retention = q @ k.transpose(-1, -2) 
        
        # 2. Apply Decay Mask (D) and Causal Mask
        # decay_mask shape: (heads, seq_len, seq_len)
        retention = retention * decay_mask.unsqueeze(0)
        
        # 3. Normalization (Section 2.2)
        # Normalize QK^T by sqrt(d)
        retention = retention / math.sqrt(self.qk_dim)
        
        # 4. Multiply by V
        output = retention @ v
        return output

    def recurrent_retention(self, q, k, v, past_kv=None):
        """
        Equation 6: Recurrent representation for O(1) inference.
        """
        # q, k, v shape: (batch, heads, 1, dim)
        
        # Initialize state if None
        if past_kv is None:
            # State shape: (batch, heads, qk_dim, v_dim)
            past_kv = torch.zeros(q.shape[0], self.n_heads, self.qk_dim, self.v_dim, device=q.device)
        
        # 1. Update State: S_n = gamma * S_{n-1} + K_n^T @ V_n
        # We need to broadcast gamma to (batch, heads, 1, 1)
        gamma = self.gammas.view(1, -1, 1, 1).to(q.device) # Ensure gammas are on correct device
        
        # k.transpose: (B, H, D_k, 1), v: (B, H, 1, D_v) -> (B, H, D_k, D_v)
        current_kv = k.transpose(-1, -2) @ v
        
        current_state = gamma * past_kv + current_kv
        
        # 2. Calculate Output: O_n = Q_n @ S_n
        # q: (B, H, 1, D_k) @ state: (B, H, D_k, D_v) -> (B, H, 1, D_v)
        output = q @ current_state
        
        return output, current_state

    def forward(self, x, past_kv=None, forward_impl='parallel', timestep=None):
        B, T, C = x.shape
        
        # 1. Projections
        q = self.w_q(x).view(B, T, self.n_heads, self.qk_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.qk_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.v_dim).transpose(1, 2)
        
        # 2. Apply Rotary Embeddings (Theta) to Q and K
        # Helper to apply rotation
        def apply_rotary(x_tensor, freqs):
            # View as complex: (B, H, T, D/2)
            x_complex = torch.view_as_complex(x_tensor.float().reshape(*x_tensor.shape[:-1], -1, 2))
            # Broadcast freqs: (1, 1, T, D/2)
            freqs = freqs.view(1, 1, T, -1)
            # Rotate
            x_rotated = x_complex * freqs
            # View back as real
            return torch.view_as_real(x_rotated).flatten(-2)

        if forward_impl == 'parallel':
            rotary_freqs = self.get_rotary_embedding(T, x.device) # (T, D/2)
            q = apply_rotary(q, rotary_freqs)
            k = apply_rotary(k, rotary_freqs.conj()) # Conjugate for K in RetNet formulation
            
            # Create Decay Mask D (Heads, T, T)
            # D_nm = gamma^(n-m) for n >= m
            n_idx = torch.arange(T, device=x.device).unsqueeze(1)
            m_idx = torch.arange(T, device=x.device).unsqueeze(0)
            # Causal mask
            causal_mask = (n_idx >= m_idx).float()
            # Exponent: n - m
            diff = n_idx - m_idx
            # Gammas: (Heads, 1, 1)
            decay_mask = self.gammas.view(-1, 1, 1).to(x.device) ** diff.unsqueeze(0)
            decay_mask = decay_mask * causal_mask.unsqueeze(0)
            
            retention_out = self.parallel_retention(q, k, v, decay_mask)
            current_kv = None # No state to return in parallel mode
        else: # Recurrent
            # For recurrent, T should be 1. Timestep is the absolute position.
            assert T == 1, "Recurrent mode expects T=1 (single token)."
            rotary_freqs = self.get_rotary_embedding(1, x.device) # (1, D/2)
            
            # Adjust rotary freqs based on absolute timestep
            if timestep is not None:
                # Shift position for current token
                position_freqs = self.get_rotary_embedding(timestep + 1, x.device)[timestep:timestep+1]
                q = apply_rotary(q, position_freqs)
                k = apply_rotary(k, position_freqs.conj())
            else: # If no timestep, assume relative position 0 for the single token
                q = apply_rotary(q, rotary_freqs)
                k = apply_rotary(k, rotary_freqs.conj())

            retention_out, current_kv = self.recurrent_retention(q, k, v, past_kv)
        
        # 4. Gating and Output Projection (Equation 8)
        # Reshape back to (B, T, H*V_dim) for GroupNorm
        retention_out = retention_out.transpose(1, 2).contiguous().view(B, T, -1)
        
        # Apply GroupNorm (concatenated heads)
        # PyTorch GroupNorm expects (B, C, T), so we permute
        normed = self.group_norm(retention_out.transpose(1, 2)).transpose(1, 2)
        
        # Swish Gate
        g = F.silu(self.w_g(x))
        
        # Final Output
        y = (g * normed)
        output = self.w_o(y)
        
        if forward_impl == 'recurrent':
            return output, current_kv
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_size):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_size)
        self.w2 = nn.Linear(ffn_size, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.w2(self.gelu(self.w1(x)))

class RetNetBlock(nn.Module):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.msr = MultiScaleRetention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.ffn_size)

    def forward(self, x, past_kv=None, forward_impl='parallel', timestep=None):
        # Pre-LayerNorm
        if forward_impl == 'recurrent':
            msr_out, new_kv = self.msr(self.ln1(x), past_kv, forward_impl=forward_impl, timestep=timestep)
            x = x + msr_out
            x = x + self.ffn(self.ln2(x))
            return x, new_kv
        else:
            x = x + self.msr(self.ln1(x), forward_impl=forward_impl)
            x = x + self.ffn(self.ln2(x))
            return x

# =============================================================================
# 2. GATED STATE SPACE MODEL (Modified for RetNet)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class GatedStateSpaceModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout=0.0):
        super().__init__()
        
        # Create RetNetConfig based on provided parameters
        self.config = RetNetConfig(
            d_model=embed_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            vocab_size=vocab_size,
            ffn_size=embed_dim * 2 # As per RetNet paper's suggestion
        )

        self.token_embedding = nn.Embedding(vocab_size, self.config.d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            RetNetBlock(self.config) for _ in range(self.config.n_layers)
        ])
        
        self.ln_f = RMSNorm(self.config.d_model) # Using RMSNorm as in original code
        self.head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        self.token_embedding.weight = self.head.weight # Weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_kvs=None, forward_impl='parallel', timestep=None):
        B, T = idx.shape
        x = self.token_embedding(idx)
        x = self.dropout(x)
        
        if forward_impl == 'recurrent':
            new_kvs = []
            for i, layer in enumerate(self.layers):
                past_kv = past_kvs[i] if past_kvs is not None else None
                x, new_kv = layer(x, past_kv=past_kv, forward_impl='recurrent', timestep=timestep)
                new_kvs.append(new_kv)
            x = self.ln_f(x)
            return self.head(x), new_kvs
        else: # 'parallel' mode for training
            for layer in self.layers:
                x = layer(x, forward_impl='parallel')
            x = self.ln_f(x)
            return self.head(x), None

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens, temperature=1.0):
        # Initialize past_kvs for recurrent generation
        past_kvs = None
        
        # Process the prompt in parallel first to get initial states
        # This is more efficient than token-by-token for the prompt
        if prompt_tokens.shape[1] > 0:
            # The forward pass for parallel mode doesn't return past_kvs directly.
            # To get the initial states for recurrent generation, we need to run
            # the prompt through the model in a way that accumulates states.
            # For simplicity and to match the recurrent API, we'll process the prompt
            # token by token to build up the initial past_kvs.
            # A more optimized approach would be to have a 'chunkwise' or 'parallel_to_recurrent'
            # mode in RetNet that computes the final state from a sequence.
            
            current_timestep = 0
            for i in range(prompt_tokens.shape[1]):
                token = prompt_tokens[:, i:i+1] # Process one token at a time
                _, past_kvs = self(token, past_kvs=past_kvs, forward_impl='recurrent', timestep=current_timestep)
                current_timestep += 1
        else:
            current_timestep = 0 # Start from timestep 0 if prompt is empty

        # Now generate new tokens recurrently
        for _ in range(max_new_tokens):
            # idx_cond is always the last generated token for recurrent mode
            idx_cond = prompt_tokens[:, -1:] 
            
            # Forward pass in recurrent mode, updating past_kvs
            logits, past_kvs = self(idx_cond, past_kvs=past_kvs, forward_impl='recurrent', timestep=current_timestep)
            
            # Logits are for the single generated token
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            prompt_tokens = torch.cat((prompt_tokens, idx_next), dim=1)
            current_timestep += 1 # Increment timestep for next token
            
        return prompt_tokens

# =============================================================================
# 3. EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_size = 50257
    seq_len = 96 # 64 # Keep small for demo stability
    epochs = 1000000
    
    # Model parameters
    embed_dim = int(512 * 1.5) # 768
    num_heads = 6
    num_layers = 13
    dropout = 0.01

    model = GatedStateSpaceModel(
        vocab_size=vocab_size, 
        embed_dim=embed_dim, 
        num_heads=num_heads,     
        num_layers=num_layers,     
        dropout=dropout
    ).to(device)
    
    checkpoint_path = 'retnetModelB_layer13_checkpoint.pth'
    
    #checkpoint_path = 'retnetModelCsmalldat_layer13_checkpoint.pth'
    

    # ---------------------------------------------------------
    # TOGGLE THIS TO SWITCH BETWEEN TRAINING AND INFERENCE
    # ---------------------------------------------------------
    #MODE = "TRAIN" 
    import sys
    MODE = sys.argv[1]
    
    if MODE == "TRAIN" or MODE == "CONTINUE":
        
        
        # load model
        if MODE == "CONTINUE":
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Checkpoint loaded successfully.")
            except FileNotFoundError:
                print(f"No checkpoint found at {checkpoint_path}. Generating with untrained weights.")

        
        
        
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        
        
        
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.005, lr=3e-4)
        
        
        # Add this after defining optimizer
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs/5000))
        
        
        
        print("\n--- Loading Datasets ---")
        textcollector = TextTrainingCollector()
        
        # --- DATA LOADING LOGIC ---
        real_data_path = '/home/r0b3/root/remoteDest_A/trainingdat_text/'
        #real_data_path = '/home/r0b3/root/remoteDest_A/trainingdat_text/midQuality/'
        
        train_tokens = []
        test_tokens = []
        
        if os.path.exists(real_data_path):
            print(f"Scanning {real_data_path}...")
            try:
                with os.scandir(real_data_path) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.endswith('.txt'):
                            textcollector.readTxtFile(entry.path)
                full_text = textcollector.retStr()
                if len(full_text) > 0:
                    train_tokens = tokenizeText(full_text)
            except Exception as e:
                print(f"Error reading train files: {e}")
        
        
        #train_tokens = train_tokens[:800000]
        
        # Split for test
        split_idx = int(len(train_tokens) * 0.95)
        test_tokens = train_tokens[split_idx:]
        train_tokens = train_tokens[:split_idx]

        # Convert to tensors
        train_document = torch.tensor([train_tokens], dtype=torch.long).to(device)
        test_document = torch.tensor([test_tokens], dtype=torch.long).to(device)
        
        print(f"Train Tokens: {train_document.size(1)}")
        
        
        print("\n--- Starting Training ---")
        
        intSeenTokens = 0
        
        for intHeap in range(10000000000):
            model.train()
            total_train_loss = 0
            steps = 0
            
            # Training Loop
            num_batches = 300 # how many batches per "heap"
            
            for _ in range(num_batches):
                
                
                
                
                # Inside the training loop
                batch_size = 4  # Add this variable
                max_start = train_document.size(1) - seq_len - 1

                # Create a batch of random indices
                ix = torch.randint(0, max_start, (batch_size,))

                # Stack them to get shape (batch_size, seq_len)
                inputs = torch.stack([train_document[0, i : i + seq_len] for i in ix])
                targets = torch.stack([train_document[0, i+1 : i + seq_len + 1] for i in ix])

                logits, _ = model(inputs, forward_impl='parallel')
                
                intSeenTokens += seq_len
                
                
                """
                # Random crop
                max_start = train_document.size(1) - seq_len - 1
                if max_start <= 0: break
                
                i = random.randint(0, max_start)
                
                rngSeqLen = random.randint(3, seq_len)
                
                inputs = train_document[:, i : i + rngSeqLen]
                targets = train_document[:, i+1 : i + rngSeqLen + 1]
                
                # Use parallel mode for training
                logits, _ = model(inputs, forward_impl='parallel')
                """
                
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
                
                
                
                new_lr = 3.0e-4
                if False: # warmup
                    new_lr = min(1.0, intHeap/420) * 3.0e-4
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                #scheduler.step() # must be after optimizer.step()
                
                
                
                
                total_train_loss += loss.item()
                steps += 1
                
                if steps % 30 == 0:
                    print(f"heap {intHeap} | Step {steps} | Loss: {loss.item():.4f}")
            
            avg_train_loss = total_train_loss / max(1, steps)
            floatEpoch = intSeenTokens / float( train_document.size(1) )
            print(f"=== epoch {floatEpoch}/{epochs} | Avg Train Loss: {avg_train_loss:.6f} ===")
            
            # Save Checkpoint
            if intHeap % 2 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'numLayers' : num_layers,
                    'numHeads' : num_heads,
                    'embed_dim' : embed_dim,
                    
                    'seqLen': seq_len, # sequence length of model
                    
                    'intHeap' : intHeap, # number of heap iteration of training
                    'epoch': floatEpoch, # epoch of training
                }, checkpoint_path)
            
            # Validation
            if intHeap % 1 == 0:
                model.eval() 
                total_test_loss = 0
                test_steps = 0
                with torch.no_grad():
                    # Validate on a few chunks
                    for _ in range(10):
                        max_start = test_document.size(1) - seq_len - 1
                        if max_start <= 0: break
                        i = random.randint(0, max_start)
                        
                        inputs = test_document[:, i : i + seq_len]
                        targets = test_document[:, i+1 : i + seq_len + 1]
                        
                        # Use parallel mode for validation
                        logits, _ = model(inputs, forward_impl='parallel')
                        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
                        total_test_loss += loss.item()
                        test_steps += 1
                
                print(f"Test Loss: {total_test_loss / max(1, test_steps):.4f}")

    elif MODE == "INFERENCE":
        print("\n--- Loading Checkpoint for Inference ---")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully.")
        except FileNotFoundError:
            print(f"No checkpoint found at {checkpoint_path}. Generating with untrained weights.")

        model.eval()
        print("\n--- Generating Text ---")
        
        strPrompt = 'a neural network is'
        strPrompt = '<ref>{{cite book |url=' # easy : appears a lot in wikitext
        strPrompt = 'Eliezer: Basically, it’s the original AI alignment organization'
        strPrompt = 'Eliezer: AI alignment is'
        arrPromptTokens = tokenizeText(strPrompt)
        tensorPromptTokens = torch.tensor([arrPromptTokens], dtype=torch.long).to(device)
        
        print(f"Prompt: {strPrompt}")
        
        # Generate
        generated_sequence = model.generate(
            prompt_tokens=tensorPromptTokens, 
            max_new_tokens=30, 
            temperature=0.05
        )
        
        flat_generated_tokens = generated_sequence.tolist()[0]
        strResponseFull = convTokensToText(flat_generated_tokens)
        print(f"\nResult:\n{strResponseFull}")
