import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer # Use AutoTokenizer for flexibility
import os
import glob
import time

# generated with LLM Gemini 2.5 pro
# https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%2218cbBaZ8GLyZ0KxgwTen7lYZu0MbPV-5x%22%5D,%22action%22:%22open%22,%22userId%22:%22114048892711590756388%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing, https://drive.google.com/file/d/1OopYks7pciCHUnNN1MImH6bFlaZuiDRe/view?usp=sharing, https://drive.google.com/file/d/1vTv1WFCdfQcSPxPb_y2wGcIlgsk7pWpl/view?usp=sharing
class RASM(nn.Module):
    """
    Conceptual Implementation of Routed Adaptive State-Space Memory (RASM).
    Combines an SSM backbone with adaptive dynamics, conditional computation
    routing, surprise-modulated memory management, and persistent memory.
    """
    def __init__(self,
                 d_model: int,      # Input/Output dimension (embed_dim)
                 d_state: int,      # Dimension of the SSM state
                 vocab_size: int,   # Size of the vocabulary
                 d_adapt_hid: int = 64, # Hidden dimension for adaptation networks
                 n_persist: int = 4,    # Number of persistent memory tokens
                 adapt_rank: int = 0,   # Rank for low-rank adaptation (0=full) - Conceptual
                 dropout: float = 0.1,
                 padding_idx: int = 0): # Added vocab_size and padding_idx
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_persist = n_persist
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        # --- Input Embedding ---
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # --- Persistent Memory ---
        if n_persist > 0:
            # Initialize persistent memory embeddings directly
            self.persistent_memory_emb = nn.Parameter(torch.randn(n_persist, d_model))

        # --- Core SSM Parameters (Initial / Base) ---
        # A: State transition (d_state, d_state)
        # B: Input transition (d_state, d_model) - Takes embedded input
        # C: Output projection (d_model, d_state) - Projects state back to model dim
        A_init = torch.eye(d_state).unsqueeze(0) # Add batch dim for broadcasting
        B_init = torch.zeros(d_state, d_model).unsqueeze(0)
        self.register_buffer("base_A", A_init)
        self.register_buffer("base_B", B_init)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * (1 / d_state**0.5))

        # --- Adaptation & Gating Networks ---
        d_control_in = d_state + d_model # Input is concat(state, embedded_token)

        self.gate_adapt_router = nn.Sequential(
            nn.Linear(d_control_in, d_adapt_hid), nn.SiLU(),
            nn.Linear(d_adapt_hid, 1)
        )
        self.hypernet_delta_A = nn.Sequential(
            nn.Linear(d_control_in, d_adapt_hid), nn.SiLU(),
            nn.Linear(d_adapt_hid, d_state * d_state),
        )
        self.hypernet_delta_B = nn.Sequential(
            nn.Linear(d_control_in, d_adapt_hid), nn.SiLU(),
            nn.Linear(d_adapt_hid, d_state * d_model),
        )
        self.gate_adapt_rate = nn.Sequential(
            nn.Linear(d_control_in, d_adapt_hid), nn.SiLU(),
            nn.Linear(d_adapt_hid, 1)
        )
        self.gate_forget = nn.Sequential(
            nn.Linear(d_control_in, d_adapt_hid), nn.SiLU(),
            nn.Linear(d_adapt_hid, d_state)
        )
        self.gate_update = nn.Sequential(
            nn.Linear(d_control_in, d_adapt_hid), nn.SiLU(),
            nn.Linear(d_adapt_hid, d_state)
        )

        # --- Output Projection ---
        # Project final model dimension output to vocabulary logits
        self.output_proj = nn.Linear(d_model, vocab_size)

        # --- Optional LayerNorm and Dropout ---
        self.norm_control = nn.LayerNorm(d_control_in)
        self.norm_state = nn.LayerNorm(d_state)
        self.norm_out = nn.LayerNorm(d_model) # Norm before final projection
        self.dropout = nn.Dropout(dropout)

        self._init_weights() # Initialize weights

    def _init_weights(self):
        # Simple initialization scheme
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Special init for embedding? Optional.
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
        if self.padding_idx is not None:
             with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
        if self.n_persist > 0:
             nn.init.normal_(self.persistent_memory_emb, mean=0, std=self.d_model**-0.5)


    def forward(self, input_ids: torch.Tensor, initial_state=None):
        """
        Processes the input sequence of token IDs.

        Args:
            input_ids (torch.Tensor): Input sequence tensor (batch, seq_len)
            initial_state (Optional[torch.Tensor]): Initial hidden state (batch, d_state)

        Returns:
            torch.Tensor: Output logits tensor (batch, seq_len, vocab_size)
            torch.Tensor: Final hidden state (batch, d_state)
        """
        b, l = input_ids.shape

        # 1. Embed Input Tokens
        x = self.embedding(input_ids) # (batch, seq_len, d_model)

        # 2. Prepend Persistent Memory Embeddings
        if self.n_persist > 0:
            persist_mem_batch = self.persistent_memory_emb.unsqueeze(0).expand(b, -1, -1)
            x = torch.cat([persist_mem_batch, x], dim=1)
            l = l + self.n_persist # Update sequence length

        # 3. Initialize State and Adaptive Parameters per batch
        h = torch.zeros(b, self.d_state, device=x.device) if initial_state is None else initial_state
        A = self.base_A.clone().repeat(b, 1, 1)
        B = self.base_B.clone().repeat(b, 1, 1)

        # 4. Iterate through sequence
        outputs = []
        for t in range(l):
            x_t_emb = x[:, t, :] # (batch, d_model) - Use embedded token

            control_input = torch.cat([h, x_t_emb], dim=-1)
            control_input_norm = self.norm_control(control_input)

            adapt_router_logit = self.gate_adapt_router(control_input_norm)
            adapt_prob = torch.sigmoid(adapt_router_logit)

            delta_A_flat = self.hypernet_delta_A(control_input_norm)
            delta_B_flat = self.hypernet_delta_B(control_input_norm)
            delta_A = delta_A_flat.view(b, self.d_state, self.d_state)
            delta_B = delta_B_flat.view(b, self.d_state, self.d_model)

            adapt_rate_logit = self.gate_adapt_rate(control_input_norm)
            adaptation_rate = torch.sigmoid(adapt_rate_logit) * adapt_prob

            A = A + adaptation_rate.unsqueeze(-1) * delta_A
            B = B + adaptation_rate.unsqueeze(-1) * delta_B

            forget_logit = self.gate_forget(control_input_norm)
            forget_gate = torch.sigmoid(forget_logit)

            update_logit = self.gate_update(control_input_norm)
            update_gate = torch.sigmoid(update_logit)

            Ah = torch.einsum('bij,bj->bi', A, h)
            Bx = torch.einsum('bij,bj->bi', B, x_t_emb) # Use embedded token

            h = forget_gate * Ah + update_gate * Bx
            h = self.norm_state(h)

            # Compute output projection from state
            y_t_proj = torch.einsum('oi,bi->bo', self.C, h)
            y_t_proj = self.norm_out(y_t_proj) # Apply norm before dropout/output proj
            y_t_proj = self.dropout(y_t_proj)

            outputs.append(y_t_proj)

        # Stack outputs
        y_proj = torch.stack(outputs, dim=1) # (batch, seq_len_with_persist, d_model)

        # Project to vocabulary
        logits = self.output_proj(y_proj) # (batch, seq_len_with_persist, vocab_size)

        # Remove persistent memory part from the output logits
        if self.n_persist > 0:
            logits = logits[:, self.n_persist:, :] # Return only non-persistent outputs

        return logits, h # Return final state as well


# --- Dataset Definition ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
             raise ValueError("Tokenizer must have a pad_token_id.")

        self.examples = []
        print("Tokenizing and chunking data...")
        for text in texts:
            if not text: continue # Skip empty lines
            tokenized_text = tokenizer.encode(text, add_special_tokens=False)
            # Chunk into sequences of max_length
            for i in range(0, len(tokenized_text) - 1, max_length): # -1 because target is shifted
                chunk = tokenized_text[i : i + max_length + 1] # Get one extra token for target
                if len(chunk) < 2: continue # Need at least one input and one target

                input_chunk = chunk[:-1]
                target_chunk = chunk[1:]

                # Pad input_chunk
                padding_len_input = max_length - len(input_chunk)
                padded_input = input_chunk + [self.pad_token_id] * padding_len_input

                # Pad target_chunk and set padding targets to -100
                padding_len_target = max_length - len(target_chunk)
                padded_target = target_chunk + [-100] * padding_len_target # Use -100 for ignore_index

                self.examples.append({
                    "input_ids": torch.tensor(padded_input, dtype=torch.long),
                    "target_ids": torch.tensor(padded_target, dtype=torch.long)
                })
        print(f"Created {len(self.examples)} examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# --- Main Execution Block ---
if __name__ == '__main__':
    PATH = "rasm_model_v1.pth" # Changed model save path name

    loadModel = True # Set to True to load weights from PATH

    # --- Configuration ---
    CORPUS_DIRECTORY = "/notebooks/trainingdata__text/" # <--- CHANGE THIS if needed

    # --- RASM Hyperparameters ---
    # VOCAB_SIZE will be set by tokenizer
    EMBED_DIM = 256      # Embedding dimension (d_model)
    STATE_DIM = 386      # SSM state dimension (d_state)
    ADAPT_HID_DIM = 128  # Hidden dim for adaptation/gating networks
    NUM_PERSIST = 4      # Number of persistent memory tokens
    DROPOUT = 0.1

    # --- General Training Hyperparameters ---
    MAX_SEQ_LEN = 128 # Adjusted sequence length
    BATCH_SIZE = 32   # Adjusted batch size
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 200 # 500 # Reduced epochs for faster example run
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_EVALUATION_PREDICTED_TOKENS = 50 # Generation length

    print(f"Using device: {DEVICE}")

    # --- Tokenizer ---
    tokenizer_name = "bert-base-uncased" # Or choose another like "gpt2", "roberta-base"
    try:
        # Ensure the directory exists
        if not os.path.exists("./tokenizer_cache"):
            os.makedirs("./tokenizer_cache")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="./tokenizer_cache")
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        print("Please ensure you have internet connection or the model is cached.")
        exit()

    VOCAB_SIZE = tokenizer.vocab_size
    # Add pad token if it doesn't exist (common for GPT-2)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a default pad token. Adding '[PAD]'.")
        # Use add_special_tokens for potential resizing needed
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        VOCAB_SIZE = len(tokenizer) # Update vocab size
        print(f"New vocab size: {VOCAB_SIZE}")

    print(f"Tokenizer loaded: {tokenizer_name}, Vocab size: {VOCAB_SIZE}")
    PAD_TOKEN_ID = tokenizer.pad_token_id
    if PAD_TOKEN_ID is None:
        raise ValueError("Tokenizer failed to set a pad_token_id")
    print(f"Pad token: '{tokenizer.pad_token}', ID: {PAD_TOKEN_ID}")


    # --- Load Data from Directory ---
    all_texts = []
    if os.path.isdir(CORPUS_DIRECTORY):
        print(f"Attempting to load text files from directory: {CORPUS_DIRECTORY}")
        text_files = glob.glob(os.path.join(CORPUS_DIRECTORY, '*.txt'))

        if not text_files:
            print(f"Warning: No '.txt' files found in {CORPUS_DIRECTORY}.")
        else:
            print(f"Found {len(text_files)} '.txt' files.")
            for file_path in text_files:
                try:
                    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                        lines = f.readlines()
                        lines = [line.strip() for line in lines if line.strip() and len(line) > 10] # Basic filter
                        all_texts.extend(lines)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}. Skipping.")

            if all_texts:
                print(f"Loaded a total of {len(all_texts)} lines from {len(text_files)} files.")
            else:
                 print(f"Warning: Found {len(text_files)} files, but couldn't load any valid text lines.")
    else:
        print(f"Directory '{CORPUS_DIRECTORY}' not found. No data loaded.")

    # --- Fallback Dummy Data ---
    if not all_texts:
        print("No data loaded from directory. Using fallback dummy data.")
        all_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming many industries.",
            "Language models learn patterns from vast amounts of text data.",
            "Recurrent neural networks process sequences step by step.",
            "Attention mechanisms allow models to focus on relevant parts of the input.",
            "State space models offer an alternative for sequence modeling.",
            "Adaptive dynamics allow models to change behavior over time.",
            "Training requires a dataset, an optimizer, and a loss function.",
            "Evaluation metrics include perplexity and accuracy.",
            "Generated text should be coherent and relevant to the prompt."
        ] * 50 # Repeat dummy data

    # --- Dataset and DataLoader ---
    train_dataset = TextDataset(all_texts, tokenizer, max_length=MAX_SEQ_LEN)

    if len(train_dataset) == 0:
        print("ERROR: Created dataset is empty. Cannot train. Check data sources and tokenization.")
        exit()

    print(f"Successfully created dataset with {len(train_dataset)} items.")
    # Use pin_memory=True if using GPU for potentially faster data transfer
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  pin_memory=True if DEVICE == torch.device("cuda") else False)
    print("DataLoader created.")

    # --- Model Instantiation ---
    print("\n--- Instantiating RASM Model ---")
    model = RASM(
        d_model=EMBED_DIM,
        d_state=STATE_DIM,
        vocab_size=VOCAB_SIZE,
        d_adapt_hid=ADAPT_HID_DIM,
        n_persist=NUM_PERSIST,
        dropout=DROPOUT,
        padding_idx=PAD_TOKEN_ID
    ).to(DEVICE)

    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- Optional: Load Pre-trained Weights ---
    if loadModel and os.path.exists(PATH):
        try:
            model.load_state_dict(torch.load(PATH, map_location=DEVICE))
            print(f"Loaded pre-trained model weights from {PATH}")
        except Exception as e:
            print(f"Error loading model weights from {PATH}: {e}")
            print("Starting training from scratch.")
    elif loadModel:
        print(f"Load model requested, but file not found at {PATH}. Starting training from scratch.")
    else:
        print("Starting training from scratch (loadModel is False or file not found).")

    # --- Optimizer and Loss ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # Use ignore_index to mask loss on padding targets (-100)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # --- Training Loop ---
    model.train()
    print("\n--- Starting Training ---")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0

        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(DEVICE)   # Shape: [batch_size, seq_len]
            target_ids = batch["target_ids"].to(DEVICE) # Shape: [batch_size, seq_len]

            # --- Forward Pass ---
            # RASM expects [batch, seq_len] input_ids
            logits, _ = model(input_ids) # Ignore final state for loss calculation
            # Logits shape: [batch_size, seq_len, vocab_size]

            # --- Loss Calculation ---
            # Reshape for CrossEntropyLoss:
            # Logits: [batch*seq_len, vocab_size]
            # Target: [batch*seq_len]
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1))

            # --- Backward Pass and Optimization ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (i + 1) % 20 == 0 or i == len(train_dataloader) - 1: # Print progress
                 print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start_time
        print(f"--- Epoch {epoch+1} Finished --- Average Loss: {avg_loss:.4f} --- Time: {epoch_time:.2f}s ---")

        # --- Save Model Checkpoint ---
        try:
            torch.save(model.state_dict(), PATH)
        except Exception as e:
            print(f"Error saving model checkpoint: {e}")

    total_training_time = time.time() - start_time
    print(f"--- Training finished. Total time: {total_training_time:.2f}s ---")

    # --- Generation / Evaluation ---
    model.eval()

    prompts = [
        'A frog is',
        'Large Language Models (LLMs) are artificial intelligence systems',
        'The weather today looks like',
        'To train a neural network, you need'
    ]

    for prompt_text in prompts:
        print(f"\n--- Generating {N_EVALUATION_PREDICTED_TOKENS} tokens ---")
        print(f"Prompt: '{prompt_text}'")

        # Encode prompt
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors='pt').to(DEVICE)
        # input_ids shape: [1, prompt_len]

        generated_ids_list = input_ids.tolist()[0] # Keep track as a list
        current_state = None # Start with no state

        with torch.no_grad():
            for i in range(N_EVALUATION_PREDICTED_TOKENS):
                # Prepare input for the *next* token prediction
                # Use only the last token ID if stateful, or the whole sequence if not
                # For simplicity here, let's feed the whole generated sequence so far
                # (Truncate if needed)
                current_input_tensor = torch.tensor([generated_ids_list], dtype=torch.long).to(DEVICE)
                current_seq_len = current_input_tensor.shape[1]

                if current_seq_len > MAX_SEQ_LEN:
                    # Truncate context fed to the model
                    current_input_tensor = current_input_tensor[:, -MAX_SEQ_LEN:]
                    # Note: State is not explicitly truncated here, assumes model handles long history

                # Forward pass - RASM processes the whole sequence
                # We only care about the *last* output logit for the next token
                logits, current_state = model(current_input_tensor, initial_state=None) # Recompute state each time for simplicity
                # logits shape: [1, generated_len, vocab_size]

                next_token_logits = logits[0, -1, :] # Logits for the last position -> next token

                # --- Sampling ---
                # Greedy decoding:
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()

                # Append prediction
                generated_ids_list.append(next_token_id)

                # Optional: Stop conditions
                if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                     print(f"\nEOS token generated at step {i+1}, stopping generation.")
                     break
                if next_token_id == PAD_TOKEN_ID:
                     print(f"\nPAD token generated at step {i+1}, stopping generation.")
                     break


        # Decode the final sequence
        generated_text = tokenizer.decode(generated_ids_list, skip_special_tokens=True)
        print("\n--- Generated Text ---")
        print(generated_text)
