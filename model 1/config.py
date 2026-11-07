"""Omega GPT-1 Configuration for Model 1.

Defines model architecture dimensions, training hyperparameters, and file paths.
All parameters match the production Omega GPT-1 Small profile.
"""

import os
import torch

# Compute device detection (CUDA GPU with automatic CPU fallback)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# Model Architecture Specifications
# ==============================================================================
vocab_size = 1200         # BPE vocabulary size (trained subwords)
block_size = 128          # Context window / maximum sequence length
context_length = 128      # Alias for block_size
n_embd = 256              # Embedding dimension (d_model)
d_model = 256             # Alias for n_embd
n_head = 4                # Number of multi-head attention heads
num_heads = 4             # Alias for n_head
n_layer = 4               # Number of Transformer blocks
num_layers = 4            # Alias for n_layer
d_ff = 1024               # Feed-forward hidden dimension (4 * d_model)
dropout = 0.1             # Dropout probability
activation = "gelu"       # Feed-forward activation ('gelu' or 'swiglu')
tie_weights = True        # Tie token embedding and LM projection head weights
tie_word_embeddings = True # Alias for tie_weights
initializer_range = 0.02  # Weight initialization standard deviation

# ==============================================================================
# Training Hyperparameters
# ==============================================================================
batch_size = 32           # Micro-batch size per training step
learning_rate = 5e-4      # Peak learning rate for AdamW optimizer
min_lr = 5e-5             # Minimum learning rate at the end of cosine decay
warmup_iters = 50         # Number of linear warmup steps
max_iters = 1000          # Total training iterations
eval_interval = 50        # Evaluate validation loss every N iterations
eval_iters = 20           # Number of batches to sample during evaluation
grad_clip = 1.0           # Maximum gradient norm for clipping
weight_decay = 0.01       # Decoupled weight decay for AdamW (applied only to 2D weights)
betas = (0.9, 0.95)       # AdamW beta coefficients

# ==============================================================================
# File and Directory Paths
# ==============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "training.txt")
raw_data_dir = os.path.join(base_dir, "data", "raw")
tokenizer_dir = os.path.join(base_dir, "tokenizer_saved")
checkpoint_dir = os.path.join(base_dir, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "omega_gpt1_best.pth")
