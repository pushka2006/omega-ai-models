# Omega GPT-1 Dataset Directory

This directory stores datasets for training the Omega GPT-1 model and tokenizer.

## Structure
- `raw/`: Place your raw UTF-8 `.txt` text files here (e.g. books, articles, code, conversations).
- `processed/`: Automatically populated during `python main.py prepare-data`.
  - `train_tokens.pt`: PyTorch tensor of token IDs for training.
  - `val_tokens.pt`: PyTorch tensor of token IDs for validation.
  - `dataset_metadata.json`: Dataset metrics (document count, char count, token count, split sizes).

## Adding Custom Data
1. Add one or more `.txt` files to `data/raw/`.
2. Run `python main.py prepare-data` to clean, normalize, tokenize, and split the corpus.

