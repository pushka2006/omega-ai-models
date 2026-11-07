# OMEGA GPT-1: Decoder-Only Transformer From Scratch

**Omega GPT-1** is an original, sovereign, decoder-only Transformer language model built entirely from scratch with PyTorch. It features random parameter initialization, a custom Byte Pair Encoding (BPE) tokenizer trained on local corpora, cross-entropy next-token prediction, AdamW optimization with cosine warmup scheduling, checkpoint recovery, and interactive sampling.

---

## 1. Architectural Highlights

- **Autoregressive Decoder-Only Transformer**: Follows modern Pre-LayerNorm GPT architecture.
- **Learned Positional Embeddings**: Combines discrete token representations with learned sequence position embeddings.
- **Causal Multi-Head Self-Attention**: Q, K, V linear projections with lower-triangular causal attention masking (`F.scaled_dot_product_attention` with FlashAttention acceleration when available).
- **GELU Feed-Forward Network**: Two-stage linear expansion ($4 \times d_{model}$) with Gaussian Error Linear Unit gating.
- **Weight Tying**: Shares parameters between the token embedding matrix and the final linear language modeling head.
- **Standardized Initialization**: Normal distribution ($\mu=0, \sigma=0.02$) with residual projection scaling ($\sigma / \sqrt{2 \cdot N_{layers}}$).

---

## 2. Directory Structure

```text
omega-gpt/
├── config/
│   ├── model_config.py          # Architectural hyperparameters & profiles
│   └── training_config.py       # Training hyperparameters, optimizer & schedules
├── data/
│   ├── raw/                     # Raw UTF-8 .txt text files
│   ├── processed/               # Tokenized PyTorch tensor files (.pt)
│   └── README.md
├── tokenizer/
│   ├── train_tokenizer.py       # BPE training from scratch
│   ├── tokenizer_utils.py       # OmegaTokenizer wrapper class & special tokens
│   └── saved/                   # Saved tokenizer.json & vocab.json
├── model/
│   ├── embeddings.py            # Token & learned position embeddings
│   ├── attention.py             # Causal multi-head self-attention
│   ├── transformer_block.py     # Pre-LN Transformer blocks + FFN
│   ├── gpt.py                   # Core OmegaGPT model class
│   └── model_utils.py           # Parameter counters & weight initialization
├── training/
│   ├── dataset.py               # Dataset ingestion, cleaning & chunking
│   ├── optimizer.py             # AdamW builder with selective weight decay
│   ├── scheduler.py             # Cosine learning rate scheduler with warmup
│   ├── checkpoint.py            # Checkpoint manager (latest, best, step)
│   ├── evaluate.py              # Loss and Perplexity evaluation
│   ├── trainer.py               # Production training engine with AMP & logging
│   └── train.py                 # Training CLI runner
├── generation/
│   ├── sampling.py              # Temperature, Top-K, Top-P & repetition penalty
│   ├── generate.py              # TextGenerator inference engine
│   └── interactive_chat.py      # Terminal chat console
├── evaluation/
│   ├── perplexity.py            # Standalone perplexity calculator
│   └── generate_samples.py      # Benchmark qualitative generation
├── scripts/
│   ├── prepare_data.py          # CLI dataset preparation
│   ├── train_tokenizer.py       # CLI tokenizer trainer
│   ├── train_model.py           # CLI model trainer
│   ├── resume_training.py       # CLI resume runner
│   └── generate_text.py         # CLI text generator
├── notebooks/
│   └── omega_gpt_training.ipynb # Google Colab end-to-end training notebook
├── tests/
│   ├── test_tokenizer.py        # Tokenizer unit tests
│   ├── test_model.py            # Architecture unit tests
│   ├── test_training.py         # Training loop & checkpoint tests
│   └── test_generation.py       # Generation & sampling tests
├── outputs/
│   ├── checkpoints/             # Saved model states (.pth)
│   ├── logs/                    # TensorBoard event logs
│   ├── samples/                 # Sample generation snapshots
│   └── graphs/                  # Loss & perplexity visualization curves
├── requirements.txt
├── .gitignore
├── README.md
└── main.py                      # Primary unified CLI entrypoint
```

---

## 3. Configuration Profiles & Parameter Counts

| Profile | Layers | Heads | Embedding ($d_{model}$) | Context Length | Vocab Size | Approx Parameters |
|:-------:|:------:|:-----:|:-----------------------:|:--------------:|:----------:|:-----------------:|
| **Tiny** | 2 | 2 | 128 | 64 | 4,000 | ~1.0M |
| **Small** (Default) | 4 | 4 | 256 | 128 | 8,000 | ~4.7M |
| **Medium** | 8 | 8 | 512 | 256 | 16,000 | ~34.8M |

---

## 4. Quickstart: Step-by-Step Execution

All commands can be run either from `omega-gpt/` or directly from `model 1/`:

### Step 1: Environment Setup
```bash
python main.py setup
```

### Step 2: Train Custom BPE Tokenizer
Trains a Byte Pair Encoding tokenizer strictly from your local raw text in `data/raw/`:
```bash
python main.py train-tokenizer --vocab_size 8000
```

### Step 3: Clean & Tokenize Dataset
Normalizes whitespace, removes duplicates, splits train/validation sets (90/10), and saves token tensors:
```bash
python main.py prepare-data --train_ratio 0.9
```

### Step 4: Run Automated Test Suite
Verifies all components (embeddings, attention, loss, scheduler, checkpoints, generation):
```bash
python main.py test
```

### Step 5: Train the Model From Scratch
```bash
python main.py train --profile small --max_steps 5000 --device auto
```

### Step 6: Resume Training (If Interrupted)
```bash
python main.py resume
```

### Step 7: Evaluate Validation Perplexity
```bash
python main.py evaluate
```

### Step 8: Generate Text
Run single-prompt generation:
```bash
python main.py generate --prompt "The future of artificial intelligence" --tokens 150 --temperature 0.8
```
Or launch interactive terminal mode:
```bash
python main.py generate --interactive
```

---

## 5. Google Colab & GPU Training

An end-to-end Google Colab notebook is provided at `notebooks/omega_gpt_training.ipynb`.
1. Upload the project folder or clone to Google Colab.
2. Under **Runtime > Change runtime type**, select **T4 GPU** or **A100 GPU**.
3. Open and run all cells sequentially.

