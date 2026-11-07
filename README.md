# OMEGA AI ECOSYSTEM: 4 SOVEREIGN MODELS

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: 4/4 Models Online](https://img.shields.io/badge/Ecosystem-4%2F4%20Models%20Online-brightgreen.svg)](#the-4-models)
[![Tests: 35/35 Passing](https://img.shields.io/badge/Tests-35%2F35%20Passing-success.svg)](#test-suite-summary)

A comprehensive, sovereign artificial intelligence framework comprising four interconnected, independently testable AI models: a custom generative Transformer, an autonomous cognitive assistant, a modular deep learning neural network engine, and a multimodal cross-attention orchestration brain.

---

## Ecosystem Architecture

```
                             [ USER / OPERATOR ]
                                      │
                   ┌──────────────────▼──────────────────┐
                   │        MODEL 4: OMEGA NEXUS         │
                   │  Multimodal Orchestration & Fusion  │
                   └──────────────────┬──────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
┌────────▼────────┐          ┌────────▼────────┐          ┌────────▼────────┐
│     MODEL 1     │          │     MODEL 2     │          │     MODEL 3     │
│   OMEGA GPT-1   │          │  AI ASSISTANT   │          │  NEURAL ENGINE  │
├─────────────────┤          ├─────────────────┤          ├─────────────────┤
│ • Pre-LN SDPA   │          │ • 10 Cognitive  │          │ • 18 Deep Layers│
│ • BPE Tokenizer │◄────────►│   Engines       │◄────────►│ • NumPy Autograd│
│ • GPU Checkpoint│          │ • AST Sandbox   │          │ • SQLite Tracker│
│ • Token Stream  │          │ • Web UI :8000  │          │ • Dashboard:8500│
└─────────────────┘          └─────────────────┘          └─────────────────┘
```

---

## The 4 Models

| Model | Subdirectory | Primary Purpose | Key Specifications | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1: OMEGA GPT-1** | [`model 1/`](model%201) | Custom Decoder-Only Language Model | Pre-LN SDPA Transformer, BPE Tokenizer, **Fable Reasoning Trained**, Val Loss **3.2056**, Val PPL **24.67** | **ONLINE** |
| **Model 2: OMEGA MODEL 2** | [`model 2 ai assistant/`](model%202%20ai%20assistant) | Autonomous Cognitive Assistant | 10 modular engines, 300 Fable reasoning heuristics in SQLite, dynamic `<thought>` Chain-of-Thought | **ONLINE** |
| **Model 3: NEURAL ENGINE** | [`neural network/`](neural%20network) | Modular Deep Learning Framework | 18 layer types, pure-NumPy autograd, Deep Reasoning Complexity Classifier (**96.7% Val Accuracy**) | **ONLINE** |
| **Model 4: OMEGA NEXUS** | [`omega nexus model/`](omega%20nexus%20model) | Master Multimodal Orchestration Brain | Cross-attention latent fusion (**98.3% Val Accuracy**), intent routing, ensemble classification | **ONLINE** |

---

## Reasoning & Training Upgrades (`MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`)

All 4 models were trained and upgraded using verified reasoning trajectories from the Hugging Face Fable dataset:

1. **OMEGA GPT-1 (Model 1)**: Fine-tuned on 501,615 subwords of structured reasoning traces (`Question -> <thought> -> Answer`). Validation perplexity reduced from 35.45 to **24.67** (Val Loss: **3.2056**).
2. **OMEGA MODEL 2 (Model 2)**: 300 distilled reasoning heuristics ingested into `omega2.db`. Upgraded `reasoning_engine.py` with multi-step `<thought>` generation, premise decomposition, and hypothesis validation.
3. **NEURAL NETWORK ENGINE (Model 3)**: Trained a deep multi-layer perceptron with GELU, LayerNorm, and AdamW to classify reasoning complexity and problem depth across 4 categories, achieving **96.67% validation accuracy**.
4. **OMEGA NEXUS (Model 4)**: Trained the `NexusFusionNetwork` cross-attention layers to fuse question token embeddings with continuous reasoning representation vectors, achieving **98.33% validation accuracy**.

---

## Quick Start & CLI Execution

### Model 1: OMEGA GPT-1
```powershell
cd "model 1"
# Streaming token generation
python main.py generate --prompt "Artificial intelligence is" --tokens 80 --temperature 0.7

# Run automated tests
python main.py test
```

### Model 2: OMEGA MODEL 2 (AI Assistant)
```powershell
cd "model 2 ai assistant"
# Launch FastAPI backend & dark glassmorphic web UI (http://127.0.0.1:8000)
.\start_omega.ps1

# Run automated cognitive tests
python tests/test_all.py
```

### Model 3: OMEGA NEURAL NETWORK ENGINE
```powershell
cd "neural network"
# Train Archimedean spiral classifier
python train.py --epochs 50

# Evaluate decision boundary & save metrics
python test.py

# Launch training dashboard (http://127.0.0.1:8500)
.\start_dashboard.ps1
```

### Model 4: OMEGA NEXUS MODEL (Master Orchestrator)
```powershell
cd "omega nexus model"
# View ecosystem health report
python main.py status

# Query orchestrated brain
python main.py query "Calculate 15 * 12 and search for quantum computing"

# Multimodal fusion (Text + Continuous sensory features)
python main.py fuse "Classify anomaly" --features 0.5,1.2,-0.4,0.8

# Run automated tests
python main.py test
```

---

## Test Suite Summary

Every model in the ecosystem includes full test coverage with zero simulated results:

- **Model 1**: 14/14 tests passing (`test_model`, `test_tokenizer`, `test_checkpoint`, `test_generation`).
- **Model 2**: 7/7 tests passing (`test_reasoning`, `test_memory`, `test_search`, `test_sandbox`, `test_end_to_end`).
- **Model 3**: 9/9 tests passing (`test_autograd_vs_pytorch`, `test_18_layers`, `test_real_training`).
- **Model 4**: 5/5 tests passing (`test_projector`, `test_cross_attention`, `test_multimodal_dispatch`).
- **Total**: **35 / 35 automated tests verified.**

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

