# OMEGA MODEL 2 — Advanced AI Assistant

**OMEGA MODEL 2** is a modular personal AI assistant architecture operating as the central intelligence hub in the sovereign OMEGA AI ecosystem. It connects directly to your custom **OMEGA GPT-1** sovereign Transformer model while providing multi-provider routing (OpenAI, Google Gemini, Anthropic Claude, Ollama), 3-tier persistent memory, live web search, semantic vector knowledge base, AST-guarded Python sandbox, and a futuristic glassmorphic UI.

---

## 1. System Architecture

```
                    OMEGA MODEL 2 ECOSYSTEM
                               |
       +-----------------------+-----------------------+
       |                                               |
[Futuristic Web UI]                         [Professional CLI]
(http://127.0.0.1:8000)                      (python cli/omega_cli.py)
       |                                               |
       +-----------------------+-----------------------+
                               |
                   [FastAPI Gateway & Router]
                               |
               +---------------+---------------+
               |                               |
       [10 Cognitive Engines]         [SQLite Persistent Core]
       ├── 1. LLM Router              ├── conversations & messages
       ├── 2. Reasoning Engine        ├── memories (3-tier)
       ├── 3. 3-Tier Memory           ├── documents & vector chunks
       ├── 4. Semantic Retrieval      ├── notes & tasks
       ├── 5. AST Python Sandbox      └── tool execution audit logs
       ├── 6. Web Search Engine
       ├── 7. Coding Assistant
       ├── 8. Tool Center
       ├── 9. Safety Gate
       └── 10. Citation Generator
               |
    +----------+----------+
    |                     |
[Model 1: Omega GPT-1]  [External Providers]
- Step 250 Checkpoint   - OpenAI (GPT-4o)
- BPE Tokenizer (1200)  - Google Gemini
- Local Inference       - Anthropic Claude
                        - Local Ollama
```

---

## 2. The 10 Cognitive Engines

1. **Language Model Engine (`backend/brain/llm_engine.py`)**: Unified model router connecting to local sovereign models and cloud APIs.
2. **Model 1 Sovereign Adapter (`backend/brain/omega1_adapter.py`)**: Direct interface loading Model 1's `omega_gpt1_best.pth` and BPE tokenizer for local inference.
3. **Multi-Step Reasoning Engine (`backend/brain/reasoning_engine.py`)**: Intent classification, tool triggers, and Chain-of-Thought decomposition.
4. **3-Tier SQLite Memory Engine (`backend/brain/memory_engine.py`)**:
   - *Short-Term Memory*: Active session context.
   - *Long-Term Memory*: User preferences, technical facts, and importance ranking.
   - *Project Memory*: Codebase architecture rules and development history.
5. **Knowledge Retrieval Engine (`backend/brain/retrieval_engine.py`)**: Ingestion of TXT, MD, JSON, CSV, PDF, and code files with semantic term-frequency vector chunking.
6. **Web Search Engine (`backend/brain/search_engine.py`)**: DuckDuckGo live search, webpage text cleaning, and verifiable URL citations.
7. **Coding Assistant & Sandbox (`backend/brain/coding_engine.py`)**: AST-guarded sandboxed Python execution with timeout and memory boundaries.
8. **Tool Execution Engine (`backend/brain/tool_engine.py`)**: Extensible registry featuring calculator, system datetime, file I/O, search, and JSON processor.
9. **Safety & Permission Gate (`backend/brain/safety_engine.py`)**: Directory traversal blocking, dangerous call prevention, and sensitive action confirmation.
10. **Response Synthesis Engine (`backend/brain/generator_engine.py`)**: Prompt assembly with memory, document citations, and response formatting.

---

## 3. Quick Start Guide (Windows PowerShell)

### Step 1: Open PowerShell in Model 2 Directory
```powershell
cd "c:\Users\Pushkar\OneDrive\Documents\ai models\model 2 ai assistant"
```

### Step 2: Run Automated Tests
```powershell
python tests/test_all.py
```
*(All 7 test suites verify memory, sandbox security, tool execution, knowledge vectors, and Model 1 adapter).*

### Step 3: Launch Web Assistant UI
```powershell
.\start_omega.ps1
```
Open your browser at: **`http://127.0.0.1:8000`**

### Step 4: Launch Terminal CLI (Alternative)
```powershell
.\start_cli.ps1
```

---

## 4. CLI Commands

Inside the terminal interface (`python cli/omega_cli.py`):

| Command | Purpose |
| :--- | :--- |
| `/model <name>` | Switch active model (`omega_offline`, `omega_gpt1`, `openai`, `gemini`) |
| `/models` | List all registered model providers and status |
| `/search <query>` | Perform live DuckDuckGo web search with citations |
| `/run <code>` | Execute Python code inside the AST-guarded sandbox |
| `/memory` | Inspect stored short-term and long-term memories |
| `/remember <text>` | Save a permanent fact or preference to long-term memory |
| `/status` | View system diagnostics and hardware telemetry |
| `/clear` | Clear the current conversation context |
| `/exit` | Exit the CLI |

---

## 5. Security & Privacy Safeguards

- **Zero Hardcoded Keys**: API keys are read strictly from environment variables (`.env`).
- **AST Sandbox Guard**: Blocks `os.system`, `subprocess`, `rmtree`, and network socket calls.
- **Offline By Default**: Operates 100% offline using `omega_offline` and local `omega_gpt1` without requiring paid accounts.
- **Audited Tools**: Every tool call is timestamped and recorded in the SQLite `tool_logs` table.

