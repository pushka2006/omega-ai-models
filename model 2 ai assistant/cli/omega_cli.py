"""OMEGA MODEL 2: Professional Cyberpunk Terminal Interface.

Interactive terminal assistant with real-time model indicators,
commands (/model, /models, /search, /run, /memory, /remember, /status, /clear),
tool execution displays, and response rendering.
"""

import os
import sys
import time

# Ensure model 2 root is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.brain.coding_engine import coding_engine
from backend.brain.conversation_manager import conversation_manager
from backend.brain.generator_engine import generator_engine
from backend.brain.llm_engine import llm_engine
from backend.brain.memory_engine import memory_engine
from backend.brain.omega1_adapter import omega1_adapter
from backend.brain.search_engine import search_engine

BANNER = r"""
   ___  __  __ ___ ___   _     __  __  ___  ___  ___ _      ___ 
  / _ \|  \/  | __/ __| /_\   |  \/  |/ _ \|   \| __| |    |_  )
 | (_) | |\/| | _| (_ |/ _ \  | |\/| | (_) | |) | _|| |__   / / 
  \___/|_|  |_|___\___/_/ \_\ |_|  |_|\___/|___/|___|____| /___|
                  [ ADVANCED AI ASSISTANT ]
"""


def print_banner():
    print("=" * 70)
    print(BANNER)
    print("  OMEGA ARCHITECTURE: 10-Tier Modular Brain | SQLite 3-Tier Memory")
    print("  SOVEREIGN LOCAL ENGINE: Omega GPT-1 Connected")
    print("=" * 70)


def print_help():
    print("\n[AVAILABLE CLI COMMANDS]")
    print("  /model <name>       Switch active provider (omega_offline, omega_gpt1, openai, gemini)")
    print("  /models             List all AI models and check status")
    print("  /search <query>     Perform live web search with citations")
    print("  /run <code>         Execute Python code in secure sandbox")
    print("  /memory             View stored short-term and long-term memories")
    print("  /remember <text>    Save a fact permanently into long-term memory")
    print("  /status             Display system diagnostics, telemetry, and device info")
    print("  /clear              Clear current conversation history")
    print("  /help               Show this command list")
    print("  /exit               Quit OMEGA assistant\n")


def run_cli():
    print_banner()

    conv_id = conversation_manager.create_conversation("Terminal Session")
    current_provider = "omega_offline"
    web_search_toggle = False

    # Check Model 1 state
    m1_status = "Available" if os.path.exists(omega1_adapter.metadata["checkpoint_path"]) else "Offline"

    print(f"  ● Active Engine      : {current_provider}")
    print(f"  ● Omega GPT-1 Model  : {m1_status}")
    print(f"  ● Web Search Toggle  : {'ON' if web_search_toggle else 'OFF'}")
    print("Type your message or /help for commands. Press Ctrl+C or type /exit to quit.\n")

    while True:
        try:
            prompt = input(f"OMEGA [{current_provider}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting OMEGA MODEL 2.")
            break

        if not prompt:
            continue

        if prompt.lower() in ["/exit", "exit", "/quit", "quit"]:
            print("Session terminated.")
            break

        if prompt.lower() in ["/help", "help"]:
            print_help()
            continue

        if prompt.startswith("/model "):
            new_prov = prompt.split()[1].strip().lower()
            current_provider = new_prov
            print(f"[Active model provider changed to: '{current_provider}']")
            continue

        if prompt == "/models":
            print("\n--- SUPPORTED AI PROVIDERS ---")
            models = llm_engine.list_models()
            for m in models:
                avail = "✔ ONLINE" if m["is_available"] else "✖ OFFLINE"
                print(f"  - {m['id']:<15} | {m['name']:<30} | {avail} | {m['description']}")
            print()
            continue

        if prompt.startswith("/search "):
            q = prompt[8:].strip()
            print(f"\n[Searching the web for: '{q}']...")
            res = search_engine.search_and_synthesize(q, max_results=3)
            if not res["found"]:
                print("No results found.")
            else:
                for c in res["citations"]:
                    print(f"  [{c['index']}] {c['title']}")
                    print(f"      Source: {c['source']} | URL: {c['url']}")
            print()
            continue

        if prompt.startswith("/run "):
            code = prompt[5:].strip()
            print("\n[Executing in isolated sandbox]...")
            out = coding_engine.execute_python_sandbox(code)
            if out["success"]:
                print(f"Output ({out['execution_time_ms']}ms):\n{out['output']}")
            else:
                print(f"Error ({out['execution_time_ms']}ms):\n{out['error']}")
            print()
            continue

        if prompt == "/memory":
            print("\n--- STORED MEMORIES ---")
            mems = memory_engine.list_all(limit=10)
            if not mems:
                print("No memories stored yet.")
            for m in mems:
                print(f"  [{m['memory_type'].upper()} - {m['category']}] (Score: {m['importance_score']}): {m['content']}")
            print()
            continue

        if prompt.startswith("/remember "):
            fact = prompt[10:].strip()
            mem_id = memory_engine.add_memory(
                memory_type="long_term",
                category="user_fact",
                content=fact,
                importance_score=0.9,
            )
            print(f"[Saved to Long-Term Memory (ID: {mem_id[:8]}...)]")
            continue

        if prompt == "/status":
            print("\n--- SYSTEM DIAGNOSTICS ---")
            print(f"  Platform      : {sys.platform}")
            print(f"  Python        : {sys.version.split()[0]}")
            print(f"  Model 1 Path  : {omega1_adapter.metadata['checkpoint_path']}")
            print(f"  Active Session: {conv_id}")
            print()
            continue

        if prompt == "/clear":
            conv_id = conversation_manager.create_conversation("New Terminal Session")
            print("[Conversation context cleared.]\n")
            continue

        # Process standard query
        print("\nThinking...", end="", flush=True)
        t0 = time.time()
        result = generator_engine.process_query(
            query=prompt,
            conversation_id=conv_id,
            provider=current_provider,
            use_web_search=web_search_toggle,
            use_knowledge_base=True,
        )
        elapsed = round(time.time() - t0, 2)
        print(f"\r{' ' * 15}\r", end="")  # Clear thinking indicator

        print(f"\n[OMEGA Assistant - {result['model']}] ({result['latency_ms']}ms)")
        print("-" * 65)
        print(result["response"])
        print("-" * 65 + "\n")


if __name__ == "__main__":
    run_cli()

