"""OMEGA NEXUS MODEL: Master Unified CLI.

Usage:
    python main.py status
    python main.py query "Your query here" [--mode auto|generative|agentic]
    python main.py fuse "Describe image" --features 0.2,0.5,1.1
    python main.py interactive
    python main.py server [--port 8600]
    python main.py test
"""

import argparse
import sys
import unittest
import torch

from core.orchestrator import OmegaNexus
from config import VERSION


def print_banner():
    print(r"""
========================================================================
     ___  __  __ ___ ___   _     _  _ _____  ___   _ ___   __  __  ___  ___  ___ _    
    / _ \|  \/  | __/ __| /_\   | \| | __\ \/ / | | / __| |  \/  |/ _ \|   \| __| |   
   | (_) | |\/| | _| (_ |/ _ \  | .` | _| >  <| |_| \__ \ | |\/| | (_) | |) | _|| |__ 
    \___/|_|  |_|___\___/_/ \_\ |_|\_|___/_/\_\\___/|___/ |_|  |_|\___/|___/|___|____|
                        SOVEREIGN MULTIMODAL INTELLIGENCE
========================================================================
    """)


def run_status():
    print_banner()
    nexus = OmegaNexus()
    status = nexus.get_system_status()
    print("=" * 70)
    print(f"  OMEGA NEXUS MODEL v{status['version']} | Compute Device: {status['device'].upper()}")
    print("=" * 70)
    for key, info in status["models"].items():
        st = info["status"]
        color_tag = "✔" if st == "ONLINE" else "✖"
        print(f"  [{color_tag}] {info['name']:<30} [{st:<7}] - {info['type']}")
    print("=" * 70)


def run_query(prompt: str, mode: str):
    nexus = OmegaNexus()
    res = nexus.dispatch(prompt=prompt, mode=mode)
    print("\n--- OMEGA NEXUS RESPONSE ---")
    print(f"Source: {res.get('source')} (Mode: {res.get('mode')})")
    print(f"Output:\n{res.get('response')}")
    if "metadata" in res and res["metadata"]:
        print(f"\nMetadata: {res['metadata']}")


def run_fuse(text: str, features_str: str):
    nexus = OmegaNexus()
    feat_list = [float(x.strip()) for x in features_str.split(",")]
    feat_tensor = torch.tensor([feat_list], dtype=torch.float32)
    res = nexus.dispatch(prompt=text, features=feat_tensor, mode="multimodal")
    print("\n--- OMEGA NEXUS MULTIMODAL FUSION RESULT ---")
    print(f"Status    : {res.get('response')}")
    print(f"Fused Dim : {res.get('fused_shape')}")
    print(f"Class     : {res.get('predicted_class')} (Confidence: {res.get('confidence')}%)")


def run_interactive():
    print_banner()
    nexus = OmegaNexus()
    print("Interactive Nexus Session. Type 'exit' to quit. Use '/generate <prompt>' for GPT-1.\n")
    while True:
        try:
            inp = input("NEXUS> ").strip()
            if not inp:
                continue
            if inp.lower() in ["exit", "quit"]:
                print("Terminating Nexus session.")
                break
            res = nexus.dispatch(inp)
            print(f"\n[{res.get('source')}]:\n{res.get('response')}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def run_tests():
    from tests.test_nexus import TestOmegaNexus
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOmegaNexus)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def main():
    parser = argparse.ArgumentParser(description="OMEGA NEXUS Master CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    subparsers.add_parser("status", help="Display status of all 4 models in ecosystem")

    query_parser = subparsers.add_parser("query", help="Dispatch prompt across models")
    query_parser.add_argument("prompt", type=str, help="Prompt text")
    query_parser.add_argument("--mode", type=str, default="auto", choices=["auto", "generative", "agentic"])

    fuse_parser = subparsers.add_parser("fuse", help="Fuse multimodal text and features")
    fuse_parser.add_argument("text", type=str, help="Text query")
    fuse_parser.add_argument("--features", type=str, required=True, help="Comma-separated floats")

    subparsers.add_parser("interactive", help="Start interactive Nexus console")
    subparsers.add_parser("test", help="Execute Nexus test suite")

    server_parser = subparsers.add_parser("server", help="Launch FastAPI server")
    server_parser.add_argument("--port", type=int, default=8600, help="Port to listen on")

    args = parser.parse_args()

    if args.command == "status" or args.command is None:
        run_status()
    elif args.command == "query":
        run_query(args.prompt, args.mode)
    elif args.command == "fuse":
        run_fuse(args.text, args.features)
    elif args.command == "interactive":
        run_interactive()
    elif args.command == "test":
        run_tests()
    elif args.command == "server":
        import uvicorn
        uvicorn.run("api.server:app", host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()

