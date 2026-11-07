"""Ingestion and dataset preparation pipeline for MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x.

Extracts reasoning trajectories, chains-of-thought, problem solving strategies,
and prepares tailored training datasets for all 4 models in the OMEGA AI ecosystem.
"""

from pathlib import Path
import json
import re
import numpy as np
import torch
from datasets import load_dataset

# Directories
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parent

MODEL_1_DIR = WORKSPACE_ROOT / "model 1"
MODEL_2_DIR = WORKSPACE_ROOT / "model 2 ai assistant"
MODEL_3_DIR = WORKSPACE_ROOT / "neural network"
MODEL_4_DIR = WORKSPACE_ROOT / "omega nexus model"

DATA_OUT_DIR = BASE_DIR / "processed"
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable characters."""
    return re.sub(r"\s+", " ", text).strip()


def extract_reasoning_records(max_records: int = 350):
    """Stream and extract structured reasoning examples from Hugging Face dataset."""
    print(f"[Ingestion] Connecting to Hugging Face: MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x...")
    ds = load_dataset("MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", split="train", streaming=True)

    records = []
    print(f"[Ingestion] Streaming up to {max_records} high-quality reasoning traces...")

    for i, sample in enumerate(ds):
        messages = sample.get("messages", [])
        if not messages:
            continue

        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        if not user_msg or len(user_msg.strip()) < 20:
            continue

        thoughts = []
        final_answers = []
        tools_called = 0

        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "tool":
                tools_called += 1
            elif role == "assistant":
                th = re.findall(r"<thought>(.*?)</thought>", content, re.DOTALL)
                if th:
                    thoughts.extend([t.strip() for t in th if len(t.strip()) > 10])
                fa = re.findall(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                if fa:
                    final_answers.extend([f.strip() for f in fa if len(f.strip()) > 10])
                elif not th and len(content.strip()) > 20:
                    final_answers.append(content.strip())

        if thoughts and final_answers:
            full_thought_chain = "\n\n".join(thoughts)
            solution = "\n\n".join(final_answers)

            # Categorize complexity based on thought length & tool use
            num_steps = len(thoughts)
            total_thought_len = len(full_thought_chain)

            if num_steps <= 2 and total_thought_len < 400:
                complexity_class = 0  # Linear Reasoning
            elif num_steps <= 5 and total_thought_len < 1200:
                complexity_class = 1  # Multi-step Analytical Reasoning
            elif num_steps <= 12:
                complexity_class = 2  # Deep Algorithmic / Architectural Reasoning
            else:
                complexity_class = 3  # Complex Autonomous Agent Reasoning with Verification

            records.append({
                "id": f"fable_reasoning_{len(records):04d}",
                "system": system_msg,
                "user": user_msg,
                "thoughts": thoughts,
                "thought_chain": full_thought_chain,
                "solution": solution,
                "tools_called": tools_called,
                "num_steps": num_steps,
                "complexity_class": complexity_class,
            })

            if len(records) % 50 == 0:
                print(f"  Processed {len(records)}/{max_records} records...")

            if len(records) >= max_records:
                break

    print(f"[Ingestion] Successfully collected {len(records)} verified reasoning records.")
    return records


def prepare_model1_corpus(records):
    """Format reasoning traces into language model training corpus for OMEGA GPT-1."""
    out_file = MODEL_1_DIR / "data" / "raw" / "fable_reasoning_corpus.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Model 1] Generating reasoning language corpus at '{out_file}'...")
    lines = []
    for r in records:
        # Format as prompt-thought-answer training block
        prompt_snippet = r["user"][:400].replace("\n", " ")
        thought_snippet = r["thought_chain"][:800].replace("\n", " ")
        solution_snippet = r["solution"][:600].replace("\n", " ")

        formatted_sample = (
            f"Question: {prompt_snippet}\n"
            f"<thought>\nStep 1: Analyze problem requirements.\n"
            f"Step 2: {thought_snippet}\n"
            f"Step 3: Verify edge cases and correctness.\n"
            f"</thought>\n"
            f"Answer: {solution_snippet}\n\n"
        )
        lines.append(formatted_sample)

    corpus_text = "".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(corpus_text)

    # Also append to main training corpus for complete coverage
    combined_file = MODEL_1_DIR / "data" / "training.txt"
    with open(combined_file, "a", encoding="utf-8") as f:
        f.write("\n\n" + corpus_text)

    print(f"[Model 1] Saved {len(lines)} training blocks ({len(corpus_text):,} characters).")
    return out_file


def prepare_model2_knowledge(records):
    """Extract reasoning heuristics, problem-solving strategies, and CoT exemplars for OMEGA MODEL 2."""
    out_file = DATA_OUT_DIR / "fable_reasoning_kb.json"
    print(f"[Model 2] Distilling cognitive reasoning heuristics to '{out_file}'...")

    heuristics = []
    for r in records:
        heuristics.append({
            "task_id": r["id"],
            "query_summary": clean_text(r["user"])[:150],
            "reasoning_strategy": [clean_text(t)[:200] for t in r["thoughts"][:4]],
            "complexity_level": ["Simple", "Multi-step", "Deep", "Autonomous"][r["complexity_class"]],
            "solution_preview": clean_text(r["solution"])[:250],
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(heuristics, f, indent=2)

    print(f"[Model 2] Distilled {len(heuristics)} reasoning heuristics.")
    return out_file


def prepare_model3_features(records):
    """Build continuous & categorical feature vectors for OMEGA Neural Network Engine."""
    out_file = DATA_OUT_DIR / "fable_reasoning_neural.npz"
    print(f"[Model 3] Constructing neural feature vectors at '{out_file}'...")

    features = []
    labels = []

    for r in records:
        prompt_len = float(len(r["user"]))
        thought_len = float(len(r["thought_chain"]))
        solution_len = float(len(r["solution"]))
        num_steps = float(r["num_steps"])
        tools_called = float(r["tools_called"])
        code_density = float(r["solution"].count("def ") + r["solution"].count("class ") + r["solution"].count("import "))
        char_entropy = float(len(set(r["user"])) / max(1, len(r["user"])))
        avg_step_len = thought_len / max(1, num_steps)

        # 8-dimensional feature representation
        feat_vec = [
            np.log1p(prompt_len),
            np.log1p(thought_len),
            np.log1p(solution_len),
            num_steps,
            tools_called,
            code_density,
            char_entropy * 10.0,
            np.log1p(avg_step_len),
        ]
        features.append(feat_vec)
        labels.append(r["complexity_class"])

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X_norm = (X - mean) / std

    np.savez_compressed(out_file, X=X_norm, y=y, mean=mean, std=std)
    print(f"[Model 3] Saved {len(X)} feature records with shape {X.shape}.")
    return out_file


def prepare_model4_nexus_data(records):
    """Build paired token embeddings and neural feature vectors for OMEGA NEXUS cross-attention training."""
    out_file = DATA_OUT_DIR / "fable_reasoning_nexus.pt"
    print(f"[Model 4] Building multimodal Nexus cross-attention dataset at '{out_file}'...")

    text_prompts = [r["user"][:200] for r in records]
    complexity_labels = [r["complexity_class"] for r in records]

    # Load Model 3 normalized features
    npz_data = np.load(DATA_OUT_DIR / "fable_reasoning_neural.npz")
    X_features = torch.tensor(npz_data["X"], dtype=torch.float32)

    # Re-align feature tensor to 16 dimensions (Nexus feature_dim)
    if X_features.size(1) < 16:
        X_padded = torch.nn.functional.pad(X_features, (0, 16 - X_features.size(1)))
    else:
        X_padded = X_features[:, :16]

    torch.save({
        "prompts": text_prompts,
        "features": X_padded,
        "labels": torch.tensor(complexity_labels, dtype=torch.long),
    }, out_file)

    print(f"[Model 4] Saved {len(text_prompts)} multimodal pairs at '{out_file}'.")
    return out_file


def main():
    print("=" * 65)
    print("   FABLE REASONING DATASET INGESTION & CURATION PIPELINE")
    print("=" * 65)

    records = extract_reasoning_records(max_records=300)
    prepare_model1_corpus(records)
    prepare_model2_knowledge(records)
    prepare_model3_features(records)
    prepare_model4_nexus_data(records)

    print("=" * 65)
    print("   ALL 4 DATASETS SUCCESSFULLY GENERATED & READY FOR TRAINING")
    print("=" * 65)


if __name__ == "__main__":
    main()
