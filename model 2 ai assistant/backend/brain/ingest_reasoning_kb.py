"""Ingest distilled Fable 5.1 Max-Reasoning knowledge into Model 2 SQLite database."""

from pathlib import Path
import json
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR.parent / "data" / "omega2.db"
JSON_PATH = BASE_DIR.parent.parent / "datasets_hub" / "processed" / "fable_reasoning_kb.json"


def ingest_fable_heuristics():
    if not JSON_PATH.exists():
        print(f"Error: {JSON_PATH} does not exist.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        heuristics = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Create table for reasoning heuristics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fable_reasoning_heuristics (
            id TEXT PRIMARY KEY,
            query_summary TEXT,
            reasoning_steps TEXT,
            complexity_level TEXT,
            solution_preview TEXT
        )
    """)

    # Populate heuristics
    count = 0
    for h in heuristics:
        cursor.execute("""
            INSERT OR REPLACE INTO fable_reasoning_heuristics 
            (id, query_summary, reasoning_steps, complexity_level, solution_preview)
            VALUES (?, ?, ?, ?, ?)
        """, (
            h["task_id"],
            h["query_summary"],
            json.dumps(h["reasoning_strategy"]),
            h["complexity_level"],
            h["solution_preview"],
        ))
        count += 1

    # Also insert top reasoning patterns into knowledge_documents
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for i, h in enumerate(heuristics[:50]):
        doc_id = f"fable_reasoning_pattern_{i+1:03d}"
        title = f"Fable Reasoning Pattern: {h['query_summary'][:40]}"
        steps_str = "\n".join(f"- {s}" for s in h["reasoning_strategy"])
        content = f"Problem: {h['query_summary']}\nStrategy:\n{steps_str}\nSolution: {h['solution_preview']}"
        cursor.execute("""
            INSERT OR REPLACE INTO knowledge_documents (id, title, content, category)
            VALUES (?, ?, ?, 'reasoning_heuristics')
        """, (doc_id, title, content))

    conn.commit()
    conn.close()
    print(f"[Model 2 DB] Ingested {count} reasoning heuristics into '{DB_PATH.name}'.")


if __name__ == "__main__":
    ingest_fable_heuristics()
