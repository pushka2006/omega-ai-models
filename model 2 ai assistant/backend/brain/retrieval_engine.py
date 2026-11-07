"""OMEGA MODEL 2: Knowledge Ingestion and Semantic Retrieval Engine.

Supports ingestion of TXT, Markdown, JSON, CSV, Code, and PDF files.
Performs chunking, TF-IDF vectorization, semantic similarity scoring,
and grounded document citations.
"""

import json
import math
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.database.db import get_db_connection


class KnowledgeRetrievalEngine:
    """Manages document ingestion, chunking, and semantic vector retrieval."""

    def __init__(self):
        self.chunk_size = 250  # Words per chunk
        self.chunk_overlap = 40

    def ingest_text(self, filename: str, content: str, file_type: str = "txt") -> str:
        """Ingest plain text or code document, chunk, and index."""
        doc_id = str(uuid.uuid4())
        now = os.path.getmtime(filename) if os.path.exists(filename) else 0

        # 1. Clean and normalize content
        cleaned_content = content.replace("\r\n", "\n")
        summary = cleaned_content[:200] + ("..." if len(cleaned_content) > 200 else "")

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO documents (id, filename, file_type, file_size, content, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, os.path.basename(filename), file_type, len(cleaned_content), cleaned_content, summary, now),
        )

        # 2. Chunking
        words = cleaned_content.split()
        chunks = []
        i = 0
        chunk_idx = 0

        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunk_str = " ".join(chunk_words)
            chunk_id = str(uuid.uuid4())

            # Generate lightweight term-frequency vector representation
            tf_dict = self._compute_tf(chunk_str)

            cursor.execute(
                """
                INSERT INTO document_chunks (id, document_id, chunk_index, chunk_text, token_count, vector_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, doc_id, chunk_idx, chunk_str, len(chunk_words), json.dumps(tf_dict)),
            )

            chunk_idx += 1
            i += self.chunk_size - self.chunk_overlap

        conn.commit()
        conn.close()
        return doc_id

    def ingest_file(self, file_path: str) -> str:
        """Ingest local file based on extension."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower().replace(".", "")
        if ext in ["txt", "md", "json", "csv", "py", "js", "ts", "html", "css", "yaml", "yml"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return self.ingest_text(file_path, content, file_type=ext)

        elif ext == "pdf":
            content = self._extract_pdf(file_path)
            return self.ingest_text(file_path, content, file_type="pdf")

        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return self.ingest_text(file_path, content, file_type="raw")

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF using pypdf or PyPDF2 if available, fallback to regex."""
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(file_path)
                return "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                with open(file_path, "rb") as f:
                    raw = f.read().decode("latin1", errors="ignore")
                    matches = re.findall(r"\(([^\(\)]+)\)", raw)
                    return " ".join(matches)

    def _compute_tf(self, text: str) -> Dict[str, float]:
        """Compute term frequency vector dictionary for a text block."""
        words = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", text.lower())
        total = len(words)
        if total == 0:
            return {}
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        return {w: round(c / total, 5) for w, c in counts.items()}

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve most semantically relevant document chunks for a query."""
        query_words = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", query.lower())
        if not query_words:
            return []

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT c.id, c.document_id, c.chunk_index, c.chunk_text, c.vector_json, d.filename, d.file_type
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        scores = []
        for row in rows:
            chunk_dict = json.loads(row["vector_json"]) if row["vector_json"] else {}
            score = 0.0

            # Compute term overlap score
            for q_word in query_words:
                if q_word in chunk_dict:
                    score += chunk_dict[q_word] * 10.0
                elif any(q_word in key for key in chunk_dict.keys()):
                    score += 0.5

            if score > 0.0:
                scores.append({
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "filename": row["filename"],
                    "file_type": row["file_type"],
                    "chunk_index": row["chunk_index"],
                    "text": row["chunk_text"],
                    "score": round(score, 4),
                })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def build_grounded_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Construct grounded context block with document citations."""
        matches = self.search(query, top_k=top_k)
        if not matches:
            return {"found": False, "context": "", "citations": []}

        citations = []
        context_parts = ["--- RETRIEVED KNOWLEDGE BASE CONTEXT ---"]
        for i, m in enumerate(matches, 1):
            citations.append({
                "citation_id": i,
                "document": m["filename"],
                "chunk_index": m["chunk_index"],
                "score": m["score"],
            })
            context_parts.append(
                f"[{i}] Document: '{m['filename']}' (Chunk {m['chunk_index']})\n"
                f"{m['text']}\n"
            )
        context_parts.append("------------------------------------------")

        return {
            "found": True,
            "context": "\n".join(context_parts),
            "citations": citations,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.id, d.filename, d.file_type, d.file_size, d.summary, d.created_at, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN document_chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """)
        rows = cursor.fetchall()
        docs = [dict(r) for r in rows]
        conn.close()
        return docs

    def delete_document(self, doc_id: str) -> bool:
        """Delete an ingested document and its chunks."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted


# Singleton instance
knowledge_engine = KnowledgeRetrievalEngine()

