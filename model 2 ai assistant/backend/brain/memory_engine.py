"""OMEGA MODEL 2: Advanced 3-Tier SQLite Memory Engine.

Supports:
1. Short-term memory: Current conversation context, active task states.
2. Long-term memory: User preferences, approved facts, importance scoring.
3. Project memory: Architecture details, codebase notes, development history.
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from backend.database.db import get_db_connection


class MemoryEngine:
    """Manages persistent SQLite memory with semantic scoring and deduplication."""

    def add_memory(
        self,
        memory_type: str,
        category: str,
        content: str,
        importance_score: float = 0.5,
        expires_in_seconds: Optional[float] = None,
    ) -> str:
        """Store a new memory item."""
        mem_id = str(uuid.uuid4())
        now = time.time()
        expires_at = now + expires_in_seconds if expires_in_seconds else None

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check for deduplication
        cursor.execute(
            "SELECT id FROM memories WHERE memory_type = ? AND category = ? AND content = ?",
            (memory_type, category, content.strip()),
        )
        existing = cursor.fetchone()
        if existing:
            # Bump access count and update timestamp
            cursor.execute(
                "UPDATE memories SET access_count = access_count + 1, updated_at = ? WHERE id = ?",
                (now, existing["id"]),
            )
            conn.commit()
            conn.close()
            return existing["id"]

        cursor.execute(
            """
            INSERT INTO memories (id, memory_type, category, content, importance_score, access_count, created_at, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (mem_id, memory_type, category, content.strip(), importance_score, now, now, expires_at),
        )
        conn.commit()
        conn.close()
        return mem_id

    def search_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        min_score: float = 0.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search memories using keyword matching and importance ranking."""
        conn = get_db_connection()
        cursor = conn.cursor()
        now = time.time()

        base_sql = """
            SELECT * FROM memories 
            WHERE (expires_at IS NULL OR expires_at > ?)
            AND importance_score >= ?
        """
        params: List[Any] = [now, min_score]

        if memory_type:
            base_sql += " AND memory_type = ?"
            params.append(memory_type)

        if query.strip():
            base_sql += " AND (content LIKE ? OR category LIKE ?)"
            params.append(f"%{query}%")
            params.append(f"%{query}%")

        base_sql += " ORDER BY importance_score DESC, access_count DESC, updated_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(base_sql, params)
        rows = cursor.fetchall()
        results = [dict(r) for r in rows]
        conn.close()
        return results

    def get_relevant_context(self, user_query: str, max_items: int = 5) -> str:
        """Construct synthesized memory context string for LLM prompting."""
        words = [w for w in user_query.lower().split() if len(w) > 3]
        matched_memories = []

        for word in words[:4]:
            found = self.search_memories(query=word, min_score=0.4, limit=3)
            for m in found:
                if m not in matched_memories:
                    matched_memories.append(m)

        if not matched_memories:
            # Fall back to high-importance long-term memories
            matched_memories = self.search_memories(query="", min_score=0.8, limit=max_items)

        if not matched_memories:
            return ""

        context_lines = ["--- RELEVANT MEMORY CONTEXT ---"]
        for m in matched_memories[:max_items]:
            m_type = m["memory_type"].upper().replace("_", " ")
            context_lines.append(f"[{m_type} - {m['category']}]: {m['content']}")
        context_lines.append("-------------------------------")
        return "\n".join(context_lines)

    def list_all(self, memory_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve stored memories for UI management."""
        conn = get_db_connection()
        cursor = conn.cursor()
        now = time.time()

        if memory_type:
            cursor.execute(
                "SELECT * FROM memories WHERE memory_type = ? AND (expires_at IS NULL OR expires_at > ?) ORDER BY updated_at DESC LIMIT ?",
                (memory_type, now, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM memories WHERE (expires_at IS NULL OR expires_at > ?) ORDER BY updated_at DESC LIMIT ?",
                (now, limit),
            )
        rows = cursor.fetchall()
        data = [dict(r) for r in rows]
        conn.close()
        return data

    def delete_memory(self, memory_id: str) -> bool:
        """Permanently delete a memory item."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def clear_all(self, memory_type: Optional[str] = None):
        """Clear memory items by type or entirely."""
        conn = get_db_connection()
        cursor = conn.cursor()
        if memory_type:
            cursor.execute("DELETE FROM memories WHERE memory_type = ?", (memory_type,))
        else:
            cursor.execute("DELETE FROM memories")
        conn.commit()
        conn.close()


# Singleton instance
memory_engine = MemoryEngine()

