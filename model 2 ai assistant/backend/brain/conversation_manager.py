"""OMEGA MODEL 2: Conversation Context and History Manager.

Manages sessions, message persistence, context window budgeting,
and conversation lifecycle operations.
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from backend.database.db import get_db_connection


class ConversationManager:
    """Manages chat sessions and historical context tracking."""

    def create_conversation(self, title: str = "New OMEGA Chat") -> str:
        """Create a new persistent conversation session."""
        conv_id = str(uuid.uuid4())
        now = time.time()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, '{}')",
            (conv_id, title, now, now),
        )
        conn.commit()
        conn.close()
        return conv_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Append a message to conversation history."""
        msg_id = str(uuid.uuid4())
        now = time.time()
        conn = get_db_connection()
        cursor = conn.cursor()

        # Ensure conversation exists
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, content[:30] + "...", now, now),
            )

        import json
        tools_json = json.dumps(tool_calls or [])

        cursor.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, model_name, provider, tool_calls, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, conversation_id, role, content, model_name, provider, tools_json, now),
        )

        cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        conn.commit()
        conn.close()
        return msg_id

    def get_messages(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve historical messages for a conversation session."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
            (conversation_id, limit),
        )
        rows = cursor.fetchall()
        messages = []
        import json
        for r in rows:
            d = dict(r)
            d["tool_calls"] = json.loads(d.get("tool_calls") or "[]")
            messages.append(d)
        conn.close()
        return messages

    def list_conversations(self, limit: int = 30) -> List[Dict[str, Any]]:
        """List recent conversation sessions."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.*, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        data = [dict(r) for r in rows]
        conn.close()
        return data

    def delete_conversation(self, conversation_id: str) -> bool:
        """Permanently delete a conversation and associated messages."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def clear_all(self):
        """Purge all conversations."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()


# Singleton instance
conversation_manager = ConversationManager()

