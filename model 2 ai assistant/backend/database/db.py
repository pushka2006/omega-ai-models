"""OMEGA MODEL 2: SQLite Database Schema and Connection Pool.

Manages relational tables for conversations, 3-tier memory (short-term, long-term,
project), knowledge documents, document chunks, personal tasks, notes, and tool execution logs.
"""

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional
from backend.config import DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_database():
    """Initialize all relational tables and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Conversations & Messages
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            metadata TEXT DEFAULT '{}'
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model_name TEXT,
            provider TEXT,
            tool_calls TEXT DEFAULT '[]',
            created_at REAL NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
        );
    """)

    # 2. Advanced 3-Tier Memory
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,   -- 'short_term', 'long_term', 'project'
            category TEXT NOT NULL,      -- 'preference', 'fact', 'architecture', 'context'
            content TEXT NOT NULL,
            importance_score REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            expires_at REAL DEFAULT NULL
        );
    """)

    # 3. Knowledge Documents & Semantic Chunks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            content TEXT NOT NULL,
            summary TEXT,
            created_at REAL NOT NULL
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            vector_json TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        );
    """)

    # 4. Personal Productivity: Notes, Tasks, Reminders
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '[]',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',   -- 'pending', 'in_progress', 'completed'
            priority TEXT DEFAULT 'medium',  -- 'low', 'medium', 'high', 'urgent'
            due_date TEXT,
            created_at REAL NOT NULL
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            remind_at REAL NOT NULL,
            is_triggered INTEGER DEFAULT 0,
            is_canceled INTEGER DEFAULT 0,
            created_at REAL NOT NULL
        );
    """)

    # 5. Tool Execution Audit Log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool_logs (
            id TEXT PRIMARY KEY,
            tool_name TEXT NOT NULL,
            inputs TEXT NOT NULL,
            outputs TEXT,
            status TEXT NOT NULL,            -- 'success', 'failed', 'blocked'
            execution_time_ms REAL NOT NULL,
            created_at REAL NOT NULL
        );
    """)

    # Performance Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id, created_at);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type, importance_score);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks (document_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminders_time ON reminders (remind_at, is_triggered);")

    conn.commit()
    conn.close()


# Initialize schema on load
init_database()

