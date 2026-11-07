"""OMEGA MODEL 2: Personal Productivity Endpoints (Notes, Tasks, Reminders)."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.database.db import get_db_connection

router = APIRouter(prefix="/api/productivity", tags=["Productivity"])


# Models
class NoteCreate(BaseModel):
    title: str
    content: str
    category: str = "general"
    tags: List[str] = []


class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    status: str = "pending"
    priority: str = "medium"
    due_date: Optional[str] = None


class ReminderCreate(BaseModel):
    title: str
    remind_in_seconds: float


# 1. NOTES
@router.get("/notes")
def list_notes():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notes ORDER BY updated_at DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    for r in rows:
        r["tags"] = json.loads(r["tags"] or "[]")
    conn.close()
    return rows


@router.post("/notes")
def create_note(req: NoteCreate):
    note_id = str(uuid.uuid4())
    now = time.time()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO notes (id, title, content, category, tags, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (note_id, req.title, req.content, req.category, json.dumps(req.tags), now, now),
    )
    conn.commit()
    conn.close()
    return {"id": note_id, "title": req.title}


@router.delete("/notes/{note_id}")
def delete_note(note_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if not deleted:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"success": True}


# 2. TASKS
@router.get("/tasks")
def list_tasks():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


@router.post("/tasks")
def create_task(req: TaskCreate):
    task_id = str(uuid.uuid4())
    now = time.time()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO tasks (id, title, description, status, priority, due_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (task_id, req.title, req.description, req.status, req.priority, req.due_date, now),
    )
    conn.commit()
    conn.close()
    return {"id": task_id, "title": req.title}


@router.patch("/tasks/{task_id}/status")
def update_task_status(task_id: str, status: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if not updated:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True, "status": status}


@router.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True}


# 3. REMINDERS
@router.get("/reminders")
def list_reminders():
    conn = get_db_connection()
    cursor = conn.cursor()
    now = time.time()
    # Auto-mark triggered reminders
    cursor.execute("UPDATE reminders SET is_triggered = 1 WHERE remind_at <= ? AND is_triggered = 0", (now,))
    conn.commit()

    cursor.execute("SELECT * FROM reminders WHERE is_canceled = 0 ORDER BY remind_at ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


@router.post("/reminders")
def create_reminder(req: ReminderCreate):
    rem_id = str(uuid.uuid4())
    now = time.time()
    remind_time = now + req.remind_in_seconds
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO reminders (id, title, remind_at, is_triggered, is_canceled, created_at)
        VALUES (?, ?, ?, 0, 0, ?)
        """,
        (rem_id, req.title, remind_time, now),
    )
    conn.commit()
    conn.close()
    return {"id": rem_id, "title": req.title, "remind_at": remind_time}

