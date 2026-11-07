"""OMEGA NEURAL NETWORK ENGINE: Experiment Tracker.

Logs, compares, and visualizes neural network runs and hyperparameter configurations in SQLite.
"""

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional
from config.engine_config import EXPERIMENT_DB_PATH


class ExperimentTracker:
    """Manages neural network experiment persistence and metric comparisons."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(EXPERIMENT_DB_PATH)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_config TEXT NOT NULL,
                dataset_info TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,
                best_val_metric REAL,
                final_train_loss REAL,
                epochs_trained INTEGER,
                duration_seconds REAL,
                checkpoint_path TEXT,
                created_at REAL NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    def log_run(
        self,
        name: str,
        model_config: Dict[str, Any],
        dataset_info: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        best_val_metric: float,
        final_train_loss: float,
        epochs_trained: int,
        duration_seconds: float,
        checkpoint_path: str = "",
    ) -> str:
        """Record an experiment run."""
        exp_id = str(uuid.uuid4())
        now = time.time()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO experiments (
                id, name, model_config, dataset_info, hyperparameters,
                best_val_metric, final_train_loss, epochs_trained,
                duration_seconds, checkpoint_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp_id,
                name,
                json.dumps(model_config),
                json.dumps(dataset_info),
                json.dumps(hyperparameters),
                best_val_metric,
                final_train_loss,
                epochs_trained,
                duration_seconds,
                checkpoint_path,
                now,
            ),
        )
        conn.commit()
        conn.close()
        return exp_id

    def list_experiments(self, limit: int = 25) -> List[Dict[str, Any]]:
        """List past experiment runs ordered by creation timestamp."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        for r in rows:
            r["model_config"] = json.loads(r["model_config"])
            r["dataset_info"] = json.loads(r["dataset_info"])
            r["hyperparameters"] = json.loads(r["hyperparameters"])
        return rows

    def compare_experiments(self, exp_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple runs side-by-side for metric delta analysis."""
        conn = self._get_connection()
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in exp_ids)
        cursor.execute(f"SELECT * FROM experiments WHERE id IN ({placeholders})", exp_ids)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows


# Singleton
experiment_tracker = ExperimentTracker()
