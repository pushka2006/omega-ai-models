"""OMEGA NEXUS MODEL: Adapter for OMEGA MODEL 2 (AI Assistant).

Enables OMEGA NEXUS to route tasks through cognitive engines:
Memory, Web Search, Code Execution, and Reasoning.
"""

from pathlib import Path
import sys
from typing import Dict, Any, Optional

from config import MODEL_2_DIR


class AssistantAdapter:
    """Interoperability wrapper around OMEGA Model 2 cognitive engines."""

    def __init__(self):
        self.available = False
        self.brain = None
        self._load()

    def _load(self):
        if not MODEL_2_DIR.exists():
            return

        model2_str = str(MODEL_2_DIR)
        if model2_str not in sys.path:
            sys.path.insert(0, model2_str)

        try:
            from backend.brain.generator_engine import generator_engine
            from backend.brain.search_engine import search_engine
            from backend.brain.coding_engine import coding_engine

            self.generator = generator_engine
            self.search_engine = search_engine
            self.coding_engine = coding_engine
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)

    def process_query(self, query: str, user_id: str = "nexus_operator") -> Dict[str, Any]:
        """Dispatch query through Model 2 assistant pipeline."""
        if not self.available:
            return {
                "source": "nexus_cognitive_mock",
                "response": f"[Nexus Assistant Fallback] Processed query: '{query}'",
                "metadata": {"status": "fallback"}
            }

        try:
            res = self.generator.process_query(query=query, conversation_id=user_id)
            return {
                "source": "omega_model_2",
                "response": res.get("response", ""),
                "thought_chain": res.get("thought_chain", ""),
                "intent": res.get("intent", "general"),
                "metadata": {
                    "provider": res.get("provider"),
                    "latency_ms": res.get("latency_ms"),
                    "thought_chain": res.get("thought_chain", ""),
                }
            }
        except Exception as e:
            return {
                "source": "omega_model_2_error",
                "response": f"Encountered error in Model 2: {e}",
                "metadata": {"error": str(e)}
            }

    def search_web(self, query: str) -> str:
        """Run real web search through Model 2 search engine."""
        if not self.available:
            return f"Mock search result for '{query}'"
        try:
            return self.search_engine.search(query)
        except Exception as e:
            return f"Search error: {e}"

    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute sandboxed Python code through Model 2 coding engine."""
        if not self.available:
            return {"success": False, "output": "Sandbox not available."}
        try:
            return self.coding_engine.execute(code)
        except Exception as e:
            return {"success": False, "output": str(e)}
