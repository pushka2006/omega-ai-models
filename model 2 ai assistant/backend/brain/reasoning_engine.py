"""OMEGA MODEL 2: Multi-Step Reasoning, Plan Decomposition, and CoT Engine.

Enhanced with distilled reasoning strategies from MoreThought/Fable-5.1-Max-Reasoning.
Analyzes user queries, determines intent, plans tool orchestration,
and synthesizes chain-of-thought rationale (<thought>) before generating responses.
"""

from pathlib import Path
import json
import re
import sqlite3
from typing import Any, Dict, List, Optional


class ReasoningEngine:
    """Classifies user intents and formulates structured multi-step execution plans."""

    def __init__(self):
        self.db_path = Path(__file__).resolve().parent.parent.parent / "data" / "omega2.db"

    def analyze_intent(self, user_query: str) -> Dict[str, Any]:
        """Classify user intent into functional categories."""
        q = user_query.strip().lower()

        # 1. Calculation Intent
        if any(w in q for w in ["calculate", "math", "evaluate", "plus", "minus", "divided by"]) or re.search(r"\d+\s*[\+\-\*\/\^]\s*[\d\(\)]+", q):
            return {
                "intent": "calculation",
                "recommended_tool": "calculator",
                "confidence": 0.95,
                "needs_reasoning": True,
            }

        # 2. Time & Date Intent
        if any(w in q for w in ["what time is it", "current time", "current date", "today's date", "what day is"]):
            return {
                "intent": "datetime",
                "recommended_tool": "get_current_time",
                "confidence": 0.98,
                "needs_reasoning": False,
            }

        # 3. Python Code Execution Intent
        if any(w in q for w in ["run this code", "execute code", "run python", "execute python", "eval python"]) or "```python" in user_query:
            return {
                "intent": "code_execution",
                "recommended_tool": "run_python_code",
                "confidence": 0.92,
                "needs_reasoning": True,
            }

        # 4. Web Search Intent
        if any(w in q for w in ["search for", "google", "search the web", "look up online", "who won", "latest news on", "recent price"]):
            return {
                "intent": "web_search",
                "recommended_tool": "web_search",
                "confidence": 0.88,
                "needs_reasoning": True,
            }

        # 5. File Inspection Intent
        if any(w in q for w in ["read file", "show file", "inspect file", "search files", "list files in"]):
            return {
                "intent": "file_operation",
                "recommended_tool": "search_workspace_files",
                "confidence": 0.85,
                "needs_reasoning": True,
            }

        # 6. Conversational / General Knowledge Intent
        return {
            "intent": "conversation",
            "recommended_tool": None,
            "confidence": 0.90,
            "needs_reasoning": False,
        }

    def find_reasoning_heuristics(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Retrieve matching reasoning patterns from the Fable knowledge store in SQLite."""
        if not self.db_path.exists():
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]

            results = []
            if words:
                # Keyword matching on ingested heuristics
                conditions = " OR ".join(["query_summary LIKE ?" for _ in words[:3]])
                params = [f"%{w}%" for w in words[:3]]
                cursor.execute(f"""
                    SELECT id, query_summary, reasoning_steps, complexity_level, solution_preview 
                    FROM fable_reasoning_heuristics 
                    WHERE {conditions}
                    LIMIT ?
                """, (*params, limit))
                rows = cursor.fetchall()
                for row in rows:
                    results.append({
                        "id": row[0],
                        "summary": row[1],
                        "steps": json.loads(row[2]) if row[2] else [],
                        "complexity": row[3],
                        "preview": row[4],
                    })

            # Fallback if no direct keyword match
            if not results:
                cursor.execute("""
                    SELECT id, query_summary, reasoning_steps, complexity_level, solution_preview 
                    FROM fable_reasoning_heuristics 
                    ORDER BY RANDOM() LIMIT ?
                """, (limit,))
                for row in cursor.fetchall():
                    results.append({
                        "id": row[0],
                        "summary": row[1],
                        "steps": json.loads(row[2]) if row[2] else [],
                        "complexity": row[3],
                        "preview": row[4],
                    })

            conn.close()
            return results
        except Exception:
            return []

    def generate_thought_chain(self, query: str) -> str:
        """Construct structured Fable-style Chain-of-Thought (<thought>) reasoning trace."""
        intent_info = self.analyze_intent(query)
        intent = intent_info["intent"]
        heuristics = self.find_reasoning_heuristics(query, limit=1)

        steps = [
            f"1. Problem Decomposition: Identify goal for intent '{intent}'. Deconstruct explicit inputs and implicit constraints.",
        ]

        if heuristics and heuristics[0].get("steps"):
            matched_step = heuristics[0]["steps"][0]
            steps.append(f"2. Strategy Formulation: Borrow heuristic ({heuristics[0]['complexity']}): '{matched_step}'")
        else:
            steps.append("2. Strategy Formulation: Establish hypothesis, define execution roadmap, and determine necessary tool invocations.")

        steps.append("3. Execution & Edge Cases: Validate boundaries, verify safety constraints, and check potential failure modes.")
        steps.append("4. Synthesis & Verification: Confirm answer satisfies the prompt with high confidence.")

        return "<thought>\n" + "\n".join(steps) + "\n</thought>"

    def generate_plan(self, query: str) -> List[str]:
        """Generate chain-of-thought breakdown steps."""
        intent_info = self.analyze_intent(query)
        intent = intent_info["intent"]

        if intent == "calculation":
            return [
                "Parse mathematical formula from user query.",
                "Verify safety of expression against math namespace.",
                "Execute calculation safely.",
                "Format result with step-by-step mathematical explanation.",
            ]
        elif intent == "web_search":
            return [
                "Formulate optimized search terms.",
                "Execute DuckDuckGo search query.",
                "Extract and clean webpage snippets.",
                "Synthesize findings with verifiable URL citations.",
            ]
        elif intent == "code_execution":
            return [
                "Extract code block from query.",
                "Validate safety against AST security whitelist.",
                "Execute code inside sandboxed environment.",
                "Capture standard output and explain runtime results.",
            ]
        elif intent == "file_operation":
            return [
                "Verify file path within safe workspace boundaries.",
                "Read contents and analyze structure.",
                "Deliver grounded answer to user.",
            ]
        else:
            return [
                "Retrieve relevant context from 3-tier SQLite memory.",
                "Check knowledge base for document citations.",
                "Generate coherent, structured response via selected model provider.",
            ]


# Singleton instance
reasoning_engine = ReasoningEngine()
