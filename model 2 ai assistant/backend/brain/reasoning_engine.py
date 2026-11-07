"""OMEGA MODEL 2: Multi-Step Reasoning and Plan Decomposition Engine.

Analyzes user queries, determines intent, plans tool orchestration,
and synthesizes chain-of-thought rationale before generating responses.
"""

import re
from typing import Any, Dict, List, Optional


class ReasoningEngine:
    """Classifies user intents and formulates structured multi-step execution plans."""

    def analyze_intent(self, user_query: str) -> Dict[str, Any]:
        """Classify user intent into functional categories."""
        q = user_query.strip().lower()

        # 1. Calculation Intent
        if any(w in q for w in ["calculate", "math", "evaluate", "plus", "minus", "divided by"]) or re.search(r"[\d\(\)]+\s*[\+\-\*\/\^]\s*[\d\(\)]+", q):
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

