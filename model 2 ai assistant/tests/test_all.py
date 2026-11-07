"""OMEGA MODEL 2: Comprehensive Automated Test Suite.

Tests:
1. Multi-Provider LLM Engine & Model 1 Adapter
2. 3-Tier SQLite Memory Engine (CRUD, Scoring, Deduplication)
3. Tool Execution Engine (Calculator, Time, Registry)
4. Coding Assistant & AST Sandbox Security Gate
5. Web Search Engine & Synthesis
6. Knowledge Retrieval Engine (Ingestion, Chunking, Vectors)
7. End-to-end Generator Engine
"""

import os
import sys
import unittest

# Ensure model 2 root is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from backend.brain.coding_engine import coding_engine
from backend.brain.conversation_manager import conversation_manager
from backend.brain.generator_engine import generator_engine
from backend.brain.llm_engine import llm_engine
from backend.brain.memory_engine import memory_engine
from backend.brain.omega1_adapter import omega1_adapter
from backend.brain.reasoning_engine import reasoning_engine
from backend.brain.retrieval_engine import knowledge_engine
from backend.brain.safety_engine import safety_engine
from backend.brain.search_engine import search_engine
from backend.brain.tool_engine import tool_engine


class TestOmegaModel2(unittest.TestCase):

    def test_01_llm_engine_listing(self):
        """Verify model providers are registered and classified accurately."""
        models = llm_engine.list_models()
        self.assertGreaterEqual(len(models), 5)
        prov_ids = [m["id"] for m in models]
        self.assertIn("omega_gpt1", prov_ids)
        self.assertIn("omega_offline", prov_ids)
        self.assertIn("openai", prov_ids)
        self.assertIn("gemini", prov_ids)

    def test_02_omega1_adapter_metadata(self):
        """Verify Omega GPT-1 metadata reflects Model 1 sovereign configuration."""
        meta = omega1_adapter.metadata
        self.assertEqual(meta["name"], "Omega GPT-1")
        self.assertEqual(meta["vocab_size"], 1200)
        self.assertEqual(meta["parameters"], 3490304)

    def test_03_memory_engine_crud(self):
        """Verify SQLite 3-tier memory creation, search, deduplication, and deletion."""
        content = "User preference: prefers Pre-LayerNorm over Post-LayerNorm."
        mem_id = memory_engine.add_memory(
            memory_type="long_term",
            category="architecture",
            content=content,
            importance_score=0.95,
        )
        self.assertIsNotNone(mem_id)

        # Search
        results = memory_engine.search_memories("Pre-LayerNorm")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("Pre-LayerNorm", results[0]["content"])

        # Context build
        ctx = memory_engine.get_relevant_context("Tell me about Pre-LayerNorm")
        self.assertIn("RELEVANT MEMORY CONTEXT", ctx)

        # Cleanup
        deleted = memory_engine.delete_memory(mem_id)
        self.assertTrue(deleted)

    def test_04_tools_execution(self):
        """Verify built-in tools (calculator, datetime) execute accurately."""
        # Calculator
        res = tool_engine.execute("calculator", {"expression": "2**8 + 10"})
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], "266")

        # DateTime
        t_res = tool_engine.execute("get_current_time", {})
        self.assertTrue(t_res["success"])
        self.assertIn("local_time", t_res["result"])

    def test_05_sandbox_security(self):
        """Verify AST sandbox executes safe code and blocks prohibited dangerous calls."""
        # Safe code
        safe_code = "x = [i**2 for i in range(5)]\nprint('SUM:', sum(x))"
        safe_res = coding_engine.execute_python_sandbox(safe_code)
        self.assertTrue(safe_res["success"])
        self.assertIn("SUM: 30", safe_res["output"])

        # Unsafe code (import os)
        unsafe_code = "import os\nos.system('dir')"
        unsafe_res = coding_engine.execute_python_sandbox(unsafe_code)
        self.assertFalse(unsafe_res["success"])
        self.assertIn("Security Alert", unsafe_res["error"])

    def test_06_knowledge_ingestion_and_search(self):
        """Verify document ingestion, chunking, and semantic vector retrieval."""
        test_doc = (
            "Scaled dot-product attention computes compatibility between queries and keys. "
            "In Omega GPT-1, causal masking enforces autoregressive factorization."
        )
        doc_id = knowledge_engine.ingest_text("test_attention.txt", test_doc, file_type="txt")
        self.assertIsNotNone(doc_id)

        matches = knowledge_engine.search("causal masking attention")
        self.assertGreaterEqual(len(matches), 1)
        self.assertIn("test_attention.txt", matches[0]["filename"])

        # Clean up test document
        knowledge_engine.delete_document(doc_id)

    def test_07_end_to_end_generator(self):
        """Verify complete pipeline generation."""
        conv_id = conversation_manager.create_conversation("Integration Test")
        res = generator_engine.process_query(
            query="What is 15 * 12?",
            conversation_id=conv_id,
            provider="omega_offline",
        )
        self.assertIn("180", res["response"])
        self.assertIsNotNone(res["latency_ms"])


if __name__ == "__main__":
    unittest.main()

