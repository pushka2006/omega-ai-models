"""OMEGA MODEL 2: Unified Response Generation and Synthesis Engine.

Orchestrates reasoning, memory retrieval, knowledge base citations,
tool execution, and multi-provider generation into a unified assistant response.
"""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.brain.conversation_manager import conversation_manager
from backend.brain.coding_engine import coding_engine
from backend.brain.llm_engine import llm_engine
from backend.brain.memory_engine import memory_engine
from backend.brain.reasoning_engine import reasoning_engine
from backend.brain.retrieval_engine import knowledge_engine
from backend.brain.safety_engine import safety_engine
from backend.brain.search_engine import search_engine
from backend.brain.tool_engine import tool_engine


class GeneratorEngine:
    """End-to-end response generation pipeline."""

    def process_query(
        self,
        query: str,
        conversation_id: str,
        provider: str = "omega_offline",
        model: Optional[str] = None,
        use_web_search: bool = False,
        use_knowledge_base: bool = True,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Process user input, orchestrate tools/memory, and generate complete response."""
        t0 = time.time()
        clean_query = safety_engine.sanitize_user_input(query)

        # 1. Record User Message in SQLite
        conversation_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=clean_query,
        )

        # 2. Analyze Intent & Planning
        intent_info = reasoning_engine.analyze_intent(clean_query)
        thought_chain = reasoning_engine.generate_thought_chain(clean_query)
        tool_executed = None
        tool_result = None

        # 3. Automatic Tool Triggering if Intent is Specialized
        if intent_info["recommended_tool"] == "calculator":
            import re
            expr_match = re.search(r"(\d+\s*[\+\-\*\/\^]\s*[\d\s\+\-\*\/\^\.\(\)]+)", clean_query)
            if expr_match:
                expr = expr_match.group(1).strip()
                t_res = tool_engine.execute("calculator", {"expression": expr})
                if t_res["success"]:
                    tool_executed = "calculator"
                    tool_result = f"Calculation result for `{expr}`: **{t_res['result']}**"

        elif intent_info["recommended_tool"] == "get_current_time":
            t_res = tool_engine.execute("get_current_time", {})
            if t_res["success"]:
                tool_executed = "get_current_time"
                tool_result = f"Current Time: **{t_res['result']['local_time']}** ({t_res['result']['day_of_week']})"

        # 4. Web Search Execution if toggled or query requests it
        search_citations = []
        search_context = ""
        if use_web_search or intent_info["intent"] == "web_search":
            s_res = search_engine.search_and_synthesize(clean_query, max_results=3)
            if s_res["found"]:
                search_citations = s_res["citations"]
                search_context = s_res["context"]
                tool_executed = "web_search"

        # 5. Knowledge Base Semantic Retrieval
        doc_citations = []
        kb_context = ""
        if use_knowledge_base:
            kb_res = knowledge_engine.build_grounded_context(clean_query, top_k=2)
            if kb_res["found"]:
                doc_citations = kb_res["citations"]
                kb_context = kb_res["context"]

        # 6. Retrieve Relevant 3-Tier Memory
        memory_context = memory_engine.get_relevant_context(clean_query, max_items=4)

        # 7. Assemble Structured Prompt
        system_prompt = (
            "You are OMEGA MODEL 2, an advanced futuristic AI assistant in the OMEGA ecosystem.\n"
            "You provide accurate, well-structured, mathematically rigorous, and helpful answers.\n"
            "Cite facts accurately and explain concepts clearly."
        )

        prompt_sections = []
        if memory_context:
            prompt_sections.append(memory_context)
        if kb_context:
            prompt_sections.append(kb_context)
        if search_context:
            prompt_sections.append(f"--- LIVE WEB SEARCH CONTEXT ---\n{search_context}\n-------------------------------")
        if tool_result:
            prompt_sections.append(f"--- TOOL EXECUTION OUTPUT ---\n{tool_result}\n----------------------------")

        # Fetch recent conversational messages
        history = conversation_manager.get_messages(conversation_id, limit=6)
        formatted_history = []
        for msg in history[:-1]:  # Exclude current user prompt already saved
            formatted_history.append({"role": msg["role"], "content": msg["content"]})

        full_prompt = clean_query
        if prompt_sections:
            full_prompt = "\n\n".join(prompt_sections) + f"\n\nUser Question: {clean_query}"

        # 8. Multi-Provider Generation
        gen_output = llm_engine.generate(
            prompt=full_prompt,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            context_messages=formatted_history,
        )

        response_text = gen_output.get("text", "")
        # If a direct tool produced a formatted result and model was generic, prepend tool result
        if tool_result and intent_info["intent"] in ["calculation", "datetime"]:
            response_text = f"{tool_result}\n\n{response_text}"

        # 9. Append Citations to Response
        if search_citations:
            citation_lines = ["\n\n### Verifiable Sources & Citations:"]
            for c in search_citations:
                citation_lines.append(f"- [[{c['index']}]] **{c['title']}** — [{c['source']}]({c['url']})")
            response_text += "\n".join(citation_lines)

        if doc_citations:
            citation_lines = ["\n\n### Knowledge Base Citations:"]
            for dc in doc_citations:
                citation_lines.append(f"- [[Doc {dc['citation_id']}]] `{dc['document']}` (Section {dc['chunk_index']})")
            response_text += "\n".join(citation_lines)

        # 10. Record Assistant Response in SQLite
        tool_calls_payload = []
        if tool_executed:
            tool_calls_payload.append({"tool": tool_executed, "time": round(time.time() - t0, 3)})

        conversation_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
            model_name=gen_output.get("model", "omega-brain"),
            provider=gen_output.get("provider", provider),
            tool_calls=tool_calls_payload,
        )

        # 11. Learn Key User Facts into Long-Term Memory
        if clean_query.lower().startswith("remember that ") or clean_query.lower().startswith("my preference is "):
            fact = clean_query.replace("remember that ", "").replace("my preference is ", "")
            memory_engine.add_memory(
                memory_type="long_term",
                category="user_preference",
                content=fact,
                importance_score=0.9,
            )

        elapsed_ms = round((time.time() - t0) * 1000, 2)
        return {
            "conversation_id": conversation_id,
            "response": response_text,
            "thought_chain": thought_chain,
            "provider": gen_output.get("provider", provider),
            "model": gen_output.get("model", "omega-brain"),
            "tool_executed": tool_executed,
            "search_citations": search_citations,
            "doc_citations": doc_citations,
            "latency_ms": elapsed_ms,
        }


# Singleton instance
generator_engine = GeneratorEngine()
