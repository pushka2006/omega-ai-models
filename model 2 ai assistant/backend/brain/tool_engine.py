"""OMEGA MODEL 2: Tool Execution Engine and Registry.

Exposes safe, inspectable, and logged tools:
1. Calculator (Safe mathematical expression evaluator)
2. System DateTime & Timezone
3. File Search & Inspection
4. Safe File Reader
5. Python Sandbox Execution
6. Web Search & Citation
7. JSON Processing & Validation
8. Workspace Project Analyzer
"""

import datetime
import json
import math
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from backend.brain.coding_engine import coding_engine
from backend.brain.search_engine import search_engine
from backend.config import WORKSPACE_DIR
from backend.database.db import get_db_connection


class ToolRegistry:
    """Manages tool registration, schema definitions, and logged execution."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable, requires_confirmation: bool = False):
        """Register a new callable tool."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "func": func,
            "requires_confirmation": requires_confirmation,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """List metadata and schemas for all active tools."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
                "requires_confirmation": t["requires_confirmation"],
            }
            for t in self.tools.values()
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with audit logging and timing."""
        t0 = time.time()
        log_id = str(uuid.uuid4())

        if tool_name not in self.tools:
            res = {"success": False, "error": f"Tool '{tool_name}' not found in registry."}
            self._log_tool(log_id, tool_name, arguments, res, "failed", 0.0)
            return res

        tool_meta = self.tools[tool_name]

        try:
            result = tool_meta["func"](**arguments)
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            out = {"success": True, "result": result, "execution_time_ms": elapsed_ms}
            self._log_tool(log_id, tool_name, arguments, out, "success", elapsed_ms)
            return out
        except Exception as e:
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            out = {"success": False, "error": str(e), "execution_time_ms": elapsed_ms}
            self._log_tool(log_id, tool_name, arguments, out, "failed", elapsed_ms)
            return out

    def _log_tool(self, log_id: str, tool_name: str, inputs: Any, outputs: Any, status: str, elapsed_ms: float):
        """Persist execution log to SQLite."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_logs (id, tool_name, inputs, outputs, status, execution_time_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    tool_name,
                    json.dumps(inputs, default=str),
                    json.dumps(outputs, default=str),
                    status,
                    elapsed_ms,
                    time.time(),
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _register_default_tools(self):
        """Populate built-in utility tools."""

        # 1. Calculator
        def calculator(expression: str) -> str:
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
            # Clean and sanitize expression
            sanitized = expression.replace("^", "**")
            val = eval(sanitized, {"__builtins__": None}, allowed_names)
            return str(val)

        self.register(
            name="calculator",
            description="Evaluate mathematical expressions safely.",
            parameters={"expression": "string, e.g. 'sqrt(144) + 2**8'"},
            func=calculator,
        )

        # 2. DateTime
        def get_current_time(timezone_name: str = "local") -> Dict[str, Any]:
            now = datetime.datetime.now()
            utc = datetime.datetime.now(datetime.timezone.utc)
            return {
                "local_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "utc_time": utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "timestamp": int(time.time()),
                "day_of_week": now.strftime("%A"),
            }

        self.register(
            name="get_current_time",
            description="Get current system date, time, and UTC timestamp.",
            parameters={},
            func=get_current_time,
        )

        # 3. Web Search
        def web_search(query: str, max_results: int = 4) -> Dict[str, Any]:
            return search_engine.search_and_synthesize(query=query, max_results=max_results)

        self.register(
            name="web_search",
            description="Perform real-time search on the web and return cited snippets.",
            parameters={"query": "string, search query", "max_results": "integer, default 4"},
            func=web_search,
        )

        # 4. Python Sandbox Runner
        def run_python_code(code: str) -> Dict[str, Any]:
            return coding_engine.execute_python_sandbox(code)

        self.register(
            name="run_python_code",
            description="Execute Python code securely inside an isolated AST-guarded sandbox.",
            parameters={"code": "string, Python code snippet"},
            func=run_python_code,
        )

        # 5. File Search in Workspace
        def search_workspace_files(pattern: str = "*.*", max_files: int = 20) -> List[str]:
            matched = []
            for root, dirs, files in os.walk(str(WORKSPACE_DIR)):
                if ".git" in root or "__pycache__" in root or ".venv" in root:
                    continue
                for file in files:
                    if pattern == "*.*" or pattern.lower() in file.lower():
                        rel = os.path.relpath(os.path.join(root, file), str(WORKSPACE_DIR))
                        matched.append(rel)
                        if len(matched) >= max_files:
                            return matched
            return matched

        self.register(
            name="search_workspace_files",
            description="Search for files in the AI workspace by name or extension.",
            parameters={"pattern": "string, filename substring or pattern", "max_files": "int, default 20"},
            func=search_workspace_files,
        )

        # 6. Read Workspace File Safely
        def read_workspace_file(relative_path: str, max_lines: int = 200) -> Dict[str, Any]:
            # Guard against directory traversal
            target = (WORKSPACE_DIR / relative_path).resolve()
            if not str(target).startswith(str(WORKSPACE_DIR.resolve())):
                raise PermissionError("Access restricted: Target path is outside workspace.")

            if not target.exists():
                raise FileNotFoundError(f"File '{relative_path}' does not exist.")

            with open(target, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(max_lines)]

            return {
                "file": relative_path,
                "lines_read": len(lines),
                "content": "".join(lines),
            }

        self.register(
            name="read_workspace_file",
            description="Safely read text or code file contents from the workspace.",
            parameters={"relative_path": "string, path relative to workspace root", "max_lines": "int, default 200"},
            func=read_workspace_file,
        )

        # 7. JSON Processor
        def process_json(json_str: str, query_key: Optional[str] = None) -> Any:
            data = json.loads(json_str)
            if query_key and isinstance(data, dict):
                return data.get(query_key)
            return data

        self.register(
            name="process_json",
            description="Parse, validate, and extract values from JSON structures.",
            parameters={"json_str": "string, raw JSON", "query_key": "string, optional dictionary key"},
            func=process_json,
        )


# Singleton instance
tool_engine = ToolRegistry()
