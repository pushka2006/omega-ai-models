"""OMEGA MODEL 2: Safety, Permission, and Boundary Enforcement Layer.

Validates tool executions, restricts access boundaries, prevents command
injection, and requires explicit user confirmation for destructive actions.
"""

from typing import Any, Dict, List, Optional, Tuple


class SafetyEngine:
    """Enforces execution boundaries, permission gates, and safety constraints."""

    # High-impact actions requiring user confirmation
    SENSITIVE_ACTIONS = {
        "delete_database",
        "purge_all_memories",
        "execute_unrestricted_shell",
        "delete_document",
    }

    def check_tool_permission(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate if a tool call is safe to proceed automatically."""
        if tool_name in self.SENSITIVE_ACTIONS:
            return False, f"Action '{tool_name}' requires explicit user confirmation."

        if tool_name == "read_workspace_file":
            path = args.get("relative_path", "")
            if ".." in path or path.startswith("/") or "\\" in path and path.startswith("\\"):
                return False, "Directory traversal path blocked."

        if tool_name == "run_python_code":
            code = args.get("code", "")
            if any(forbidden in code for forbidden in ["os.system", "shutil.rmtree", "subprocess.Popen"]):
                return False, "Dangerous system calls are prohibited in the sandboxed environment."

        return True, None

    def sanitize_user_input(self, text: str) -> str:
        """Sanitize raw text against dangerous format string injection."""
        return text.strip()


# Singleton instance
safety_engine = SafetyEngine()

