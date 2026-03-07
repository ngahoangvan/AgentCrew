from typing import Any


def extract_tool_name(tool_def: Any) -> str:
    """
    Extract a tool name from a definition dict regardless of provider format.

    Supports:
      - Claude / generic format: {"name": "...", ...}
      - OpenAI function format:  {"function": {"name": "..."}, ...}

    Args:
        tool_def: The tool definition mapping.

    Returns:
        The tool name string.

    Raises:
        ValueError: If the tool name cannot be extracted.
    """
    if isinstance(tool_def, dict):
        if "name" in tool_def:
            return tool_def["name"]
        if "function" in tool_def and "name" in tool_def["function"]:
            return tool_def["function"]["name"]
    raise ValueError(f"Could not extract tool name from definition: {tool_def!r}")
