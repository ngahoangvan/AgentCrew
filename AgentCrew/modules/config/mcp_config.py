import json
import os
from typing import Any, Dict


class MCPConfig:
    """Manages mcp_servers.json — MCP server definitions."""

    @property
    def _path(self) -> str:
        return os.getenv("MCP_CONFIG_PATH", os.path.expanduser("./mcp_servers.json"))

    def read(self) -> Dict[str, Any]:
        """Return MCP server config, or {} on error."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def write(self, config_data: Dict[str, Any]) -> None:
        """Persist config_data and trigger agent reload."""
        from AgentCrew.modules.config.agents_config import AgentsConfig

        try:
            dir_path = os.path.dirname(self._path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)
            AgentsConfig().reload()
        except Exception as e:
            raise ValueError(f"Error writing MCP configuration: {str(e)}")
