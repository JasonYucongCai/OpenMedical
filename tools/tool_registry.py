"""
tools.tool_registry — Tool Registry for agent tool discovery and invocation
============================================================================
Provides a central registry for all available tools. Tools can be registered,
discovered, and invoked by name. Supports schema generation for agent function
calling.
"""

from typing import Any, Callable
from .ct_analysis_tool import CTAnalysisTool


class ToolRegistry:
    """
    Registry of tools available for agent invocation.

    Usage:
        registry = ToolRegistry()
        registry.register(CTAnalysisTool())
        result = registry.invoke("ct_analysis", image_path="path/to/xray.png")
    """

    def __init__(self):
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        """Register a tool instance. Tool must have a `name` attribute."""
        name = getattr(tool, "name", None)
        if name is None:
            raise ValueError(f"Tool {tool} has no 'name' attribute.")
        self._tools[name] = tool
        print(f"[ToolRegistry] Registered: '{name}'")

    def get(self, name: str) -> Any:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def invoke(self, tool_name: str, **kwargs) -> Any:
        """
        Invoke a tool by name with keyword arguments.

        Example:
            registry.invoke("ct_analysis", image_path="path/to/image.png")
        """
        tool = self.get(tool_name)
        if hasattr(tool, "run"):
            return tool.run(**kwargs)
        raise AttributeError(f"Tool '{tool_name}' has no 'run' method.")

    def get_tool_descriptions(self) -> list[dict]:
        """
        Generate tool descriptions for agent function-calling schema.

        Returns a list of dicts with 'name', 'description', and 'parameters'
        suitable for OpenAI/DeepSeek function calling format.
        """
        descriptions = []
        for name, tool in self._tools.items():
            desc = {
                "name": name,
                "description": getattr(tool, "description", "No description available."),
            }
            descriptions.append(desc)
        return descriptions

    def get_prompt_context(self) -> str:
        """
        Generate a text block describing all available tools for the agent prompt.
        """
        if not self._tools:
            return "No tools available."

        lines = ["Available Tools:", "=" * 40]
        for name, tool in self._tools.items():
            desc = getattr(tool, "description", "No description")
            lines.append(f"\n  [{name}]")
            lines.append(f"    {desc}")
        return "\n".join(lines)


# ---- Singleton ----
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the singleton tool registry with default tools."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _registry.register(CTAnalysisTool())
    return _registry


def get_ct_analysis_tool() -> CTAnalysisTool:
    """Get the CT Analysis tool from the registry."""
    return get_tool_registry().get("ct_analysis")
