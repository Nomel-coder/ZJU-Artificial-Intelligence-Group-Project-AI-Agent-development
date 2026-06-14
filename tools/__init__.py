"""Public API for the project's Agent tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult, ToolSpec
from .calculator import CalculatorTool
from .file_io import ListFilesTool, ReadFileTool, WriteFileTool
from .react_compat import calculate, execute_react_tool, file_io, wikipedia_search
from .registry import ToolRegistry
from .wikipedia_search import WikipediaSearchTool


def create_default_registry(
    file_root: str | Path = "agent_files",
    *,
    wikipedia_module: Any | None = None,
) -> ToolRegistry:
    """Create the standard tool set used by the Agent."""

    return ToolRegistry(
        [
            CalculatorTool(),
            WikipediaSearchTool(wikipedia_module=wikipedia_module),
            ReadFileTool(file_root),
            WriteFileTool(file_root),
            ListFilesTool(file_root),
        ]
    )


__all__ = [
    "BaseTool",
    "CalculatorTool",
    "ListFilesTool",
    "ReadFileTool",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "WikipediaSearchTool",
    "WriteFileTool",
    "calculate",
    "create_default_registry",
    "execute_react_tool",
    "file_io",
    "wikipedia_search",
]
