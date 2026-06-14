"""Compatibility helpers for the ReAct prompt used by member 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import ToolResult
from .calculator import CalculatorTool
from .file_io import ReadFileTool, WriteFileTool
from .registry import ToolRegistry
from .wikipedia_search import WikipediaSearchTool


def execute_react_tool(
    registry: ToolRegistry,
    tool_name: str,
    tool_input: str,
) -> str:
    """Execute the exact Calculator/Wikipedia_Search/File_IO prompt protocol."""

    normalized_name = tool_name.strip().lower().replace("_", "").replace(" ", "")

    if normalized_name in {"calculator", "calculate"}:
        result = registry.execute("calculator", {"expression": tool_input})
    elif normalized_name in {"wikipediasearch", "wikipedia", "wikisearch"}:
        result = registry.execute("wikipedia_search", {"query": tool_input})
    elif normalized_name in {"fileio", "file", "localfile"}:
        result = _execute_file_io(registry, tool_input)
    else:
        result = ToolResult(
            tool=tool_name,
            success=False,
            error=(
                f"未知工具：{tool_name}。可用工具为 "
                "Calculator、Wikipedia_Search、File_IO"
            ),
            error_type="ToolNotFoundError",
        )

    return result.to_observation()


def calculate(expression: str) -> str:
    """Drop-in replacement for member 2's tools.calculate()."""

    return CalculatorTool().execute(expression=expression).to_observation()


def wikipedia_search(query: str) -> str:
    """Drop-in replacement for member 2's tools.wikipedia_search()."""

    return WikipediaSearchTool().execute(query=query).to_observation()


def file_io(action: str, filename: str, content: str = "") -> str:
    """Drop-in replacement for member 2's tools.file_io()."""

    root_dir = Path(os.getenv("AGENT_WORKSPACE_DIR", "."))
    registry = ToolRegistry(
        [
            ReadFileTool(root_dir),
            WriteFileTool(root_dir),
        ]
    )
    tool_input = json.dumps(
        {
            "action": action,
            "filename": filename,
            "content": content,
        },
        ensure_ascii=False,
    )
    return _execute_file_io(registry, tool_input).to_observation()


def _execute_file_io(registry: ToolRegistry, tool_input: str) -> ToolResult:
    try:
        params: Any = json.loads(tool_input)
    except json.JSONDecodeError as exc:
        return ToolResult(
            tool="File_IO",
            success=False,
            error=f"File_IO 输入必须是合法的单行 JSON：{exc.msg}",
            error_type="ToolInputError",
        )

    if not isinstance(params, dict):
        return ToolResult(
            tool="File_IO",
            success=False,
            error="File_IO JSON 参数必须是对象",
            error_type="ToolInputError",
        )

    action = params.get("action")
    filename = params.get("filename")
    content = params.get("content", "")

    if not isinstance(action, str) or not action.strip():
        return ToolResult(
            tool="File_IO",
            success=False,
            error="File_IO 缺少字符串参数 action",
            error_type="ToolInputError",
        )
    if not isinstance(filename, str) or not filename.strip():
        return ToolResult(
            tool="File_IO",
            success=False,
            error="File_IO 缺少字符串参数 filename",
            error_type="ToolInputError",
        )

    normalized_action = action.strip().lower()
    if normalized_action == "read":
        return registry.execute("read_file", {"path": filename})
    if normalized_action in {"write", "create"}:
        return registry.execute(
            "write_file",
            {"path": filename, "content": content, "append": False},
        )
    if normalized_action == "append":
        return registry.execute(
            "write_file",
            {"path": filename, "content": content, "append": True},
        )

    return ToolResult(
        tool="File_IO",
        success=False,
        error="action 仅支持 read、write、create 或 append",
        error_type="ToolInputError",
    )
