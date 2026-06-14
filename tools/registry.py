"""Tool registration and dispatch used by the Agent's Tool Router."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .base import BaseTool, ToolResult


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"工具名称重复：{name}")
        self._tools[name] = tool

    def get_specs(self) -> list[dict[str, Any]]:
        return [tool.spec.to_dict() for tool in self._tools.values()]

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str | None = None,
    ) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool=tool_name,
                success=False,
                error=f"未知工具：{tool_name}",
                error_type="ToolNotFoundError",
            )

        normalized = self._normalize_arguments(tool_name, arguments)
        if isinstance(normalized, ToolResult):
            return normalized
        return tool.execute(**normalized)

    @staticmethod
    def _normalize_arguments(
        tool_name: str,
        arguments: dict[str, Any] | str | None,
    ) -> dict[str, Any] | ToolResult:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            return ToolResult(
                tool=tool_name,
                success=False,
                error="工具参数必须是对象或 JSON 字符串",
                error_type="ToolInputError",
            )

        text = arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            single_string_parameters = {
                "calculator": "expression",
                "wikipedia_search": "query",
                "read_file": "path",
                "list_files": "path",
            }
            parameter = single_string_parameters.get(tool_name)
            if parameter is None:
                return ToolResult(
                    tool=tool_name,
                    success=False,
                    error="该工具的纯文本参数不明确，请传入 JSON 对象",
                    error_type="ToolInputError",
                )
            return {parameter: text}

        if not isinstance(parsed, dict):
            return ToolResult(
                tool=tool_name,
                success=False,
                error="JSON 工具参数必须是对象",
                error_type="ToolInputError",
            )
        return parsed
