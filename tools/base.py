"""Common interfaces shared by all Agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
from typing import Any

from .exceptions import ToolError


@dataclass(frozen=True)
class ToolSpec:
    """Metadata exposed to the prompt and Tool Router."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolResult:
    """A stable, JSON-serializable result returned by every tool."""

    tool: str
    success: bool
    output: Any = None
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_observation(self) -> str:
        """Convert the result into text that can be appended to a ReAct prompt."""

        if self.success:
            payload = self.output
        else:
            payload = {
                "success": False,
                "error_type": self.error_type,
                "error": self.error,
            }

        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)


class BaseTool(ABC):
    """Base class that converts exceptions into structured results."""

    spec: ToolSpec

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            output = self._run(**kwargs)
            return ToolResult(tool=self.spec.name, success=True, output=output)
        except ToolError as exc:
            return ToolResult(
                tool=self.spec.name,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )
        except Exception as exc:  # Keep an unexpected tool failure out of the Agent loop.
            return ToolResult(
                tool=self.spec.name,
                success=False,
                error=f"工具执行失败：{exc}",
                error_type=type(exc).__name__,
            )

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Execute the tool and return JSON-serializable output."""
