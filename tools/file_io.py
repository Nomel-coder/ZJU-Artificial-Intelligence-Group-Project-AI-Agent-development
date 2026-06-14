"""Sandboxed local text-file tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolSpec
from .exceptions import ToolExecutionError, ToolInputError


class SandboxedFileTool(BaseTool):
    """Base class for tools restricted to one configured directory."""

    def __init__(self, root_dir: str | Path, *, max_bytes: int = 1_000_000) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _resolve_path(self, raw_path: Any) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolInputError("path 必须是非空字符串")
        if Path(raw_path).is_absolute():
            raise ToolInputError("path 必须使用相对于工具工作目录的路径")

        candidate = (self.root_dir / raw_path).resolve()
        try:
            candidate.relative_to(self.root_dir)
        except ValueError as exc:
            raise ToolInputError("禁止访问工具工作目录以外的文件") from exc
        return candidate


class ReadFileTool(SandboxedFileTool):
    spec = ToolSpec(
        name="read_file",
        description="读取工具工作目录内的 UTF-8 文本文件。",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "相对文件路径"}},
            "required": ["path"],
        },
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        path = self._resolve_path(kwargs.get("path"))
        if not path.exists():
            raise ToolExecutionError(f"文件不存在：{path.relative_to(self.root_dir)}")
        if not path.is_file():
            raise ToolInputError("指定路径不是文件")
        if path.stat().st_size > self.max_bytes:
            raise ToolInputError(f"文件过大，最多允许读取 {self.max_bytes} 字节")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError("文件不是有效的 UTF-8 文本") from exc
        except OSError as exc:
            raise ToolExecutionError(f"读取文件失败：{exc}") from exc

        return {
            "path": path.relative_to(self.root_dir).as_posix(),
            "content": content,
            "size": path.stat().st_size,
        }


class WriteFileTool(SandboxedFileTool):
    spec = ToolSpec(
        name="write_file",
        description="写入工具工作目录内的 UTF-8 文本文件，可选择覆盖或追加。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对文件路径"},
                "content": {"type": "string", "description": "要写入的文本"},
                "append": {
                    "type": "boolean",
                    "description": "是否追加内容，默认 false",
                    "default": False,
                },
            },
            "required": ["path", "content"],
        },
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        path = self._resolve_path(kwargs.get("path"))
        content = kwargs.get("content")
        append = kwargs.get("append", False)

        if not isinstance(content, str):
            raise ToolInputError("content 必须是字符串")
        if not isinstance(append, bool):
            raise ToolInputError("append 必须是布尔值")

        encoded_size = len(content.encode("utf-8"))
        existing_size = path.stat().st_size if append and path.exists() else 0
        if existing_size + encoded_size > self.max_bytes:
            raise ToolInputError(f"写入后文件最多允许 {self.max_bytes} 字节")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a" if append else "w", encoding="utf-8", newline="") as file:
                file.write(content)
        except OSError as exc:
            raise ToolExecutionError(f"写入文件失败：{exc}") from exc

        return {
            "path": path.relative_to(self.root_dir).as_posix(),
            "bytes_written": encoded_size,
            "mode": "append" if append else "overwrite",
        }


class ListFilesTool(SandboxedFileTool):
    spec = ToolSpec(
        name="list_files",
        description="列出工具工作目录内某个目录中的文件和子目录。",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对目录路径，默认当前目录",
                    "default": ".",
                }
            },
        },
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        path = self._resolve_path(kwargs.get("path", "."))
        if not path.exists():
            raise ToolExecutionError(f"目录不存在：{path.relative_to(self.root_dir)}")
        if not path.is_dir():
            raise ToolInputError("指定路径不是目录")

        try:
            entries = [
                {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                }
                for item in sorted(path.iterdir(), key=lambda item: item.name.lower())
            ]
        except OSError as exc:
            raise ToolExecutionError(f"列出目录失败：{exc}") from exc

        return {
            "path": path.relative_to(self.root_dir).as_posix() or ".",
            "entries": entries,
        }
