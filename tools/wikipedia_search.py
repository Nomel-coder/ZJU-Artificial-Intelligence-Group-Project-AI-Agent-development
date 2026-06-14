"""Wikipedia search tool with concise, Agent-friendly output."""

from __future__ import annotations

from typing import Any

from .base import BaseTool, ToolSpec
from .exceptions import ToolExecutionError, ToolInputError


class WikipediaSearchTool(BaseTool):
    spec = ToolSpec(
        name="wikipedia_search",
        description="在 Wikipedia 中搜索主题并返回摘要、页面标题和链接。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要搜索的主题"},
                "language": {
                    "type": "string",
                    "description": "Wikipedia 语言代码，默认按查询文本自动选择 zh/en",
                    "default": "auto",
                },
                "sentences": {
                    "type": "integer",
                    "description": "摘要句数，范围 1-10，默认 3",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    )

    def __init__(self, wikipedia_module: Any | None = None) -> None:
        self._wikipedia = wikipedia_module

    def _get_module(self) -> Any:
        if self._wikipedia is not None:
            return self._wikipedia
        try:
            import wikipedia
        except ImportError as exc:
            raise ToolExecutionError(
                "缺少 wikipedia 依赖，请先安装 requirements.txt"
            ) from exc
        return wikipedia

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query")
        language = kwargs.get("language", "auto")
        sentences = kwargs.get("sentences", 3)

        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query 必须是非空字符串")
        if not isinstance(language, str) or not language.strip():
            raise ToolInputError("language 必须是有效的语言代码")
        if isinstance(sentences, bool) or not isinstance(sentences, int):
            raise ToolInputError("sentences 必须是整数")
        if not 1 <= sentences <= 10:
            raise ToolInputError("sentences 必须在 1 到 10 之间")

        normalized_language = language.strip().lower()
        if normalized_language == "auto":
            normalized_language = (
                "zh" if any("\u4e00" <= character <= "\u9fff" for character in query) else "en"
            )

        wikipedia = self._get_module()
        wikipedia.set_lang(normalized_language)

        try:
            page = wikipedia.page(query.strip(), auto_suggest=True)
            summary = wikipedia.summary(
                page.title,
                sentences=sentences,
                auto_suggest=False,
            )
        except wikipedia.exceptions.DisambiguationError as exc:
            options = list(exc.options[:10])
            raise ToolInputError(
                f"搜索词存在歧义，请提供更具体的主题。候选项：{', '.join(options)}"
            ) from exc
        except wikipedia.exceptions.PageError as exc:
            raise ToolExecutionError(f"未找到 Wikipedia 页面：{query.strip()}") from exc
        except Exception as exc:
            raise ToolExecutionError(f"Wikipedia 请求失败：{exc}") from exc

        return {
            "title": page.title,
            "summary": summary,
            "url": page.url,
            "language": normalized_language,
        }
