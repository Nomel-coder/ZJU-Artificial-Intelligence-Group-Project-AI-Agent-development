"""
Context Manager — Token 估算与上下文窗口管理
==============================================
为 DeepSeek API 提供混合中英文的 token 计数和滑动窗口裁剪。
不依赖 tiktoken（DeepSeek 使用自己的 tokenizer），采用保守的字符级估算。

Author: 林放 (Member 4)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional


# ---- 字符集检测 ----

_ZH_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
_EN_RE = re.compile(r"[a-zA-Z]")
_DIGIT_RE = re.compile(r"\d")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _count_chars(text: str) -> dict[str, int]:
    """统计文本中各类字符的数量。"""
    return {
        "zh": len(_ZH_RE.findall(text)),
        "en": len(_EN_RE.findall(text)),
        "digit": len(_DIGIT_RE.findall(text)),
        "punct": len(_PUNCT_RE.findall(text)),
        "other": len(text) - len(_ZH_RE.findall(text)) - len(_EN_RE.findall(text))
                  - len(_DIGIT_RE.findall(text)) - len(_PUNCT_RE.findall(text)),
    }


# ---- Token 估算 ----

def estimate_tokens(text: str) -> int:
    """
    保守估算中英文混合文本的 token 数。

    基于经验值：
    - 一个中文字符 ≈ 1.5 token（实际 1.2~2.0，取偏保守值）
    - 一个英文字母/数字 ≈ 0.25 token（实际 ~0.25）
    - 标点/空格 ≈ 0.3 token
    - 其他字符 ≈ 1 token（保守）

    总估算 = ceil(中文×1.5 + 英文/数字×0.25 + 标点/空格×0.3 + 其他×1.0)
    """
    counts = _count_chars(text)
    raw = (
        counts["zh"] * 1.5
        + (counts["en"] + counts["digit"]) * 0.25
        + counts["punct"] * 0.3
        + counts["other"] * 1.0
    )
    return max(1, int(raw + 0.5))  # 向上取整，最小为 1


# ---- 消息级别 Token 估算 ----

# 每条消息的 role 字段 ≈ 4 token 的 overhead
_MESSAGE_OVERHEAD = 4
# 整体 overhead（包括 JSON 结构等）
_BATCH_OVERHEAD = 3


def estimate_message_tokens(message: dict[str, str]) -> int:
    """估算单条 OpenAI 格式消息的 token 数。"""
    tokens = _MESSAGE_OVERHEAD
    for key, value in message.items():
        if isinstance(value, str):
            tokens += estimate_tokens(value)
    return tokens


def estimate_message_list_tokens(messages: List[dict[str, str]]) -> int:
    """估算消息列表的总 token 数。"""
    if not messages:
        return 0
    total = _BATCH_OVERHEAD
    for msg in messages:
        total += estimate_message_tokens(msg)
    return max(1, total)


# ---- ContextManager ----

@dataclass
class ContextStats:
    """上下文使用统计。"""
    total_tokens: int = 0
    max_tokens: int = 4096
    message_count: int = 0
    usage_ratio: float = 0.0


class ContextManager:
    """
    上下文窗口管理器。

    职责：
    - 估算消息列表的 token 用量
    - 在超出限制时按规则裁剪历史消息
    - 提供安全检查和用量报告

    裁剪策略（trim_to_fit）：
    1. 始终保留 system prompt（第一条消息）
    2. 从旧到新逐条丢弃，直到 token 总数回到限制以内
    3. 不会把需要生成的空间（reserved_tokens）也用掉

    Usage::

        ctx = ContextManager(max_tokens=4096, reserved_tokens=500)
        messages = [{"role": "system", "content": "..."}, ...]
        if not ctx.is_safe(messages):
            messages = ctx.trim_to_fit(messages)
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        reserved_tokens: int = 500,
    ) -> None:
        """
        Args:
            max_tokens: 模型上下文窗口上限。
            reserved_tokens: 预留给模型生成输出的 token 数。
        """
        if reserved_tokens >= max_tokens:
            raise ValueError("reserved_tokens 必须小于 max_tokens")
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        # 可用于输入的实际上限
        self.input_limit = max_tokens - reserved_tokens

    # ---- Token 估算（委托给模块级函数）----

    def count_tokens(self, text: str) -> int:
        """估算文本的 token 数。"""
        return estimate_tokens(text)

    def count_message_tokens(self, message: dict[str, str]) -> int:
        """估算单条消息的 token 数。"""
        return estimate_message_tokens(message)

    def count_message_list_tokens(self, messages: List[dict[str, str]]) -> int:
        """估算消息列表的总 token 数。"""
        return estimate_message_list_tokens(messages)

    # ---- 安全检查 ----

    def is_safe(self, messages: List[dict[str, str]], threshold: float = 0.75) -> bool:
        """
        检查上下文用量是否在安全范围内。

        Args:
            messages: 消息列表。
            threshold: 安全阈值（相对于 input_limit 的比例，默认 75%）。

        Returns:
            True 如果 token 数在安全范围内。
        """
        current = self.count_message_list_tokens(messages)
        return current < self.input_limit * threshold

    def usage_ratio(self, messages: List[dict[str, str]]) -> float:
        """返回上下文占用比例（0.0 ~ 1.0+）。"""
        current = self.count_message_list_tokens(messages)
        return current / self.input_limit if self.input_limit > 0 else 1.0

    def get_stats(self, messages: List[dict[str, str]]) -> ContextStats:
        """获取当前上下文使用统计。"""
        total = self.count_message_list_tokens(messages)
        return ContextStats(
            total_tokens=total,
            max_tokens=self.max_tokens,
            message_count=len(messages),
            usage_ratio=self.usage_ratio(messages),
        )

    # ---- 裁剪 ----

    def trim_to_fit(
        self,
        messages: List[dict[str, str]],
        *,
        preserve_system: bool = True,
    ) -> List[dict[str, str]]:
        """
        将消息列表裁剪到 input_limit 以内。

        策略：
        1. 如果已经安全，直接返回。
        2. 保留 system prompt。
        3. 从最早的消息开始丢弃（system prompt 之后）。
        4. 始终保留最新一条 user 消息（当前提问）。

        Args:
            messages: 要裁剪的消息列表。
            preserve_system: 是否保留第一条 system 消息，默认 True。

        Returns:
            裁剪后的消息列表（可能是原列表的副本）。
        """
        if not messages:
            return []

        current_tokens = self.count_message_list_tokens(messages)
        if current_tokens <= self.input_limit:
            return messages

        # 分离 system prompt
        start = 1 if (preserve_system and messages and messages[0].get("role") == "system") else 0

        # 从 system prompt 之后开始，逐条丢弃最早的消息
        # 但至少保留最后一条 user 消息
        trimmed = list(messages)
        while len(trimmed) > start + 1:
            current_tokens = self.count_message_list_tokens(trimmed)
            if current_tokens <= self.input_limit:
                return trimmed
            # 丢弃 system prompt 之后的第一条消息
            del trimmed[start]

        # 如果还是超限且只剩 system + 最后一条消息，
        # 则截断最后一条消息的内容
        if len(trimmed) <= start + 1 and self.count_message_list_tokens(trimmed) > self.input_limit:
            trimmed = self._truncate_last_message(trimmed, start)

        return trimmed

    def _truncate_last_message(
        self,
        messages: List[dict[str, str]],
        start: int,
    ) -> List[dict[str, str]]:
        """截断最后一条消息的内容，使其适应窗口限制。"""
        if len(messages) <= start:
            return messages

        result = list(messages)
        last = result[-1]
        content = last.get("content", "")

        # 留出 overhead 空间
        overhead = sum(
            estimate_message_tokens(result[i]) - estimate_tokens(result[i].get("content", ""))
            for i in range(len(result))
            if i != len(result) - 1
        )
        available = self.input_limit - overhead - estimate_message_tokens({})  # 当前这条的 overhead

        while estimate_tokens(content) > available and content:
            # 每次截短约 20%
            cut = max(1, len(content) // 5)
            content = content[:-cut]

        result[-1] = {**last, "content": content + "\n...[上下文过长，已截断]"}
        return result

    # ---- 带统计的裁剪 ----

    def trim_with_stats(
        self,
        messages: List[dict[str, str]],
    ) -> tuple[List[dict[str, str]], ContextStats, ContextStats]:
        """
        裁剪并返回裁剪前后的统计信息。

        Returns:
            (裁剪后消息列表, 裁剪前统计, 裁剪后统计)
        """
        before = self.get_stats(messages)
        trimmed = self.trim_to_fit(messages)
        after = self.get_stats(trimmed)
        return trimmed, before, after
