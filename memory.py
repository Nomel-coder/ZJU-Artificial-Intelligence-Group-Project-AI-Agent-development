"""
Memory Module — 短期记忆与上下文压缩
====================================
通过抽取式压缩将旧轮对话精简为一行摘要，控制上下文窗口不爆炸。

策略：
- 保留最近 keep_verbatim 轮完整对话（高保真）
- 更早的轮次压缩为一行摘要：
  "[历史#N] 问: ... → 答: ..."
- 可选的 LLM 摘要（当上下文极度紧张时触发）

Author: 林放 (Member 4)
"""

from __future__ import annotations

from typing import List, Optional

from context_manager import ContextManager, estimate_tokens
from history_manager import ConversationTurn, HistoryManager


class ShortTermMemory:
    """
    短期记忆管理器。

    基于抽取式压缩：将旧轮对话精简为一行摘要，
    保留最近几轮的完整内容。

    Usage::

        memory = ShortTermMemory(keep_verbatim=3, context_manager=ctx)
        messages = memory.build_context(
            history.turns, user_message, system_prompt
        )
    """

    # 压缩摘要的 token overhead（前缀 "[历史#N]" 等）
    _COMPRESSION_OVERHEAD = 8

    def __init__(
        self,
        keep_verbatim: int = 3,
        context_manager: Optional[ContextManager] = None,
    ) -> None:
        """
        Args:
            keep_verbatim: 保留完整内容的最新轮次数。
            context_manager: 可选的 ContextManager，用于智能触发压缩。
        """
        self.keep_verbatim = keep_verbatim
        self.ctx = context_manager or ContextManager()

    # ---- 压缩 ----

    def compress_turn(self, turn: ConversationTurn) -> str:
        """
        将一轮对话压缩为一句话摘要。

        Args:
            turn: 要压缩的对话轮次。

        Returns:
            压缩后的单行文本。
        """
        user_preview = self._truncate(turn.user_message, 50)
        if turn.final_answer:
            ans_preview = self._truncate(turn.final_answer, 80)
            return f"[历史#{turn.turn_id}] 问: {user_preview} → 答: {ans_preview}"
        elif turn.action:
            return f"[历史#{turn.turn_id}] 问: {user_preview} → 调用 {turn.action}"
        else:
            return f"[历史#{turn.turn_id}] 问: {user_preview} → (未完成)"

    def compress_turns(self, turns: List[ConversationTurn]) -> str:
        """
        压缩多轮对话为一个摘要块。

        Args:
            turns: 要压缩的轮次列表。

        Returns:
            格式化的压缩摘要。
        """
        if not turns:
            return ""

        summaries = [self.compress_turn(t) for t in turns]

        # 如果压缩后的总 token 也比较大，只保留首尾
        full = "\n".join(summaries)
        if estimate_tokens(full) > 500:
            # 保留第一轮和最后一轮
            kept = [summaries[0]]
            if len(summaries) > 1:
                kept.append(f"  ... 省略 {len(summaries) - 2} 轮 ...")
                kept.append(summaries[-1])
            full = "\n".join(kept)

        return full

    # ---- 上下文构建 ----

    def build_context(
        self,
        turns: List[ConversationTurn],
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> List[dict]:
        """
        构建含记忆的消息列表。

        策略：
        1. system prompt（始终保留）
        2. 压缩的旧轮历史（如果轮次 > keep_verbatim）
        3. 最近 keep_verbatim 轮的完整对话
        4. 当前用户消息

        Args:
            turns: 所有历史轮次。
            user_message: 当前用户提问。
            system_prompt: 系统提示词。

        Returns:
            OpenAI 格式的消息列表。
        """
        messages: List[dict] = []

        # 1. System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. 分离旧轮和新轮
        if len(turns) > self.keep_verbatim:
            old_turns = turns[:-self.keep_verbatim]
            recent_turns = turns[-self.keep_verbatim:]

            # 压缩旧轮
            compressed = self.compress_turns(list(old_turns))
            if compressed:
                memory_content = (
                    "以下是之前对话的摘要（较旧的对话已压缩）：\n"
                    f"{compressed}\n"
                    "--- 以下是最近的完整对话 ---"
                )
                messages.append({"role": "system", "content": memory_content})
        else:
            recent_turns = turns

        # 3. 最近轮次的完整消息
        for turn in recent_turns:
            messages.extend(turn.to_message_pair())

        # 4. 当前用户消息
        messages.append({"role": "user", "content": user_message})

        return messages

    def build_context_with_observations(
        self,
        turns: List[ConversationTurn],
        system_prompt: str,
    ) -> List[dict]:
        """
        构建含完整 ReAct 追踪的上下文（用于 agent 内部 loop）。

        与 build_context 不同，这个方法保留 Observation 反馈消息。

        Args:
            turns: 所有历史轮次。
            system_prompt: Agent 系统提示词。

        Returns:
            含 ReAct 追踪的消息列表。
        """
        messages: List[dict] = [{"role": "system", "content": system_prompt}]

        # 压缩旧轮
        if len(turns) > self.keep_verbatim:
            old_turns = turns[:-self.keep_verbatim]
            recent_turns = turns[-self.keep_verbatim:]

            compressed = self.compress_turns(list(old_turns))
            if compressed:
                messages.append({
                    "role": "system",
                    "content": f"[对话历史摘要]\n{compressed}\n---",
                })
        else:
            recent_turns = turns

        # 最近轮次的完整 ReAct 追踪
        for turn in recent_turns:
            messages.extend(turn.to_react_trace())

        return messages

    # ---- 智能触发 ----

    def should_compress(self, turns: List[ConversationTurn]) -> bool:
        """
        判断是否需要压缩历史。

        当轮次超过 keep_verbatim 时就需要压缩。
        """
        return len(turns) > self.keep_verbatim

    def should_llm_summarize(
        self,
        messages: List[dict],
        threshold: float = 0.90,
    ) -> bool:
        """
        判断是否需要 LLM 级摘要（上下文极度紧张时）。

        Args:
            messages: 当前消息列表。
            threshold: 触发 LLM 摘要的用量阈值。

        Returns:
            True 如果应该用 LLM 做深度摘要。
        """
        if len(messages) <= 3:
            return False
        return self.ctx.usage_ratio(messages) > threshold

    # ---- LLM 摘要（扩展功能）----

    def summarize_with_llm(
        self,
        turns: List[ConversationTurn],
        llm_client,
        model: str = "deepseek-chat",
    ) -> str:
        """
        使用 LLM 对旧对话做深度摘要。

        仅在上下文极度紧张时调用（should_llm_summarize 返回 True）。

        Args:
            turns: 要摘要的轮次。
            llm_client: OpenAI 兼容的客户端。
            model: 模型名称。

        Returns:
            LLM 生成的历史摘要。
        """
        # 构建摘要 prompt
        history_text = "\n".join(t.to_summary() for t in turns)

        summary_prompt = (
            "请将以下对话历史压缩为一段简洁的摘要（不超过 200 字），"
            "保留关键的事实信息、计算结果和用户意图：\n\n"
            f"{history_text}"
        )

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # 回退到提取式压缩
            return self.compress_turns(list(turns))

    # ---- 工具方法 ----

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """截断文本，超出部分用 ... 表示。"""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
