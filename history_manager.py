"""
History Manager — 结构化对话历史管理
====================================
将扁平的 messages 列表提升为结构化的 ConversationTurn 追踪，
支持消息列表重建、历史摘要和 token 统计。

每一轮完整的用户-Agent 交互被记录为一个 ConversationTurn：
- 用户提问
- Thought（思考过程）
- Action（工具调用）
- Observation（工具返回）
- Final Answer（最终回答）
- 时间戳和 token 统计

Author: 林放 (Member 4)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from context_manager import estimate_tokens


# ---- ConversationTurn ----

@dataclass
class ConversationTurn:
    """一轮完整的用户-Agent 对话。"""

    turn_id: int
    user_message: str = ""
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    success: bool = False

    def estimate_tokens(self) -> int:
        """估算该轮对话占用的 token 数。"""
        total = 0
        for field_name in (
            "user_message",
            "thought",
            "action",
            "action_input",
            "observation",
            "final_answer",
        ):
            value = getattr(self, field_name)
            if value:
                total += estimate_tokens(value)
        self.token_count = total
        return total

    def to_summary(self) -> str:
        """将本轮对话压缩为一行摘要。"""
        user_preview = self.user_message[:60] + ("..." if len(self.user_message) > 60 else "")
        if self.final_answer:
            ans_preview = self.final_answer[:80] + ("..." if len(self.final_answer) > 80 else "")
            return f"[Turn #{self.turn_id}] 问: {user_preview} → 答: {ans_preview}"
        if self.action:
            return f"[Turn #{self.turn_id}] 问: {user_preview} → 调用 {self.action}"
        return f"[Turn #{self.turn_id}] 问: {user_preview} → (未完成)"

    def to_message_pair(self) -> List[Dict[str, str]]:
        """
        将本轮对话转为 OpenAI 格式的 user/assistant 消息对。

        Returns:
            [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        """
        messages: List[Dict[str, str]] = []
        messages.append({"role": "user", "content": self.user_message})

        # 重建 assistant 输出
        assistant_parts = []
        if self.thought:
            assistant_parts.append(f"Thought: {self.thought}")
        if self.action and self.action_input:
            assistant_parts.append(f"Action: {self.action}")
            assistant_parts.append(f"Action Input: {self.action_input}")
        if self.final_answer:
            assistant_parts.append(f"Final Answer: {self.final_answer}")

        if assistant_parts:
            messages.append({"role": "assistant", "content": "\n".join(assistant_parts)})

        return messages

    def to_react_trace(self) -> List[Dict[str, str]]:
        """
        转为完整的 ReAct 追踪（含 observation 反馈）。

        用于重建 agent 内部的 loop_messages 格式：
        assistant（含 Thought/Action/Action Input）
        → user（含 Observation）
        → assistant（含 Final Answer）

        Returns:
            消息列表，可直接追加到 loop_messages 中。
        """
        trace: List[Dict[str, str]] = []

        # 用户提问
        trace.append({"role": "user", "content": self.user_message})

        # Assistant 输出
        assistant_content = []
        if self.thought:
            assistant_content.append(f"Thought: {self.thought}")
        if self.action and self.action_input:
            assistant_content.append(f"Action: {self.action}")
            assistant_content.append(f"Action Input: {self.action_input}")
        if self.final_answer:
            assistant_content.append(f"Final Answer: {self.final_answer}")

        if assistant_content:
            trace.append({"role": "assistant", "content": "\n".join(assistant_content)})

        # Observation（如果有工具调用）
        if self.observation:
            trace.append({"role": "user", "content": f"Observation: {self.observation}"})

        return trace


# ---- HistoryManager ----

@dataclass
class HistoryStats:
    """对话历史统计。"""
    total_turns: int = 0
    completed_turns: int = 0
    failed_turns: int = 0
    total_tokens: int = 0
    avg_tokens_per_turn: float = 0.0
    tools_used: Dict[str, int] = field(default_factory=dict)


class HistoryManager:
    """
    管理结构化对话历史。

    特性：
    - 上限控制（max_turns），超出时自动丢弃最早轮次
    - 支持按 turn_id 查询
    - 可转换为 OpenAI 消息列表
    - 生成历史摘要

    Usage::

        hist = HistoryManager(max_turns=10)
        tid = hist.start_turn("计算 3.14 * 10 的平方")
        hist.record_thought(tid, "我需要用 Calculator")
        hist.record_action(tid, "Calculator", "3.14 * 10 ** 2")
        hist.record_observation(tid, "314.0")
        hist.record_final_answer(tid, "结果是 314.0")
        hist.end_turn(tid, success=True)

        # 获取消息列表传给 LLM
        messages = hist.to_message_list()
    """

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._turns: Deque[ConversationTurn] = deque(maxlen=max_turns)
        self._counter: int = 0
        self._pending: Dict[int, ConversationTurn] = {}

    # ---- Turn 生命周期 ----

    def start_turn(self, user_message: str) -> int:
        """
        开始记录新的一轮对话。

        Args:
            user_message: 用户的提问。

        Returns:
            turn_id，用于后续记录操作。
        """
        self._counter += 1
        turn = ConversationTurn(
            turn_id=self._counter,
            user_message=user_message,
        )
        self._pending[self._counter] = turn
        return self._counter

    def record_thought(self, turn_id: int, thought: str) -> None:
        """记录思考过程。"""
        turn = self._pending.get(turn_id)
        if turn:
            turn.thought = thought

    def record_action(self, turn_id: int, action: str, action_input: str) -> None:
        """记录工具调用。"""
        turn = self._pending.get(turn_id)
        if turn:
            turn.action = action
            turn.action_input = action_input

    def record_observation(self, turn_id: int, observation: str) -> None:
        """记录工具返回结果。"""
        turn = self._pending.get(turn_id)
        if turn:
            turn.observation = observation

    def record_final_answer(self, turn_id: int, answer: str) -> None:
        """记录最终回答。"""
        turn = self._pending.get(turn_id)
        if turn:
            turn.final_answer = answer

    def end_turn(self, turn_id: int, success: bool = True) -> Optional[ConversationTurn]:
        """
        结束一轮对话，将其归档到历史中。

        Args:
            turn_id: 要结束的 turn_id。
            success: 该轮是否成功完成。

        Returns:
            已归档的 ConversationTurn，如果 turn_id 无效则返回 None。
        """
        turn = self._pending.pop(turn_id, None)
        if turn is None:
            return None
        turn.success = success
        turn.estimate_tokens()
        self._turns.append(turn)
        return turn

    # ---- 查询 ----

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        """返回所有已完成的轮次（按时间顺序）。"""
        return tuple(self._turns)

    @property
    def last_turn(self) -> Optional[ConversationTurn]:
        """返回最近一轮对话。"""
        return self._turns[-1] if self._turns else None

    def get_turn(self, turn_id: int) -> Optional[ConversationTurn]:
        """按 turn_id 获取指定轮次。"""
        # 先查 pending
        if turn_id in self._pending:
            return self._pending[turn_id]
        # 再查历史
        for turn in self._turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def get_recent_turns(self, n: int) -> List[ConversationTurn]:
        """获取最近 n 轮对话。"""
        turns_list = list(self._turns)
        return turns_list[-n:] if n > 0 else []

    # ---- 消息列表重构 ----

    def to_message_list(
        self,
        system_prompt: Optional[str] = None,
        keep_last_n: Optional[int] = None,
        include_observations: bool = False,
    ) -> List[Dict[str, str]]:
        """
        将对话历史转为 OpenAI 格式的消息列表。

        Args:
            system_prompt: 可选的 system prompt（放在最前面）。
            keep_last_n: 仅保留最近 n 轮（None = 全部）。
            include_observations: 是否包含 Observation 反馈消息（用于 loop_messages）。

        Returns:
            消息列表。
        """
        messages: List[Dict[str, str]] = []

        # System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 选择轮次
        turns = self.get_recent_turns(keep_last_n) if keep_last_n else list(self._turns)

        for turn in turns:
            if include_observations:
                # 完整 ReAct 追踪格式
                messages.extend(turn.to_react_trace())
            else:
                # 简单 user/assistant 对
                messages.extend(turn.to_message_pair())

        return messages

    def to_loop_messages(
        self,
        system_prompt: str,
        keep_last_n: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        构建 ReAct Agent 循环用的消息列表（含 Observation 反馈）。

        这是 agent.py 中 loop_messages 的替代品。

        Args:
            system_prompt: Agent 的系统提示词。
            keep_last_n: 仅保留最近 n 轮。

        Returns:
            消息列表，格式适配 ReAct 循环。
        """
        return self.to_message_list(
            system_prompt=system_prompt,
            keep_last_n=keep_last_n,
            include_observations=True,
        )

    # ---- 摘要与统计 ----

    def get_history_summary(self) -> str:
        """生成对话历史的文本摘要。"""
        if not self._turns:
            return "（无对话历史）"

        lines = [f"共 {len(self._turns)} 轮对话："]
        for turn in self._turns:
            lines.append(f"  {turn.to_summary()}")
        return "\n".join(lines)

    def get_stats(self) -> HistoryStats:
        """获取历史统计信息。"""
        turns_list = list(self._turns)
        completed = [t for t in turns_list if t.success]
        total_tokens = sum(t.token_count for t in turns_list)
        tools_used: Dict[str, int] = {}
        for t in turns_list:
            if t.action:
                tools_used[t.action] = tools_used.get(t.action, 0) + 1

        return HistoryStats(
            total_turns=len(turns_list),
            completed_turns=len(completed),
            failed_turns=len(turns_list) - len(completed),
            total_tokens=total_tokens,
            avg_tokens_per_turn=total_tokens / len(turns_list) if turns_list else 0.0,
            tools_used=tools_used,
        )

    # ---- 生命周期 ----

    def clear(self) -> None:
        """清空所有历史。"""
        self._turns.clear()
        self._pending.clear()
        self._counter = 0

    def reset(self) -> None:
        """同 clear()，更语义化的方法名。"""
        self.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"<HistoryManager turns={len(self._turns)} pending={len(self._pending)} max={self.max_turns}>"
