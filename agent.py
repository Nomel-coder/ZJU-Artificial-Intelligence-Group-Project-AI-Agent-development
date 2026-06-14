"""
ReAct Agent — 增强版（林放 Member 4 集成）
===========================================
基于 Member2 的原始 ReAct Agent，集成了：

- 林放的多策略解析器 (parser.py)
- 林放的对话历史管理器 (history_manager.py)
- 林放的上下文窗口管理器 (context_manager.py)
- 林放的短期记忆模块 (memory.py)
- Member3 的标准化工具模块 (tools/)

向后兼容：
  app.py 和 main.py 无需任何修改即可使用。

Author: 孙晋荣(原始框架) + 林放(解析/历史/上下文/记忆增强)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Generator, List, Optional

import openai
from dotenv import load_dotenv

import parser
from context_manager import ContextManager
from history_manager import HistoryManager
from memory import ShortTermMemory
from tools import create_default_registry
from tools.registry import ToolRegistry

# Load environment variables
load_dotenv()

# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = """你是一个具备工具调用能力的 AI 助手，名叫 ReAct Agent。你必须使用以下工具协同回答用户的问题：

1. Calculator: 用于执行数学计算。输入应该是一个合法的 Python 数学表达式，例如 "2 * 3.14 * 15"。
2. Wikipedia_Search: 用于搜索外部百科知识。输入是一个搜索关键词，例如 "人工智能"。
3. File_IO: 用于读写或追加本地文件。输入是符合 JSON 格式的操作指令，例如：
   - 读取文件：{"action": "read", "filename": "test.txt"}
   - 写入文件：{"action": "write", "filename": "test.txt", "content": "hello"}
   - 追加写入：{"action": "append", "filename": "test.txt", "content": "world"}

当你收到用户请求时，你必须严格按照以下格式进行推理和行动，不得输出任何多余的包裹文本：

Thought: 思考你当前需要做什么，以及是否需要使用工具。如果需要使用工具，思考需要使用哪个工具以及它的入参是什么。
Action: [要调用的工具名称，只能是 Calculator, Wikipedia_Search 或 File_IO 中的一个]
Action Input: [工具的输入参数。如果是 File_IO，输入必须是合法的单行 JSON 字符串；如果是其他工具，输入是纯文本参数]

（随后系统执行对应的工具并返回 Observation）
Observation: [工具执行的结果]

（继续你的推理，你可以进行多轮 Thought/Action/Action Input 循环，直到获得最终答案）
Thought: 根据上一步工具返回的结果，继续思考。如果能够直接回答，请准备输出最终答案。
Final Answer: [给用户的最终回答内容]

重要规则：
1. 每次响应必须以 Thought: 开头。
2. 每次响应只能输出一个步骤，即输出 Action Input: 或 Final Answer: 后必须立刻停止生成，等待系统返回 Observation，绝对不能自己模拟生成 Observation。
3. 保持输出的可解析性：Action 必须完全匹配预定义的工具名称，Action Input 必须紧接在 Action Input: 冒号后面。
4. 如果发生错误，系统会将错误信息作为 Observation 返回，请在接下来的 Thought 中进行反思并修正。
"""


class ReActAgent:
    """
    ReAct AI Agent — 具备工具调用、多轮推理和记忆能力。

    Usage::

        agent = ReActAgent()
        for step in agent.run("计算 3.14 * 10 的平方"):
            print(step)
        print(agent.get_conversation_summary())
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        max_iterations: int = 8,
        # ---- 林放新增参数（有默认值，不影响原有代码）----
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        max_history_turns: int = 20,
        memory_keep_verbatim: int = 3,
        workspace_dir: Optional[str] = None,
    ):
        # ---- API 配置 ----
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set in the environment variables.")

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_iterations = max_iterations

        # ---- 系统提示词 ----
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # ---- 工作目录 ----
        self.workspace_dir = workspace_dir or os.path.dirname(os.path.abspath(__file__))

        # ---- 林放模块 ----
        self.context_manager = ContextManager(max_tokens=max_tokens)
        self.history = HistoryManager(max_turns=max_history_turns)
        self.memory = ShortTermMemory(
            keep_verbatim=memory_keep_verbatim,
            context_manager=self.context_manager,
        )  # 注意：这里调用时传入 context_manager 作为关键字参数

        # ---- Member3 工具注册中心 ----
        self.tool_registry = create_default_registry(self.workspace_dir)

        # ---- 兼容原有代码 ----
        self.messages: List[Dict[str, str]] = []
        self._current_turn_id: Optional[int] = None
        self._total_iterations: int = 0

        self.reset()

    # ========================================================================
    # 公共 API
    # ========================================================================

    def reset(self) -> None:
        """重置 Agent 状态。向后兼容 Member2。"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.history.clear()
        self._current_turn_id = None
        self._total_iterations = 0

    def run(
        self,
        user_query: str,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        运行 ReAct 推理循环（生成器）。

        向后兼容 Member2 的 run() 接口。

        Yields:
            {"type": "thought"|"tool_call"|"observation"|"final_answer"|"error", ...}
        """
        if not user_query.strip():
            yield {"type": "final_answer", "content": "请输入有效的问题。"}
            return

        # ---- 林放：记录新一轮对话 ----
        self._current_turn_id = self.history.start_turn(user_query)

        # 构建消息列表（含历史压缩）
        self.messages.append({"role": "user", "content": user_query})

        iteration = 0
        loop_messages = list(self.messages)

        while iteration < self.max_iterations:
            iteration += 1
            self._total_iterations += 1

            # ---- 林放：每轮检查上下文安全 ----
            if not self.context_manager.is_safe(loop_messages):
                _before = len(loop_messages)
                loop_messages = self.context_manager.trim_to_fit(loop_messages)
                if len(loop_messages) < _before:
                    yield {
                        "type": "observation",
                        "content": f"[系统] 上下文已裁剪 ({_before} -> {len(loop_messages)} 条消息)",
                    }

            # ---- Step 1: 调用 LLM ----
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=loop_messages,
                    temperature=0.0,
                    stop=["Observation:", "observation:"],
                )
            except Exception as e:
                yield {"type": "error", "content": f"LLM API Error: {str(e)}"}
                self.history.end_turn(self._current_turn_id, success=False)
                break

            llm_output = response.choices[0].message.content.strip()

            # ---- Step 2: 使用林放的多策略解析器 ----
            parse_result = parser.parse_react_output_v2(
                llm_output,
                tool_registry=self.tool_registry,
            )

            # Yield thought
            if parse_result.thought:
                yield {"type": "thought", "content": parse_result.thought}
                self.history.record_thought(self._current_turn_id, parse_result.thought)

            # ---- 最终答案 ----
            if parse_result.type == "final_answer":
                yield {"type": "final_answer", "content": parse_result.final_answer}
                self.messages.append({"role": "assistant", "content": llm_output})
                self.history.record_final_answer(
                    self._current_turn_id, parse_result.final_answer
                )
                self.history.end_turn(self._current_turn_id, success=True)
                break

            # ---- 工具调用 ----
            if parse_result.type == "action" and parse_result.action:
                yield {
                    "type": "tool_call",
                    "tool": parse_result.action,
                    "input": parse_result.action_input,
                }

                # 执行工具（林放的路由 -> Member3 的 execute_react_tool）
                observation = parser.route_tool(
                    parse_result.action,
                    parse_result.action_input,
                    registry=self.tool_registry,
                    workspace=self.workspace_dir,
                )
                yield {"type": "observation", "content": observation}

                # 记录到历史
                self.history.record_action(
                    self._current_turn_id,
                    parse_result.action,
                    parse_result.action_input,
                )
                self.history.record_observation(self._current_turn_id, observation)

                # 更新 loop_messages
                loop_messages.append({"role": "assistant", "content": llm_output})
                loop_messages.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )
            else:
                # ---- 解析失败（error 类型或无 action）----
                error_msg = (
                    parse_result.error
                    or "Your response did not match the ReAct format. "
                       "Make sure to use 'Thought: ...' followed by "
                       "'Action: ...' and 'Action Input: ...'."
                )
                yield {"type": "error", "content": error_msg}

                loop_messages.append({"role": "assistant", "content": llm_output})
                loop_messages.append(
                    {
                        "role": "user",
                        "content": f"Observation: System Error: {error_msg}",
                    }
                )

        # ---- 达到最大迭代次数 ----
        if iteration >= self.max_iterations:
            fallback = "抱歉，由于达到了最大推理步数限制，我无法为您提供最终答案。"
            yield {"type": "final_answer", "content": fallback}
            self.history.record_final_answer(self._current_turn_id, fallback)
            self.history.end_turn(self._current_turn_id, success=False)

    # ========================================================================
    # 林放新增公共 API
    # ========================================================================

    def get_conversation_summary(self) -> str:
        """获取对话历史摘要。"""
        return self.history.get_history_summary()

    def get_history_stats(self) -> Dict[str, Any]:
        """获取历史统计信息。"""
        stats = self.history.get_stats()
        return {
            "total_turns": stats.total_turns,
            "completed_turns": stats.completed_turns,
            "failed_turns": stats.failed_turns,
            "total_tokens": stats.total_tokens,
            "avg_tokens_per_turn": f"{stats.avg_tokens_per_turn:.1f}",
            "tools_used": stats.tools_used,
        }

    def get_context_usage(self) -> float:
        """获取当前上下文使用率 (0.0 ~ 1.0+)。"""
        return self.context_manager.usage_ratio(self.messages)

    @property
    def total_iterations(self) -> int:
        """获取累计迭代次数。"""
        return self._total_iterations

    @property
    def total_turns(self) -> int:
        """获取历史轮次数。"""
        return len(self.history)
