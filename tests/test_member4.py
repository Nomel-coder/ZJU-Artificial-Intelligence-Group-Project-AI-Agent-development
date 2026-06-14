"""
Member 4 (林放) 模块测试
========================
覆盖：解析器、路由、上下文管理、历史管理、记忆模块、Agent 集成

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---- 添加父目录到 sys.path ----
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_manager import (
    ContextManager,
    ContextStats,
    estimate_tokens,
    estimate_message_tokens,
    estimate_message_list_tokens,
)
from history_manager import ConversationTurn, HistoryManager, HistoryStats
from memory import ShortTermMemory
from parser import (
    ParseResult,
    format_observation,
    parse_react_output_v2,
    parse_react_response,
    route_tool,
    validate_tool_name,
)


# ============================================================================
# Parser Tests (15 tests)
# ============================================================================


class ParserStandardFormatTests(unittest.TestCase):
    """解析器 - 标准格式（策略2：增强正则）。"""

    def test_parse_standard_thought_action(self):
        """标准 Thought + Action + Action Input 格式。"""
        text = "Thought: I need to calculate\nAction: Calculator\nAction Input: 2 + 3"
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "action")
        self.assertEqual(result.action, "Calculator")
        self.assertEqual(result.action_input, "2 + 3")
        self.assertIn("I need to calculate", result.thought)
        self.assertGreater(result.confidence, 0.8)

    def test_parse_final_answer(self):
        """标准 Final Answer 格式。"""
        text = "Thought: The answer is 42\nFinal Answer: 42"
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "final_answer")
        self.assertEqual(result.final_answer, "42")
        self.assertGreater(result.confidence, 0.8)

    def test_parse_no_thought(self):
        """缺少 Thought（合法情况）。"""
        text = "Action: Calculator\nAction Input: 1+1"
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "action")
        self.assertEqual(result.action, "Calculator")

    def test_parse_thought_only(self):
        """仅有 Thought。"""
        text = "Thought: I am thinking about what to do"
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "thought")


class ParserChineseTests(unittest.TestCase):
    """解析器 - 中文标点支持。"""

    def test_chinese_thought(self):
        """中文 '思考：' 关键词。"""
        text = "思考：我需要计算数学表达式\n动作：Calculator\n动作输入：3+5"
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "action")
        self.assertEqual(result.action, "Calculator")
        self.assertEqual(result.action_input, "3+5")

    def test_chinese_final_answer(self):
        """中文 '最终答案：' 关键词。"""
        text = "思考：已完成计算\n最终答案：结果是 8"
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "final_answer")
        self.assertEqual(result.final_answer, "结果是 8")

    def test_mixed_chinese_english(self):
        """中英文混合 Thought。"""
        text = "Thought: Use Wikipedia 搜索\nAction: Wikipedia_Search\nAction Input: 人工智能发展历史"
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "action")
        self.assertEqual(result.action, "Wikipedia_Search")


class ParserEdgeCaseTests(unittest.TestCase):
    """解析器 - 边界情况。"""

    def test_case_insensitive(self):
        """大小写不敏感。"""
        text = "thought: test\nACTION: Calculator\naction input: 9 * 9"
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "action")

    def test_empty_output(self):
        """空输入。"""
        result = parse_react_output_v2("")
        self.assertEqual(result.type, "error")
        self.assertEqual(result.strategy, "none")

    def test_multiline_action_input(self):
        """多行 Action Input。"""
        text = (
            "Thought: Write file\n"
            "Action: File_IO\n"
            "Action Input: {\n"
            '  "action": "write",\n'
            '  "filename": "test.txt",\n'
            '  "content": "hello world"\n'
            "}"
        )
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "action")
        self.assertIn("write", result.action_input)

    def test_markdown_code_block_in_input(self):
        """Markdown 代码块包裹的 Action Input。"""
        text = (
            "Thought: File operation\n"
            "Action: File_IO\n"
            "Action Input: ```json\n"
            '{"action": "read", "filename": "note.txt"}\n'
            "```"
        )
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "action")
        self.assertIn("read", result.action_input)
        self.assertNotIn("```", result.action_input)

    def test_unknown_tool_lower_confidence(self):
        """未知工具名降低置信度但不报错。"""
        text = "Thought: do it\nAction: UnknownTool\nAction Input: test"
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "action")
        self.assertLess(result.confidence, 0.75)
        self.assertIsNotNone(result.error)


class ParserStructuredTagTests(unittest.TestCase):
    """解析器 - 策略1: 结构化标签。"""

    def test_xml_style_tags(self):
        """XML 风格标签解析。"""
        text = (
            "<thought>Calculate something</thought>\n"
            "<action>Calculator</action>\n"
            "<action_input>100 / 4</action_input>"
        )
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "action")
        self.assertEqual(result.strategy, "structured_tags")
        self.assertAlmostEqual(result.confidence, 0.95, delta=0.01)

    def test_final_answer_tags(self):
        """Final Answer 标签。"""
        text = (
            "<thought>Done</thought>\n"
            "<final_answer>The result is 25</final_answer>"
        )
        result = parse_react_output_v2(text)
        self.assertEqual(result.type, "final_answer")
        self.assertEqual(result.strategy, "structured_tags")


class ParserBackwardCompatTests(unittest.TestCase):
    """解析器 - 向后兼容。"""

    def test_old_interface_returns_dict(self):
        """parse_react_response 返回旧 dict 格式。"""
        text = "Thought: test\nAction: Calculator\nAction Input: 1+1"
        d = parse_react_response(text)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["thought"], "test")
        self.assertEqual(d["action"], "Calculator")
        self.assertEqual(d["action_input"], "1+1")


# ============================================================================
# Tool Router Tests (5 tests)
# ============================================================================


class ToolRouterTests(unittest.TestCase):
    """路由测试。"""

    def test_route_calculator(self):
        """路由到计算器。"""
        obs = route_tool("Calculator", "6 * 7")
        self.assertIn("42", obs)

    def test_route_file_io(self):
        """路由到文件读写。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = route_tool(
                "File_IO",
                json.dumps({"action": "write", "filename": "hello.txt", "content": "world"}),
                workspace=tmpdir,
            )
            self.assertIn("bytes_written", obs.lower())

    def test_route_unknown_tool(self):
        """路由到不存在的工具。"""
        obs = route_tool("NonExistent", "test")
        self.assertIn("ToolNotFoundError", obs)

    def test_route_invalid_input(self):
        """路由到无效的计算表达式。"""
        obs = route_tool("Calculator", "1 / 0")
        self.assertIn("false", obs.lower())

    def test_format_observation(self):
        """Observation 格式化。"""
        formatted = format_observation("42", tool_name="Calculator")
        self.assertIn("Calculator", formatted)
        self.assertIn("42", formatted)


# ============================================================================
# Context Manager Tests (5 tests)
# ============================================================================


class ContextManagerTests(unittest.TestCase):
    """上下文管理器测试。"""

    def setUp(self):
        self.ctx = ContextManager(max_tokens=4096, reserved_tokens=500)

    def test_token_estimation_english(self):
        """英文 token 估算。"""
        tokens = estimate_tokens("hello world")
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, 20)  # 应该很小

    def test_token_estimation_chinese(self):
        """中文 token 估算。"""
        tokens = estimate_tokens("你好世界")
        self.assertGreater(tokens, 0)
        # 4个中文 * 1.5 = 6
        self.assertAlmostEqual(tokens, 6, delta=3)

    def test_trim_keeps_system_prompt(self):
        """裁剪保留 system prompt。"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "x" * 10000},  # 制造超大消息
            {"role": "user", "content": "Current"},
        ]
        trimmed = self.ctx.trim_to_fit(messages)
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertEqual(trimmed[-1]["role"], "user")

    def test_is_safe(self):
        """安全检查。"""
        messages = [{"role": "system", "content": "Hello"}]
        self.assertTrue(self.ctx.is_safe(messages))

    def test_usage_ratio(self):
        """使用率计算。"""
        messages = [{"role": "system", "content": "Hello"}]
        ratio = self.ctx.usage_ratio(messages)
        self.assertLess(ratio, 0.1)


# ============================================================================
# History Manager Tests (5 tests)
# ============================================================================


class HistoryManagerTests(unittest.TestCase):
    """历史管理器测试。"""

    def setUp(self):
        self.hist = HistoryManager(max_turns=10)

    def test_add_and_end_turn(self):
        """添加并完成一轮对话。"""
        tid = self.hist.start_turn("Calculate 1+1")
        self.hist.record_thought(tid, "Use calculator")
        self.hist.record_action(tid, "Calculator", "1+1")
        self.hist.record_observation(tid, "2")
        self.hist.record_final_answer(tid, "The answer is 2")
        turn = self.hist.end_turn(tid, success=True)

        self.assertIsNotNone(turn)
        self.assertEqual(len(self.hist), 1)
        self.assertTrue(turn.success)
        self.assertEqual(turn.user_message, "Calculate 1+1")

    def test_max_turns_enforced(self):
        """上限控制正常。"""
        hist = HistoryManager(max_turns=3)
        for i in range(5):
            tid = hist.start_turn(f"Q{i}")
            hist.record_final_answer(tid, f"A{i}")
            hist.end_turn(tid)
        self.assertEqual(len(hist), 3)

    def test_to_message_list_format(self):
        """消息列表格式正确。"""
        tid = self.hist.start_turn("Hello")
        self.hist.record_thought(tid, "Greet")
        self.hist.record_final_answer(tid, "Hi there!")
        self.hist.end_turn(tid, success=True)

        msgs = self.hist.to_message_list(system_prompt="You are helpful.")
        self.assertGreater(len(msgs), 0)
        self.assertEqual(msgs[0]["role"], "system")

    def test_get_history_summary(self):
        """历史摘要生成。"""
        tid = self.hist.start_turn("Q1")
        self.hist.record_final_answer(tid, "A1")
        self.hist.end_turn(tid)

        summary = self.hist.get_history_summary()
        self.assertIn("Q1", summary)
        self.assertIn("A1", summary)

    def test_get_stats(self):
        """统计信息正确。"""
        tid = self.hist.start_turn("Calc")
        self.hist.record_action(tid, "Calculator", "2+2")
        self.hist.record_final_answer(tid, "4")
        self.hist.end_turn(tid, success=True)

        stats = self.hist.get_stats()
        self.assertEqual(stats.total_turns, 1)
        self.assertEqual(stats.completed_turns, 1)
        self.assertIn("Calculator", stats.tools_used)


# ============================================================================
# Memory Tests (3 tests)
# ============================================================================


class MemoryTests(unittest.TestCase):
    """记忆模块测试。"""

    def setUp(self):
        self.ctx = ContextManager(max_tokens=4096, reserved_tokens=500)
        self.memory = ShortTermMemory(keep_verbatim=2, context_manager=self.ctx)

    def test_build_context_within_verbatim(self):
        """轮次未超过 keep_verbatim 时不压缩。"""
        turns = [
            ConversationTurn(turn_id=1, user_message="Q1", final_answer="A1"),
            ConversationTurn(turn_id=2, user_message="Q2", final_answer="A2"),
        ]
        msgs = self.memory.build_context(turns, "Current Q", system_prompt="System")
        # 应该有 system + Q1/A1 + Q2/A2 + Current Q
        self.assertGreater(len(msgs), 3)

    def test_compress_old_turns(self):
        """超过 keep_verbatim 时压缩旧轮。"""
        turns = []
        for i in range(5):
            turns.append(
                ConversationTurn(
                    turn_id=i + 1,
                    user_message=f"Question {i+1}",
                    final_answer=f"Answer {i+1}",
                )
            )
        msgs = self.memory.build_context(turns, "Current Q", system_prompt="System")
        # 应该包含 [历史摘要] 之类的 system 消息
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[0], "system")  # system prompt

    def test_should_compress_detection(self):
        """压缩判断逻辑。"""
        turns = [ConversationTurn(turn_id=i, user_message="Q") for i in range(5)]
        self.assertTrue(self.memory.should_compress(turns))

        few_turns = [ConversationTurn(turn_id=1, user_message="Q")]
        self.assertFalse(self.memory.should_compress(few_turns))


# ============================================================================
# Agent Integration Tests (3 tests)
# ============================================================================


class AgentIntegrationTests(unittest.TestCase):
    """Agent 集成测试。"""

    @classmethod
    def setUpClass(cls):
        """跳过需要 API key 的测试如果 key 不可用。"""
        from dotenv import load_dotenv
        load_dotenv()
        cls.has_api_key = bool(os.getenv("DEEPSEEK_API_KEY"))

    @unittest.skipIf(
        not os.getenv("DEEPSEEK_API_KEY"),
        "No DEEPSEEK_API_KEY found — skipping integration test",
    )
    def test_agent_constructor_module_integration(self):
        """Agent 构造时正确初始化所有模块。"""
        from agent import ReActAgent

        agent = ReActAgent()

        self.assertIsNotNone(agent.tool_registry)
        self.assertIsNotNone(agent.context_manager)
        self.assertIsNotNone(agent.history)
        self.assertIsNotNone(agent.memory)

        # 验证工具注册中心可用
        specs = agent.tool_registry.get_specs()
        self.assertGreaterEqual(len(specs), 5)

    @unittest.skipIf(
        not os.getenv("DEEPSEEK_API_KEY"),
        "No DEEPSEEK_API_KEY found — skipping integration test",
    )
    def test_agent_reset_compatibility(self):
        """reset() 恢复初始状态。"""
        from agent import ReActAgent

        agent = ReActAgent()
        agent.reset()

        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertEqual(agent.total_iterations, 0)
        self.assertEqual(agent.total_turns, 0)

    @unittest.skipIf(
        not os.getenv("DEEPSEEK_API_KEY"),
        "No DEEPSEEK_API_KEY found — skipping API test",
    )
    def test_agent_run_simple_calculator(self):
        """完整 ReAct 循环 - 简单计算。"""
        from agent import ReActAgent

        agent = ReActAgent(max_iterations=3)
        steps = list(agent.run("What is 2 + 3?"))

        # 应该有步骤输出
        self.assertGreater(len(steps), 0)
        types = [s["type"] for s in steps]
        self.assertIn("final_answer", types)


# ============================================================================
# Tool Registry Compatibility Tests (2 tests)
# ============================================================================


class ToolRegistryCompatTests(unittest.TestCase):
    """验证 Member3 工具仍在工作。"""

    def test_member3_tests_still_pass(self):
        """Member3 的原始测试不受影响。"""
        # 这是因为 tools/ 目录未被修改
        from tools import create_default_registry
        registry = create_default_registry(tempfile.mkdtemp())
        specs = registry.get_specs()
        names = [s["name"] for s in specs]
        self.assertIn("calculator", names)
        self.assertIn("wikipedia_search", names)

    def test_member2_compat_functions_exist(self):
        """Member2 兼容函数仍然可用。"""
        from tools.react_compat import calculate, execute_react_tool

        result = calculate("3 * 4")
        self.assertIn("12", result)


if __name__ == "__main__":
    unittest.main()
