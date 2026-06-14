"""
ReAct Output Parser — 3 策略级联解析器 + Tool Router
======================================================
林放 (Member 4) 核心交付物。

解析 LLM 输出的 Thought / Action / Action Input / Final Answer，
支持多种格式变体，逐级降级保证鲁棒性。

策略级联：
  策略 1: 结构化标签解析  (<thought>, <action>, <final_answer>)
  策略 2: 增强正则解析    (中英文标点、大小写不敏感、多行支持)
  策略 3: 模糊启发式回退  (关键词匹配 + 位置推断)

同时提供：
  - route_tool(): 委托 Member3 的 ToolRegistry 执行工具
  - format_observation(): 标准化 Observation 文本格式

向后兼容：
  - parse_react_response(text) → dict (Member2 原始接口)
  - route_tool(action, action_input) → str (Member2 原始接口)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from context_manager import estimate_tokens
from tools.react_compat import execute_react_tool
from tools.registry import ToolRegistry


# ============================================================================
# ParseResult — 结构化解析结果
# ============================================================================

ParseType = Literal["thought", "action", "final_answer", "error"]


@dataclass
class ParseResult:
    """LLM 输出解析结果。"""
    type: ParseType
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[str] = None
    final_answer: Optional[str] = None
    confidence: float = 0.0
    raw_text: str = ""
    error: Optional[str] = None
    strategy: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """转为旧版兼容 dict（Member2 的 parse_react_response 返回格式）。"""
        return {
            "thought": self.thought or "",
            "action": self.action,
            "action_input": self.action_input,
            "final_answer": self.final_answer,
        }


# ============================================================================
# 已知工具名（用于模糊匹配验证）
# ============================================================================

_KNOWN_TOOL_NAMES = {
    "calculator", "calculate", "math",
    "wikipedia_search", "wikipedia", "wikipediasearch", "wikisearch",
    "file_io", "file", "fileio", "localfile",
    "read_file", "write_file", "list_files",
}


def get_known_tool_names(registry: Optional[ToolRegistry] = None) -> set[str]:
    """获取已知工具名集合（合并 registry 中的和内置的）。"""
    if registry is not None:
        specs = registry.get_specs()
        names = {s["name"].lower().replace("_", "").replace(" ", "") for s in specs}
        return names | _KNOWN_TOOL_NAMES
    return _KNOWN_TOOL_NAMES


# ============================================================================
# 正则关键词（中英文双语）
# ============================================================================

_THOUGHT_KEY = r"(?:Thought|思考|想法|thought|THOUGHT)"
_ACTION_KEY = r"(?:Action|动作|行动|action|ACTION)"
_ACTION_INPUT_KEY = r"(?:Action\s*Input|动作输入|参数|action\s*input|ACTION\s*INPUT)"
_FINAL_KEY = r"(?:Final\s*Answer|最终答案|最终回答|答案|final\s*answer|FINAL\s*ANSWER)"


# ============================================================================
# 策略 1: 结构化标签解析
# ============================================================================

_TAG_PATTERNS = {
    "thought": re.compile(
        r"<thought>\s*(.*?)\s*</thought>", re.DOTALL | re.IGNORECASE
    ),
    "action": re.compile(
        r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE
    ),
    "action_input": re.compile(
        r"<action_input>\s*(.*?)\s*</action_input>", re.DOTALL | re.IGNORECASE
    ),
    "final_answer": re.compile(
        r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL | re.IGNORECASE
    ),
}


def _parse_with_tags(text: str) -> Optional[ParseResult]:
    """策略 1: XML 风格标签解析。"""
    thought_match = _TAG_PATTERNS["thought"].search(text)
    action_match = _TAG_PATTERNS["action"].search(text)
    action_input_match = _TAG_PATTERNS["action_input"].search(text)
    final_match = _TAG_PATTERNS["final_answer"].search(text)

    thought = thought_match.group(1).strip() if thought_match else None
    action = action_match.group(1).strip() if action_match else None
    action_input = action_input_match.group(1).strip() if action_input_match else None
    final_answer = final_match.group(1).strip() if final_match else None

    if thought is None and action is None and final_answer is None:
        return None
    if action is not None and action_input is None:
        return None

    if final_answer:
        return ParseResult(
            type="final_answer", thought=thought, final_answer=final_answer,
            confidence=0.95, raw_text=text, strategy="structured_tags",
        )
    elif action:
        return ParseResult(
            type="action", thought=thought, action=action, action_input=action_input,
            confidence=0.95, raw_text=text, strategy="structured_tags",
        )
    elif thought:
        return ParseResult(
            type="thought", thought=thought,
            confidence=0.90, raw_text=text, strategy="structured_tags",
        )
    return None


# ============================================================================
# 策略 2: 增强正则解析
# ============================================================================


def _parse_with_regex(text: str) -> Optional[ParseResult]:
    """策略 2: 增强正则解析。支持中英文、大小写不敏感、多行 Action Input。"""
    # ---- Final Answer ----
    final_pattern = re.compile(
        rf"{_FINAL_KEY}\s*[：:]\s*(.*?)$", re.DOTALL | re.IGNORECASE
    )
    final_match = final_pattern.search(text)
    if final_match:
        thought = _extract_field(text, _THOUGHT_KEY)
        return ParseResult(
            type="final_answer", thought=thought,
            final_answer=final_match.group(1).strip(),
            confidence=0.85, raw_text=text, strategy="regex",
        )

    # ---- Action + Action Input ----
    action = _extract_field(text, _ACTION_KEY)
    action_input = _extract_field_multiline(text, _ACTION_INPUT_KEY)

    if action and action_input:
        thought = _extract_field(text, _THOUGHT_KEY)
        return ParseResult(
            type="action", thought=thought, action=action, action_input=action_input,
            confidence=0.85, raw_text=text, strategy="regex",
        )

    # ---- 仅有 Thought ----
    thought = _extract_field(text, _THOUGHT_KEY)
    if thought:
        return ParseResult(
            type="thought", thought=thought,
            confidence=0.75, raw_text=text, strategy="regex",
        )

    return None


def _extract_field(text: str, key_pattern: str) -> Optional[str]:
    """提取 Key: value 格式的字段（单行值）。"""
    pattern = re.compile(
        rf"{key_pattern}\s*[：:]\s*(.*?)(?=(?:{_THOUGHT_KEY}|{_ACTION_KEY}|{_ACTION_INPUT_KEY}|{_FINAL_KEY})\s*[：:]|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        value = match.group(1).strip()
        value = re.sub(r"\s*[-=]{3,}\s*$", "", value).strip()
        return value or None
    return None


def _extract_field_multiline(text: str, key_pattern: str) -> Optional[str]:
    """提取可能多行的字段值，并去掉 Markdown 代码块。"""
    pattern = re.compile(
        rf"{key_pattern}\s*[：:]\s*(.*?)(?=(?:{_THOUGHT_KEY}|{_ACTION_KEY}|{_FINAL_KEY})\s*[：:]|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None

    value = match.group(1).strip()
    value = _strip_markdown_code_block(value)
    value = re.sub(r"\s*[-=]{3,}\s*$", "", value).strip()
    return value or None


def _strip_markdown_code_block(text: str) -> str:
    """去除 ```...``` 代码块包裹。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


# ============================================================================
# 策略 3: 模糊启发式回退
# ============================================================================


def _parse_fuzzy(text: str) -> ParseResult:
    """策略 3: 关键词匹配 + 位置推断。"""

    # 尝试找已知工具名
    action = None
    action_input = None
    lower_text = text.lower()
    for tool_name in sorted(_KNOWN_TOOL_NAMES, key=len, reverse=True):
        if tool_name in lower_text:
            action = tool_name
            idx = lower_text.index(tool_name)
            after = text[idx + len(tool_name):].strip()
            if after.startswith(":") or after.startswith("："):
                after = after[1:].strip()
            if after:
                action_input = after.split("\n")[0].strip()
            break

    # 检查"最终答案"等关键词
    final_answer = None
    final_indicators = [
        r"(?:最终答案|答案|结果是|所以|因此|综上|Final\s*Answer)\s*[：:]\s*(.+)",
        r"(?:回答|回复)\s*[：:]\s*(.+)",
    ]
    for pattern in final_indicators:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            final_answer = m.group(1).strip()
            break

    # 兜底：把末尾当最终回答
    if not action and not final_answer:
        split_point = max(1, len(text) * 7 // 10)
        final_answer = text[split_point:].strip()

    if final_answer:
        return ParseResult(
            type="final_answer", final_answer=final_answer,
            confidence=0.40, raw_text=text, strategy="fuzzy",
        )
    elif action:
        return ParseResult(
            type="action", action=action, action_input=action_input,
            confidence=0.35, raw_text=text, strategy="fuzzy",
        )
    else:
        return ParseResult(
            type="error", error="无法解析 LLM 输出：所有策略均失败",
            confidence=0.0, raw_text=text, strategy="fuzzy",
        )


# ============================================================================
# 主解析入口
# ============================================================================


def parse_react_output_v2(
    llm_output: str,
    tool_registry: Optional[ToolRegistry] = None,
) -> ParseResult:
    """
    3 策略级联解析 LLM 的 ReAct 输出。

    优先级: 结构化标签 → 增强正则 → 模糊回退

    Args:
        llm_output: LLM 原始输出文本。
        tool_registry: 可选的 ToolRegistry（用于验证工具名）。

    Returns:
        ParseResult，含类型、置信度和所有提取的字段。
    """
    if not llm_output or not llm_output.strip():
        return ParseResult(
            type="error", error="LLM 输出为空",
            confidence=0.0, raw_text="", strategy="none",
        )

    text = llm_output.strip()
    known_names = get_known_tool_names(tool_registry)

    # 策略 1: 结构化标签
    result = _parse_with_tags(text)
    if result is not None:
        return _post_validate(result, known_names)

    # 策略 2: 增强正则
    result = _parse_with_regex(text)
    if result is not None:
        return _post_validate(result, known_names)

    # 策略 3: 模糊回退
    return _parse_fuzzy(text)


def _post_validate(result: ParseResult, known_names: set[str]) -> ParseResult:
    """验证和修正解析结果。"""
    if result.type == "action" and result.action:
        normalized = result.action.strip().lower().replace("_", "").replace(" ", "")
        if normalized not in known_names:
            result.confidence -= 0.15
            result.error = (
                f"警告：工具名 '{result.action}' 不在已知列表中 "
                f"({', '.join(sorted(known_names))})"
            )
    return result


# ============================================================================
# 向后兼容接口
# ============================================================================


def parse_react_response(text: str) -> Dict[str, Any]:
    """
    向后兼容 Member2 的 parse_react_response()。

    Returns:
        {"thought": str, "action": str|None, "action_input": str|None, "final_answer": str|None}
    """
    return parse_react_output_v2(text).to_dict()


# ============================================================================
# Observation 格式化
# ============================================================================


def format_observation(
    result_text: str,
    tool_name: Optional[str] = None,
    max_length: int = 1200,
) -> str:
    """
    标准化 Observation 格式。

    Args:
        result_text: 工具执行结果。
        tool_name: 工具名（可选）。
        max_length: 最大长度，超出截断。

    Returns:
        格式化的 Observation 文本。
    """
    is_error = "error" in result_text.lower()[:50] or "失败" in result_text[:50]
    prefix = f"Observation ({tool_name}): " if tool_name and is_error else \
              f"Observation ({tool_name} returned): " if tool_name else \
              "Observation: "

    if len(result_text) > max_length:
        result_text = result_text[:max_length - 30] + "\n...[结果过长，已截断]"

    return prefix + result_text


# ============================================================================
# Tool Router（委托 Member3 的 ToolRegistry）
# ============================================================================

_default_registry: Optional[ToolRegistry] = None
_default_workspace: Optional[str] = None


def _get_registry(workspace: Optional[str] = None) -> ToolRegistry:
    """获取默认 ToolRegistry（延迟初始化）。"""
    global _default_registry, _default_workspace
    effective_workspace = workspace or _default_workspace or "."
    if _default_registry is None or workspace != _default_workspace:
        from tools import create_default_registry
        _default_registry = create_default_registry(effective_workspace)
        _default_workspace = workspace
    return _default_registry


def route_tool(
    action: str,
    action_input: str,
    registry: Optional[ToolRegistry] = None,
    workspace: Optional[str] = None,
) -> str:
    """
    工具路由 — 委托 Member3 的 execute_react_tool() 执行。

    向后兼容 Member2 的 route_tool(action, action_input) 签名。

    Args:
        action: 工具名（"Calculator", "Wikipedia_Search", "File_IO"）。
        action_input: 工具参数（纯文本或 JSON 字符串）。
        registry: 可选的 ToolRegistry（默认自动创建）。
        workspace: 可选的 Agent 工作目录。

    Returns:
        标准化 Observation 文本。
    """
    effective_registry = registry or _get_registry(workspace)

    observation_text = execute_react_tool(effective_registry, action, action_input)

    return format_observation(observation_text, action)


# ============================================================================
# 验证工具名
# ============================================================================


def validate_tool_name(
    action: str,
    registry: Optional[ToolRegistry] = None,
) -> bool:
    """验证工具名有效性。"""
    known = get_known_tool_names(registry)
    normalized = action.strip().lower().replace("_", "").replace(" ", "")
    return normalized in known
