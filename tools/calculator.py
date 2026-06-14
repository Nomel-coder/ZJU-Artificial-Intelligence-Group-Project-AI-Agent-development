"""A calculator that evaluates arithmetic without using eval()."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from .base import BaseTool, ToolSpec
from .exceptions import ToolInputError


BinaryOperator = Callable[[int | float, int | float], int | float]


class CalculatorTool(BaseTool):
    spec = ToolSpec(
        name="calculator",
        description="计算一个仅包含数字、括号和常见算术运算符的表达式。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "例如：(12 + 8) * 3 / 5",
                }
            },
            "required": ["expression"],
        },
    )

    _binary_operators: dict[type[ast.operator], BinaryOperator] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_operators: dict[type[ast.unaryop], Callable[[Any], Any]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(
        self,
        *,
        max_expression_length: int = 500,
        max_nodes: int = 100,
        max_exponent: int = 1000,
        max_result_bits: int = 10_000,
    ) -> None:
        self.max_expression_length = max_expression_length
        self.max_nodes = max_nodes
        self.max_exponent = max_exponent
        self.max_result_bits = max_result_bits

    def _run(self, **kwargs: Any) -> int | float:
        expression = kwargs.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ToolInputError("expression 必须是非空字符串")
        if len(expression) > self.max_expression_length:
            raise ToolInputError(
                f"表达式过长，最多允许 {self.max_expression_length} 个字符"
            )

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolInputError("表达式语法错误") from exc

        if sum(1 for _ in ast.walk(tree)) > self.max_nodes:
            raise ToolInputError("表达式过于复杂")

        try:
            result = self._evaluate(tree.body)
        except ZeroDivisionError as exc:
            raise ToolInputError("除数不能为零") from exc
        except OverflowError as exc:
            raise ToolInputError("计算结果超出允许范围") from exc

        if isinstance(result, float) and not math.isfinite(result):
            raise ToolInputError("计算结果不是有限数值")
        return result

    def _evaluate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolInputError("表达式只能包含数字")
            return self._validate_result(node.value)

        if isinstance(node, ast.UnaryOp):
            operation = self._unary_operators.get(type(node.op))
            if operation is None:
                raise ToolInputError("表达式包含不支持的一元运算符")
            return self._validate_result(operation(self._evaluate(node.operand)))

        if isinstance(node, ast.BinOp):
            operation = self._binary_operators.get(type(node.op))
            if operation is None:
                raise ToolInputError("表达式包含不支持的运算符")
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > self.max_exponent:
                raise ToolInputError(f"指数绝对值不能超过 {self.max_exponent}")
            if (
                isinstance(node.op, ast.Pow)
                and right > 0
                and abs(left) > 1
                and right * math.log2(abs(left)) > self.max_result_bits
            ):
                raise ToolInputError("计算结果过大")
            return self._validate_result(operation(left, right))

        raise ToolInputError("表达式包含函数、变量或其他不安全内容")

    def _validate_result(self, value: Any) -> int | float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolInputError("计算结果必须是实数")
        if isinstance(value, int) and value.bit_length() > self.max_result_bits:
            raise ToolInputError("计算结果过大")
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolInputError("计算结果不是有限数值")
        return value
