from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from tools import (
    CalculatorTool,
    ListFilesTool,
    ReadFileTool,
    ToolRegistry,
    WikipediaSearchTool,
    WriteFileTool,
    calculate,
    create_default_registry,
    execute_react_tool,
    file_io,
)


class CalculatorToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CalculatorTool()

    def test_calculates_arithmetic_expression(self) -> None:
        result = self.tool.execute(expression="(12 + 8) * 3 / 5")

        self.assertTrue(result.success)
        self.assertEqual(result.output, 12)

    def test_rejects_code_execution(self) -> None:
        result = self.tool.execute(expression="__import__('os').getcwd()")

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ToolInputError")

    def test_reports_division_by_zero(self) -> None:
        result = self.tool.execute(expression="1 / 0")

        self.assertFalse(result.success)
        self.assertIn("除数不能为零", result.error or "")
        self.assertIn('"success": false', result.to_observation())

    def test_limits_exponent(self) -> None:
        result = self.tool.execute(expression="2 ** 1001")

        self.assertFalse(result.success)
        self.assertIn("指数", result.error or "")

    def test_limits_result_size(self) -> None:
        result = self.tool.execute(expression="9999999999 ** 1000")

        self.assertFalse(result.success)
        self.assertIn("结果过大", result.error or "")

    def test_rejects_complex_result(self) -> None:
        result = self.tool.execute(expression="(-1) ** 0.5")

        self.assertFalse(result.success)
        self.assertIn("实数", result.error or "")


class FileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.reader = ReadFileTool(self.root)
        self.writer = WriteFileTool(self.root)
        self.lister = ListFilesTool(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_write_read_append_and_list(self) -> None:
        first_write = self.writer.execute(path="notes/demo.txt", content="hello")
        second_write = self.writer.execute(
            path="notes/demo.txt", content=" world", append=True
        )
        read = self.reader.execute(path="notes/demo.txt")
        listing = self.lister.execute(path="notes")

        self.assertTrue(first_write.success)
        self.assertTrue(second_write.success)
        self.assertEqual(read.output["content"], "hello world")
        self.assertEqual(
            listing.output["entries"],
            [{"name": "demo.txt", "type": "file"}],
        )

    def test_blocks_parent_directory_escape(self) -> None:
        result = self.writer.execute(path="../outside.txt", content="blocked")

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ToolInputError")
        self.assertFalse((self.root.parent / "outside.txt").exists())

    def test_blocks_absolute_path(self) -> None:
        result = self.writer.execute(path=str(self.root / "absolute.txt"), content="x")

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ToolInputError")

    def test_reports_missing_file(self) -> None:
        result = self.reader.execute(path="missing.txt")

        self.assertFalse(result.success)
        self.assertIn("文件不存在", result.error or "")


class FakeDisambiguationError(Exception):
    def __init__(self, options: list[str]) -> None:
        super().__init__("ambiguous")
        self.options = options


class FakePageError(Exception):
    pass


class FakeWikipedia:
    exceptions = SimpleNamespace(
        DisambiguationError=FakeDisambiguationError,
        PageError=FakePageError,
    )

    def __init__(self) -> None:
        self.language = ""

    def set_lang(self, language: str) -> None:
        self.language = language

    @staticmethod
    def page(query: str, auto_suggest: bool) -> SimpleNamespace:
        return SimpleNamespace(title=query.title(), url="https://example.test/page")

    @staticmethod
    def summary(title: str, sentences: int, auto_suggest: bool) -> str:
        return f"{title}: {sentences} sentences"


class WikipediaSearchToolTests(unittest.TestCase):
    def test_returns_structured_summary(self) -> None:
        fake = FakeWikipedia()
        tool = WikipediaSearchTool(wikipedia_module=fake)

        result = tool.execute(query="artificial intelligence", language="en", sentences=2)

        self.assertTrue(result.success)
        self.assertEqual(fake.language, "en")
        self.assertEqual(result.output["title"], "Artificial Intelligence")
        self.assertEqual(result.output["language"], "en")
        self.assertIn('"title": "Artificial Intelligence"', result.to_observation())

    def test_rejects_invalid_sentence_count(self) -> None:
        tool = WikipediaSearchTool(wikipedia_module=FakeWikipedia())

        result = tool.execute(query="AI", sentences=0)

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ToolInputError")

    def test_auto_selects_chinese_language(self) -> None:
        fake = FakeWikipedia()
        tool = WikipediaSearchTool(wikipedia_module=fake)

        result = tool.execute(query="人工智能")

        self.assertTrue(result.success)
        self.assertEqual(fake.language, "zh")


class ReactCompatibilityTests(unittest.TestCase):
    def test_supports_member2_calculator_name(self) -> None:
        registry = ToolRegistry([CalculatorTool()])

        observation = execute_react_tool(registry, "Calculator", "6 * 7")

        self.assertEqual(observation, "42")

    def test_supports_member2_wikipedia_name(self) -> None:
        registry = ToolRegistry(
            [WikipediaSearchTool(wikipedia_module=FakeWikipedia())]
        )

        observation = execute_react_tool(
            registry,
            "Wikipedia_Search",
            "artificial intelligence",
        )

        self.assertIn('"title": "Artificial Intelligence"', observation)

    def test_supports_member2_file_io_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ToolRegistry(
                [
                    ReadFileTool(temp_dir),
                    WriteFileTool(temp_dir),
                ]
            )

            write_observation = execute_react_tool(
                registry,
                "File_IO",
                '{"action": "write", "filename": "note.txt", "content": "hello"}',
            )
            read_observation = execute_react_tool(
                registry,
                "File_IO",
                '{"action": "read", "filename": "note.txt"}',
            )

        self.assertIn('"mode": "overwrite"', write_observation)
        self.assertIn('"content": "hello"', read_observation)

    def test_reports_invalid_file_io_json(self) -> None:
        observation = execute_react_tool(
            ToolRegistry(),
            "File_IO",
            "not json",
        )

        self.assertIn('"success": false', observation)
        self.assertIn("合法的单行 JSON", observation)

    def test_exposes_member2_calculate_function(self) -> None:
        self.assertEqual(calculate("8 * 8"), "64")

    def test_exposes_member2_file_io_function(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with unittest.mock.patch.dict(
                "os.environ",
                {"AGENT_WORKSPACE_DIR": temp_dir},
            ):
                file_io("write", "note.txt", "hello")
                observation = file_io("read", "note.txt")

        self.assertIn('"content": "hello"', observation)


class ToolRegistryTests(unittest.TestCase):
    def test_dispatches_plain_calculator_input(self) -> None:
        registry = ToolRegistry([CalculatorTool()])

        result = registry.execute("calculator", "6 * 7")

        self.assertTrue(result.success)
        self.assertEqual(result.output, 42)

    def test_dispatches_json_arguments(self) -> None:
        registry = ToolRegistry([CalculatorTool()])

        result = registry.execute("calculator", '{"expression": "10 % 3"}')

        self.assertTrue(result.success)
        self.assertEqual(result.output, 1)

    def test_dispatches_plain_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "note.txt").write_text("hello", encoding="utf-8")
            registry = ToolRegistry([ReadFileTool(temp_dir)])

            result = registry.execute("read_file", "note.txt")

        self.assertTrue(result.success)
        self.assertEqual(result.output["content"], "hello")

    def test_returns_error_for_unknown_tool(self) -> None:
        result = ToolRegistry().execute("missing_tool", {})

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ToolNotFoundError")

    def test_default_registry_exposes_all_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = create_default_registry(
                temp_dir, wikipedia_module=FakeWikipedia()
            )

            names = [spec["name"] for spec in registry.get_specs()]

        self.assertEqual(
            names,
            [
                "calculator",
                "wikipedia_search",
                "read_file",
                "write_file",
                "list_files",
            ],
        )


if __name__ == "__main__":
    unittest.main()
