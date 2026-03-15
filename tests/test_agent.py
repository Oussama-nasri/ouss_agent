"""
Tests you can run without Ollama running.
The LLM is mocked — only real logic is tested.

Run: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock

from agent.core     import Agent
from agent.memory   import Memory
from tools.registry import ToolRegistry
from tools.file_io  import read_file, write_file, list_files
from tools.code_exec import run_python


# ─── Memory Tests ────────────────────────────────────────────────────────────

class TestMemory:
    def test_adds_messages(self):
        m = Memory()
        m.add("user", "hello")
        m.add("assistant", "hi")
        assert len(m) == 2

    def test_system_preserved_after_clear(self):
        m = Memory(system="You are helpful.")
        m.add("user", "test")
        m.clear(keep_system=True)
        msgs = m.get_all()
        assert msgs[0]["role"] == "system"
        assert len(m) == 0

    def test_sliding_window(self):
        from config.settings import settings
        original = settings.max_history
        settings.max_history = 4

        m = Memory()
        for i in range(10):
            m.add("user", f"msg {i}")

        assert len(m) == 4  # only last 4 kept
        settings.max_history = original


# ─── Tool Registry Tests ──────────────────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_run(self):
        r = ToolRegistry()

        @r.register("double")
        def double(n: str) -> str:
            return str(int(n) * 2)

        result = r.run("double", n="5")
        assert result == "10"

    def test_unknown_tool_returns_error(self):
        r = ToolRegistry()
        result = r.run("nonexistent")
        assert "not found" in result

    def test_prompt_docs_uses_docstring(self):
        r = ToolRegistry()

        @r.register("my_tool")
        def my_tool(x: str) -> str:
            """Does something useful."""
            return x

        docs = r.prompt_docs()
        assert "my_tool" in docs
        assert "Does something useful" in docs


# ─── File I/O Tests ───────────────────────────────────────────────────────────

class TestFileIO:
    def test_write_and_read(self):
        write_file("test_rw.txt", "hello world")
        content = read_file("test_rw.txt")
        assert content == "hello world"

    def test_read_missing_file(self):
        result = read_file("does_not_exist.txt")
        assert "does not exist" in result

    def test_list_files_includes_written(self):
        write_file("list_test.txt", "data")
        result = list_files()
        assert "list_test.txt" in result

    def test_path_traversal_blocked(self):
        result = write_file("../../etc/passwd", "evil")
        assert "denied" in result.lower() or "Error" in result


# ─── Code Execution Tests ─────────────────────────────────────────────────────

class TestCodeExec:
    def test_simple_math(self):
        result = run_python("print(2 + 2)")
        assert result == "4"

    def test_blocked_import(self):
        result = run_python("import os\nprint(os.getcwd())")
        assert "blocked" in result.lower()

    def test_timeout(self):
        result = run_python("while True: pass", timeout=1)
        assert "timed out" in result.lower()

    def test_syntax_error(self):
        result = run_python("def broken(")
        assert "Error" in result


# ─── Agent Tests (LLM mocked) ─────────────────────────────────────────────────

class TestAgent:
    def _make_agent(self, llm_responses: list[str]) -> Agent:
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = llm_responses
        return Agent(llm=mock_llm, system="Test system prompt.")

    def test_direct_answer(self):
        agent = self._make_agent(["The sky is blue."])
        response = agent("What color is the sky?")
        assert response == "The sky is blue."

    def test_tool_call_then_answer(self):
        agent = self._make_agent([
            'TOOL: run_python({"code": "print(6*7)"})',
            "The answer is 42.",
        ])
        response = agent("What is 6 times 7?")
        assert response == "The answer is 42."

    def test_max_steps_guard(self):
        # LLM keeps calling tools forever → should hit max_steps
        from config.settings import settings
        original = settings.max_steps
        settings.max_steps = 3

        agent = self._make_agent([
            'TOOL: list_files({})',
            'TOOL: list_files({})',
            'TOOL: list_files({})',
            'TOOL: list_files({})',  # never reached
        ])
        response = agent("Loop forever")
        assert "step limit" in response.lower()
        settings.max_steps = original