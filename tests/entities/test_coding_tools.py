"""Tests for coding tools: bash / read / write / edit / delete / glob / grep."""
from __future__ import annotations

import pytest

from pharos.entities.tools import ToolRegistry
from pharos.entities.tools_coding import register_coding_tools


@pytest.fixture()
def reg() -> ToolRegistry:
    r = ToolRegistry()
    register_coding_tools(r)
    return r


@pytest.fixture()
def tmpdir_path(tmp_path) -> str:
    return str(tmp_path)


# --------------------------------------------------------------------
# bash
# --------------------------------------------------------------------


class TestBashTool:
    async def test_echo(self, reg, tmpdir_path):
        result = await reg.execute(
            "bash",
            {"command": "echo hello-coding"},
            granted_permissions={"bash:execute"},
        )
        assert result.is_error is False
        assert "hello-coding" in result.output

    async def test_no_permission(self, reg):
        result = await reg.execute("bash", {"command": "echo hi"})
        assert result.is_error is True
        assert "permission denied" in (result.error or "")

    async def test_dangerous_blocked(self, reg):
        result = await reg.execute(
            "bash",
            {"command": "rm -rf /"},
            granted_permissions={"bash:execute"},
        )
        assert "[BLOCKED]" in result.output

    async def test_nonzero_exit(self, reg):
        result = await reg.execute(
            "bash",
            {"command": "exit 7"},
            granted_permissions={"bash:execute"},
        )
        assert "exit code: 7" in result.output

    async def test_timeout(self, reg):
        result = await reg.execute(
            "bash",
            {"command": "sleep 10", "timeout": 0.5},
            granted_permissions={"bash:execute"},
        )
        assert "[TIMEOUT]" in result.output


# --------------------------------------------------------------------
# read
# --------------------------------------------------------------------


class TestReadTool:
    async def test_read_file(self, reg, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await reg.execute(
            "read",
            {"path": str(f)},
            granted_permissions={"fs:read"},
        )
        assert "line1" in result.output
        assert "line2" in result.output
        assert "line3" in result.output

    async def test_read_with_offset(self, reg, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        result = await reg.execute(
            "read",
            {"path": str(f), "offset": 3, "limit": 2},
            granted_permissions={"fs:read"},
        )
        assert "c" in result.output
        assert "d" in result.output
        # "e" line content should not appear (only "... (1 more lines)")
        lines = result.output.splitlines()
        content_lines = [ln for ln in lines if not ln.startswith("...") and "|" in ln]
        assert len(content_lines) == 2  # only c and d

    async def test_read_nonexistent(self, reg):
        result = await reg.execute(
            "read",
            {"path": "/nonexistent/file.txt"},
            granted_permissions={"fs:read"},
        )
        assert "[ERROR]" in result.output
        assert "not found" in result.output

    async def test_no_permission(self, reg, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hi")
        result = await reg.execute("read", {"path": str(f)})
        assert result.is_error is True


# --------------------------------------------------------------------
# write
# --------------------------------------------------------------------


class TestWriteTool:
    async def test_write_new_file(self, reg, tmp_path):
        f = tmp_path / "new.txt"
        result = await reg.execute(
            "write",
            {"path": str(f), "content": "hello world"},
            granted_permissions={"fs:write"},
        )
        assert "wrote" in result.output
        assert f.read_text() == "hello world"

    async def test_write_creates_parent_dirs(self, reg, tmp_path):
        f = tmp_path / "a" / "b" / "c.txt"
        result = await reg.execute(
            "write",
            {"path": str(f), "content": "nested"},
            granted_permissions={"fs:write"},
        )
        assert "wrote" in result.output
        assert f.read_text() == "nested"

    async def test_write_overwrite(self, reg, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old")
        await reg.execute(
            "write",
            {"path": str(f), "content": "new"},
            granted_permissions={"fs:write"},
        )
        assert f.read_text() == "new"


# --------------------------------------------------------------------
# edit — precise string replacement
# --------------------------------------------------------------------


class TestEditTool:
    async def test_edit_success(self, reg, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello world\nfoo bar")
        result = await reg.execute(
            "edit",
            {
                "path": str(f),
                "old_string": "hello world",
                "new_string": "HELLO WORLD",
            },
            granted_permissions={"fs:write"},
        )
        assert "edited" in result.output
        assert "HELLO WORLD" in f.read_text()

    async def test_edit_no_match(self, reg, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        result = await reg.execute(
            "edit",
            {
                "path": str(f),
                "old_string": "nonexistent string",
                "new_string": "replacement",
            },
            granted_permissions={"fs:write"},
        )
        assert "[ERROR]" in result.output
        assert "not found" in result.output
        # File unchanged
        assert f.read_text() == "hello world"

    async def test_edit_multiple_matches(self, reg, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("dup\ndup\ndup")
        result = await reg.execute(
            "edit",
            {
                "path": str(f),
                "old_string": "dup",
                "new_string": "unique",
            },
            granted_permissions={"fs:write"},
        )
        assert "[ERROR]" in result.output
        assert "3 times" in result.output

    async def test_edit_file_not_found(self, reg):
        result = await reg.execute(
            "edit",
            {
                "path": "/nonexistent",
                "old_string": "a",
                "new_string": "b",
            },
            granted_permissions={"fs:write"},
        )
        assert "[ERROR]" in result.output


# --------------------------------------------------------------------
# delete
# --------------------------------------------------------------------


class TestDeleteTool:
    async def test_delete_file(self, reg, tmp_path):
        f = tmp_path / "to_delete.txt"
        f.write_text("bye")
        result = await reg.execute(
            "delete",
            {"path": str(f)},
            granted_permissions={"fs:write"},
        )
        assert "deleted" in result.output
        assert not f.exists()

    async def test_delete_nonexistent(self, reg):
        result = await reg.execute(
            "delete",
            {"path": "/nonexistent"},
            granted_permissions={"fs:write"},
        )
        assert "[ERROR]" in result.output

    async def test_delete_directory_fails(self, reg, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        result = await reg.execute(
            "delete",
            {"path": str(d)},
            granted_permissions={"fs:write"},
        )
        assert "[ERROR]" in result.output
        assert "directory" in result.output


# --------------------------------------------------------------------
# glob
# --------------------------------------------------------------------


class TestGlobTool:
    async def test_glob_finds_files(self, reg, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("x")
        (tmp_path / "c.txt").write_text("x")
        result = await reg.execute(
            "glob",
            {"pattern": "*.py", "path": str(tmp_path)},
            granted_permissions={"fs:read"},
        )
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "c.txt" not in result.output

    async def test_glob_no_matches(self, reg, tmp_path):
        result = await reg.execute(
            "glob",
            {"pattern": "*.nonexistent", "path": str(tmp_path)},
            granted_permissions={"fs:read"},
        )
        assert "no matches" in result.output


# --------------------------------------------------------------------
# grep
# --------------------------------------------------------------------


class TestGrepTool:
    async def test_grep_finds_pattern(self, reg, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("def world():\n    pass\n")
        result = await reg.execute(
            "grep",
            {"pattern": "def", "path": str(tmp_path), "file_glob": "*.py"},
            granted_permissions={"fs:read"},
        )
        assert "hello" in result.output
        assert "world" in result.output

    async def test_grep_no_matches(self, reg, tmp_path):
        (tmp_path / "a.py").write_text("nothing here")
        result = await reg.execute(
            "grep",
            {"pattern": "xyz123", "path": str(tmp_path)},
            granted_permissions={"fs:read"},
        )
        assert "no matches" in result.output

    async def test_grep_regex(self, reg, tmp_path):
        (tmp_path / "a.py").write_text("import os\nimport sys\nfrom typing import Any")
        result = await reg.execute(
            "grep",
            {"pattern": r"import\s+\w+", "path": str(tmp_path)},
            granted_permissions={"fs:read"},
        )
        assert "import os" in result.output
        assert "import sys" in result.output


# --------------------------------------------------------------------
# Integration: ToolRegistry → list_tools → Context
# --------------------------------------------------------------------


class TestToolTypes:
    def test_list_tools_returns_tool_objects(self, reg):
        """Verify list_tools() returns Tool objects, not dicts."""
        from pharos.llm.types import Tool

        tools = reg.list_tools()
        assert len(tools) == 7  # bash/read/write/edit/delete/glob/grep
        for t in tools:
            assert isinstance(t, Tool)
            assert t.name
            assert t.description
            assert isinstance(t.parameters, dict)

    def test_tool_names(self, reg):
        names = set(reg.tool_names())
        assert names == {"bash", "read", "write", "edit", "delete", "glob", "grep"}

    async def test_parallel_execution(self, reg, tmp_path):
        """execute_batch runs tool calls in parallel."""
        import time

        f1 = tmp_path / "f1.txt"
        f2 = tmp_path / "f2.txt"
        f1.write_text("content1")
        f2.write_text("content2")

        calls = [
            {"name": "read", "arguments": {"path": str(f1)}, "tool_call_id": "1"},
            {"name": "read", "arguments": {"path": str(f2)}, "tool_call_id": "2"},
        ]
        start = time.monotonic()
        results = await reg.execute_batch(
            calls, granted_permissions={"fs:read"}
        )
        elapsed = time.monotonic() - start
        assert len(results) == 2
        assert "content1" in results[0].output
        assert "content2" in results[1].output
        # Both reads should complete in well under 1s (parallel)
        assert elapsed < 1.0