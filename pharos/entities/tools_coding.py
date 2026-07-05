"""Coding agent tool set — bash / read / write / edit / delete.

Each tool declares `required_permission` so the RBAC layer can
gate access:

    bash:execute       — run shell commands
    fs:read            — read files
    fs:write           — write / create / edit / delete files

Usage:
    from pharos.entities.tools_coding import register_coding_tools

    reg = ToolRegistry()
    register_coding_tools(reg)

CLI:
    pharos run coder.yaml --input "..." \\
        --grant bash:execute --grant fs:read --grant fs:write
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pharos.entities.tools import ToolRegistry

# DANGEROUS PATTERNS — blocked even with --grant bash:execute
_BASH_BLOCKLIST: list[str] = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf $HOME",
    "mkfs",
    "dd if=/dev/zero of=/dev/",
    ":(){:|:&};:",  # fork bomb
    "> /dev/sda",
    "chmod -R 777 /",
]


def _is_dangerous(cmd: str) -> str | None:
    """Return a reason string if the command is blocked, else None."""
    stripped = cmd.strip().lower()
    for pat in _BASH_BLOCKLIST:
        if pat.lower() in stripped:
            return f"blocked: command matches dangerous pattern {pat!r}"
    return None


# --------------------------------------------------------------------
# bash
# --------------------------------------------------------------------


def _register_bash(reg: ToolRegistry) -> None:
    reg.register(
        name="bash",
        description=(
            "Run a bash command and return stdout + stderr. "
            "The command runs in the graph's working directory. "
            "Timeout: 30 seconds (default)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default 30).",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
        fn=_bash_fn,
        required_permission="bash:execute",
    )


def _bash_fn(command: str, timeout: float = 30.0) -> str:
    """Execute a bash command. Returns stdout+stderr on success,
    or an error message."""
    blocked = _is_dangerous(command)
    if blocked:
        return f"[BLOCKED] {blocked}"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        out = proc.stdout
        if proc.stderr:
            out += f"\n[stderr]\n{proc.stderr}"
        if proc.returncode != 0:
            out += f"\n[exit code: {proc.returncode}]"
        return out.strip() or "[no output]"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] command exceeded {timeout}s"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# --------------------------------------------------------------------
# read
# --------------------------------------------------------------------


def _register_read(reg: ToolRegistry) -> None:
    reg.register(
        name="read",
        description=(
            "Read a file and return its contents. Supports line offset "
            "and limit for large files. Lines are 1-indexed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative or absolute).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed, default 1).",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to read (default 2000).",
                    "default": 2000,
                },
            },
            "required": ["path"],
        },
        fn=_read_fn,
        required_permission="fs:read",
    )


def _read_fn(path: str, offset: int = 1, limit: int = 2000) -> str:
    """Read a file, return contents with line numbers."""
    p = Path(path)
    if not p.exists():
        return f"[ERROR] file not found: {path}"
    if not p.is_file():
        return f"[ERROR] not a file: {path}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, offset - 1)
        end = start + limit
        selected = lines[start:end]
        # Format: LINE_NUM|content (like read_file tool)
        formatted = [f"{start + i + 1:>6}|{line}" for i, line in enumerate(selected)]
        result = "\n".join(formatted)
        if end < len(lines):
            result += f"\n... ({len(lines) - end} more lines)"
        return result or "[empty file]"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# --------------------------------------------------------------------
# write
# --------------------------------------------------------------------


def _register_write(reg: ToolRegistry) -> None:
    reg.register(
        name="write",
        description=(
            "Write content to a file (overwrites if exists, creates "
            "if not). Parent directories are created automatically."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write.",
                },
            },
            "required": ["path", "content"],
        },
        fn=_write_fn,
        required_permission="fs:write",
    )


def _write_fn(path: str, content: str) -> str:
    """Write a file, creating parent dirs."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# --------------------------------------------------------------------
# edit — precise string replacement with anchor validation
# --------------------------------------------------------------------


def _register_edit(reg: ToolRegistry) -> None:
    reg.register(
        name="edit",
        description=(
            "Replace a specific string in a file. The old_string must "
            "appear exactly once in the file (anchor-based edit). "
            "If it appears zero or multiple times, the edit fails."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        fn=_edit_fn,
        required_permission="fs:write",
    )


def _edit_fn(path: str, old_string: str, new_string: str) -> str:
    """Precise string replacement. Fails if old_string is not found
    or appears more than once."""
    p = Path(path)
    if not p.exists():
        return f"[ERROR] file not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return (
                f"[ERROR] old_string not found in {path}. "
                f"The file may have been modified — read it again."
            )
        if count > 1:
            return (
                f"[ERROR] old_string appears {count} times in {path}. "
                f"Provide a longer/more specific old_string (anchor)."
            )
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"edited {path}: replaced 1 occurrence"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# --------------------------------------------------------------------
# delete
# --------------------------------------------------------------------


def _register_delete(reg: ToolRegistry) -> None:
    reg.register(
        name="delete",
        description="Delete a file. Fails if the path is a directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to delete.",
                },
            },
            "required": ["path"],
        },
        fn=_delete_fn,
        required_permission="fs:write",
    )


def _delete_fn(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"[ERROR] file not found: {path}"
    if p.is_dir():
        return f"[ERROR] {path} is a directory, not a file"
    try:
        p.unlink()
        return f"deleted {path}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# --------------------------------------------------------------------
# glob — find files by pattern
# --------------------------------------------------------------------


def _register_glob(reg: ToolRegistry) -> None:
    reg.register(
        name="glob",
        description="Find files matching a glob pattern (e.g. '**/*.py').",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.ts').",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory (default: current).",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
        fn=_glob_fn,
        required_permission="fs:read",
    )


def _glob_fn(pattern: str, path: str = ".") -> str:
    try:
        base = Path(path)
        matches = sorted(base.glob(pattern))
        if not matches:
            return "[no matches]"
        return "\n".join(str(m) for m in matches[:200])
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# --------------------------------------------------------------------
# grep — search file contents
# --------------------------------------------------------------------


def _register_grep(reg: ToolRegistry) -> None:
    reg.register(
        name="grep",
        description=(
            "Search for a regex pattern in files. Returns matching lines "
            "with file paths and line numbers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in.",
                    "default": ".",
                },
                "file_glob": {
                    "type": "string",
                    "description": "File pattern filter (e.g. '*.py').",
                    "default": "*",
                },
            },
            "required": ["pattern"],
        },
        fn=_grep_fn,
        required_permission="fs:read",
    )


def _grep_fn(pattern: str, path: str = ".", file_glob: str = "*") -> str:
    import re

    try:
        regex = re.compile(pattern)
        base = Path(path)
        if base.is_file():
            files = [base]
        else:
            files = sorted(base.rglob(file_glob))
            files = [f for f in files if f.is_file()]
        results: list[str] = []
        for f in files[:500]:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{f}:{i}: {line}")
                        if len(results) >= 200:
                            results.append("... (truncated, 200+ matches)")
                            return "\n".join(results)
            except Exception:
                pass
        if not results:
            return "[no matches]"
        return "\n".join(results)
    except re.error as e:
        return f"[ERROR] invalid regex: {e}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# --------------------------------------------------------------------
# register all
# --------------------------------------------------------------------


def register_coding_tools(reg: ToolRegistry) -> None:
    """Register the full coding tool set on `reg`.

    Tools registered:
        bash, read, write, edit, delete, glob, grep

    Required permissions:
        bash:execute  — bash
        fs:read      — read, glob, grep
        fs:write     — write, edit, delete
    """
    _register_bash(reg)
    _register_read(reg)
    _register_write(reg)
    _register_edit(reg)
    _register_delete(reg)
    _register_glob(reg)
    _register_grep(reg)


__all__ = ["register_coding_tools"]