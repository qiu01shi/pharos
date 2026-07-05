"""ShellEntity — runs a shell command and emits its output.

Ports:
    ins:
        command:  TextPort  — the command to run
        stdin:    TextPort  — optional stdin
    outs:
        stdout:   TextPort  — command stdout
        stderr:   TextPort  — command stderr
        exit_code: IntPort  — process exit code
        duration_ms: IntPort — wall time

The command is a plain string passed to `bash -c`. Working dir is
the run's cwd (default: process cwd).
"""

from __future__ import annotations

import asyncio
import time

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue


@entity
class ShellEntity(Entity):
    """Run a shell command, capture stdout/stderr + exit code."""

    ins = {
        "command": InputPort(name="command", accepted_types=["text"]),
        "stdin": InputPort(name="stdin", accepted_types=["text"]),
    }
    outs = {
        "stdout": OutputPort(name="stdout", accepted_types=["text"]),
        "stderr": OutputPort(name="stderr", accepted_types=["text"]),
        "exit_code": OutputPort(name="exit_code", accepted_types=["int"]),
        "duration_ms": OutputPort(name="duration_ms", accepted_types=["int"]),
    }

    def __init__(
        self,
        node_id: str,
        timeout: float = 300.0,
        cwd: str | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.timeout = timeout
        self.cwd = cwd

    async def fire(self, ctx) -> None:  # type: ignore[override]
        cmds = self.ins["command"].consume()
        if not cmds:
            return
        cmd = "".join(t.value.payload for t in cmds)
        stdin_tokens = self.ins["stdin"].consume()
        stdin_data = (
            "".join(t.value.payload for t in stdin_tokens).encode()
            if stdin_tokens
            else None
        )

        t0 = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=(
                    asyncio.subprocess.PIPE if stdin_data is not None else None
                ),
                cwd=self.cwd,
            )
        except Exception as e:
            self.outs["stderr"].emit(
                TypedValue(type="text", payload=f"spawn error: {e}")
            )
            self.outs["exit_code"].emit(TypedValue(type="int", payload=-1))
            return

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=self.timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            self.outs["stderr"].emit(
                TypedValue(type="text", payload=f"timeout after {self.timeout}s")
            )
            self.outs["exit_code"].emit(TypedValue(type="int", payload=-1))
            return

        elapsed = int((time.perf_counter() - t0) * 1000)
        stdout_text = (stdout_b or b"").decode(errors="replace")
        stderr_text = (stderr_b or b"").decode(errors="replace")

        self.outs["stdout"].emit(TypedValue(type="text", payload=stdout_text))
        if stderr_text:
            self.outs["stderr"].emit(TypedValue(type="text", payload=stderr_text))
        self.outs["exit_code"].emit(
            TypedValue(type="int", payload=proc.returncode or 0)
        )
        self.outs["duration_ms"].emit(TypedValue(type="int", payload=elapsed))


__all__ = ["ShellEntity"]
