# r4d_subprocess.py
# Copyright RomanAILabs - Daniel Harding
# Christ is King.

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from fourdollama.config import Settings


def _child_env(settings: Settings) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    if settings.pkg_root is not None and settings.pkg_root.is_dir():
        env["R4D_PKG_ROOT"] = str(settings.pkg_root.resolve())
    return env


async def stream_r4d_lines(
    settings: Settings,
    kernel_path: Path,
    cwd: Path,
) -> AsyncIterator[str]:
    exe = settings.r4d_exe
    if not exe.is_file() and os.path.sep not in str(exe):
        pass
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            str(exe),
            "run",
            str(kernel_path),
            cwd=str(cwd),
            env=_child_env(settings),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None
        while True:
            try:
                raw = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=settings.request_timeout_sec,
                )
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                raise
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                yield line
        rc = await proc.wait()
        if rc != 0:
            err = b""
            if proc.stderr is not None:
                err = await proc.stderr.read()
            msg = err.decode("utf-8", errors="replace").strip() or f"exit {rc}"
            yield f"[fourdollama:r4d_error] {msg}"
    except asyncio.CancelledError:
        if proc is not None and proc.returncode is None:
            proc.kill()
            try:
                await proc.wait()
            except asyncio.CancelledError:
                raise
        raise
    except OSError as e:
        yield f"[fourdollama:os_error] {e}"
    finally:
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError, asyncio.CancelledError):
                pass


async def collect_r4d_output(settings: Settings, kernel_path: Path, cwd: Path) -> tuple[list[str], bool]:
    lines: list[str] = []
    err = False
    async for ln in stream_r4d_lines(settings, kernel_path, cwd):
        lines.append(ln)
        if ln.startswith("[fourdollama:"):
            err = True
    return lines, err
