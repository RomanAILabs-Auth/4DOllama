# config.py
# Copyright RomanAILabs - Daniel Harding
# Christ is King.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    r4d_exe: Path
    pkg_root: Path | None
    data_dir: Path
    request_timeout_sec: float

    @staticmethod
    def load() -> Settings:
        host = os.environ.get("FOURDOLLAMA_HOST", "127.0.0.1").strip() or "127.0.0.1"
        port = _env_int("FOURDOLLAMA_PORT", 13377)
        r4d_raw = os.environ.get("FOURDOLLAMA_R4D", "").strip()
        if r4d_raw:
            r4d = Path(r4d_raw).expanduser()
        else:
            r4d = Path("r4d.exe")
            for name in ("r4d.exe", "r4d", "roma4d.exe", "roma4d"):
                p = _which_path(name)
                if p is not None:
                    r4d = p
                    break
        pr = os.environ.get("R4D_PKG_ROOT", "").strip()
        pkg = Path(pr).expanduser() if pr else None
        data = Path(os.environ.get("FOURDOLLAMA_DATA", "")).expanduser()
        if not str(data):
            data = Path.home() / ".fourdollama"
        to = float(os.environ.get("FOURDOLLAMA_R4D_TIMEOUT", "120") or "120")
        return Settings(
            host=host,
            port=port,
            r4d_exe=r4d,
            pkg_root=pkg,
            data_dir=data,
            request_timeout_sec=to,
        )


def _which_path(name: str) -> Path | None:
    import shutil

    w = shutil.which(name)
    return Path(w) if w else None
