# registry.py
# Copyright RomanAILabs - Daniel Harding
# Christ is King.

from __future__ import annotations

import json
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fourdollama.config import Settings


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ModelRecord:
    name: str
    modified_at: str
    size: int
    digest: str
    details: dict[str, Any]

    def as_tag(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"] = self.name
        return d


_BUILTIN: list[ModelRecord] = [
    ModelRecord(
        name="cl40-wxyz:latest",
        modified_at=_iso_now(),
        size=0,
        digest="sha256:fourd-topology-v1",
        details={
            "family": "roma4d",
            "format": "r4d",
            "parameter_size": "4D-Cl(4,0)",
        },
    ),
    ModelRecord(
        name="rotor-chain:latest",
        modified_at=_iso_now(),
        size=0,
        digest="sha256:fourd-rotor-v1",
        details={"family": "roma4d", "format": "r4d"},
    ),
]


def _registry_path(settings: Settings) -> Path:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir / "registry.json"


def load_registry(settings: Settings) -> dict[str, ModelRecord]:
    out: dict[str, ModelRecord] = {m.name: m for m in _BUILTIN}
    path = _registry_path(settings)
    if not path.is_file():
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for row in raw.get("models", []):
        try:
            name = str(row["name"])
            rec = ModelRecord(
                name=name,
                modified_at=str(row.get("modified_at", _iso_now())),
                size=int(row.get("size", 0)),
                digest=str(row.get("digest", "")),
                details=dict(row.get("details", {})),
            )
            out[name] = rec
        except (KeyError, TypeError, ValueError):
            continue
    return out


def save_user_registry(settings: Settings, models: dict[str, ModelRecord]) -> None:
    user_only = {k: v for k, v in models.items() if k not in {b.name for b in _BUILTIN}}
    path = _registry_path(settings)
    payload = {"models": [asdict(v) for v in user_only.values()]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def remove_model(settings: Settings, name: str) -> bool:
    models = load_registry(settings)
    key = normalize_model_name(name)
    if key not in models:
        return False
    if key in {b.name for b in _BUILTIN}:
        return False
    del models[key]
    save_user_registry(settings, models)
    return True


def normalize_model_name(raw: str) -> str:
    s = raw.strip()
    if not s:
        return s
    if ":" not in s:
        s = f"{s}:latest"
    return s


def virtual_model_record(canonical: str) -> ModelRecord:
    h = zlib.crc32(canonical.encode("utf-8", errors="replace")) & 0xFFFFFFFF
    return ModelRecord(
        name=canonical,
        modified_at=_iso_now(),
        size=0,
        digest=f"sha256:roma4d-{h:08x}",
        details={
            "family": "roma4d",
            "format": "r4d",
            "parameter_size": "4D-Cl(4,0)",
            "engine": "r4d",
        },
    )


def ensure_model_registered(settings: Settings, raw_name: str) -> str:
    canonical = normalize_model_name(raw_name)
    if not canonical:
        raise ValueError("model name is empty")
    reg = load_registry(settings)
    if canonical in reg:
        return canonical
    add_user_model(settings, virtual_model_record(canonical))
    return canonical


def add_user_model(settings: Settings, rec: ModelRecord) -> None:
    models = load_registry(settings)
    if rec.name in models:
        return
    models[rec.name] = rec
    save_user_registry(settings, models)
