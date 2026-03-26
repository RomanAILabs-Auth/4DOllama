# server.py
# Copyright RomanAILabs - Daniel Harding
# Christ is King.

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fourdollama.config import Settings
from fourdollama.kernel import write_kernel_r4d
from fourdollama.r4d_subprocess import collect_r4d_output, stream_r4d_lines
from fourdollama.registry import ensure_model_registered, load_registry
from fourdollama.schemas import ChatRequest, GenerateRequest, ModelTag, ShowRequest, TagsResponse


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ns(t0: float, t1: float) -> int:
    return int((t1 - t0) * 1_000_000_000)


def create_app() -> FastAPI:
    settings = Settings.load()
    work_root = settings.data_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="4DOllama", version="0.1.0", lifespan=lifespan)

    def _canonical_model(name: str) -> str:
        try:
            return ensure_model_registered(settings, name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/api/tags")
    async def api_tags() -> TagsResponse:
        reg = load_registry(settings)
        models = [ModelTag(**m.as_tag()) for m in reg.values()]
        return TagsResponse(models=models)

    @app.get("/api/ps")
    async def api_ps() -> dict[str, Any]:
        # Ollama-shaped: running models (none for static r4d bridge).
        return {"models": []}

    @app.post("/api/show")
    async def api_show(body: ShowRequest) -> dict[str, Any]:
        key = _canonical_model(body.model)
        reg = load_registry(settings)
        m = reg[key]
        return {
            "license": "RomanAILabs / Roma4D engine bridge",
            "modelfile": f"# Roma4D structural topology: {m.name}\n",
            "parameters": str(m.details),
            "template": "{{ .Prompt }}",
            "system": "",
            "details": {
                "parent_model": "",
                "format": m.details.get("format", "r4d"),
                "family": m.details.get("family", "roma4d"),
                "parameter_size": m.details.get("parameter_size", ""),
                "quantization_level": "4D",
            },
        }

    def _run_kernel(model: str, prompt: str, system: str | None) -> tuple[Path, Path, int]:
        job = uuid.uuid4().hex
        d = work_root / job
        d.mkdir(parents=True, exist_ok=True)
        kpath = d / "kernel.r4d"
        nodes = write_kernel_r4d(kpath, model, prompt, system)
        return d, kpath, nodes

    @app.post("/api/generate")
    async def api_generate(req: GenerateRequest) -> StreamingResponse | JSONResponse:
        model = _canonical_model(req.model)
        t0 = time.perf_counter()
        work_dir, kpath, eval_count = _run_kernel(model, req.prompt, req.system)

        if not req.stream:
            lines, _ = await collect_r4d_output(settings, kpath, work_dir)
            t1 = time.perf_counter()
            text = "".join(ln + "\n" for ln in lines if not ln.startswith("[fourdollama:"))
            return JSONResponse(
                {
                    "model": model,
                    "created_at": _now_iso(),
                    "response": text,
                    "done": True,
                    "total_duration": _ns(t0, t1),
                    "load_duration": 0,
                    "prompt_eval_count": len(req.prompt),
                    "eval_count": eval_count,
                    "context": req.context or [],
                }
            )

        async def gen_ndjson() -> AsyncIterator[bytes]:
            t_stream = time.perf_counter()
            n = 0
            try:
                async for line in stream_r4d_lines(settings, kpath, work_dir):
                    payload = {
                        "model": model,
                        "created_at": _now_iso(),
                        "response": line + "\n",
                        "done": False,
                    }
                    yield (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                    n += 1
            finally:
                t1 = time.perf_counter()
                final = {
                    "model": model,
                    "created_at": _now_iso(),
                    "response": "",
                    "done": True,
                    "total_duration": _ns(t0, t1),
                    "load_duration": _ns(t0, t_stream),
                    "prompt_eval_count": len(req.prompt),
                    "eval_count": n,
                    "context": req.context or [],
                }
                yield (json.dumps(final, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(gen_ndjson(), media_type="application/x-ndjson")

    @app.post("/api/chat")
    async def api_chat(req: ChatRequest) -> StreamingResponse | JSONResponse:
        model = _canonical_model(req.model)
        sys_p = next((m.content for m in req.messages if m.role == "system"), None)
        user_parts = [m.content for m in req.messages if m.role == "user"]
        prompt = "\n".join(user_parts) if user_parts else ""
        t0 = time.perf_counter()
        work_dir, kpath, eval_count = _run_kernel(model, prompt, sys_p)

        if not req.stream:
            lines, _ = await collect_r4d_output(settings, kpath, work_dir)
            t1 = time.perf_counter()
            text = "".join(ln + "\n" for ln in lines if not ln.startswith("[fourdollama:"))
            return JSONResponse(
                {
                    "model": model,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": text},
                    "done": True,
                    "total_duration": _ns(t0, t1),
                    "load_duration": 0,
                    "prompt_eval_count": len(prompt),
                    "eval_count": eval_count,
                }
            )

        async def chat_ndjson() -> AsyncIterator[bytes]:
            t_stream = time.perf_counter()
            n = 0
            try:
                async for line in stream_r4d_lines(settings, kpath, work_dir):
                    content = line + "\n"
                    payload: dict[str, Any] = {
                        "model": model,
                        "created_at": _now_iso(),
                        "message": {"role": "assistant", "content": content},
                        "done": False,
                    }
                    yield (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                    n += 1
            finally:
                t1 = time.perf_counter()
                final = {
                    "model": model,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "total_duration": _ns(t0, t1),
                    "load_duration": _ns(t0, t_stream),
                    "prompt_eval_count": len(prompt),
                    "eval_count": n,
                }
                yield (json.dumps(final, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(chat_ndjson(), media_type="application/x-ndjson")

    return app


app = create_app()
