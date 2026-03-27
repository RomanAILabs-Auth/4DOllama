#!/usr/bin/env python3
# Copyright RomanAILabs - Daniel Harding
# Roma4D HyperEngine — GGUF cognitive bridge (llama-cpp-python).
# CLI: python gguf_inference.py <request.json> <model.gguf> <n_gpu_layers_int>
# Stdout: one JSON object (PredictLiftability result).

from __future__ import annotations

import json
import re
import sys
import time


SYSTEM_PROMPT = """You are the Roma4D HyperEngine Cognitive Controller (Phase 3).
Reason in 4D spacetime: X, Y, Z are spatial axes; W is the homogeneous component.
Z also represents the temporal cost axis: respect the user's predicted CPython time budget — prefer concise, actionable refactors.
Output ONLY valid JSON on a single line with these keys:
{"liftable_after_refactor": bool, "suggested_kernel_r4d": string, "confidence_wxyz": [w,x,y,z] four floats in [0,1], "rationale": string}
suggested_kernel_r4d must be Roma4D-style: use vec4, rotor, list[vec4], spacetime: par for where appropriate; no Python dicts in the hot loop.
If you cannot propose a safe kernel, set liftable_after_refactor false and still give a short rationale."""


def _parse_json_blob(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def predict_liftability(req: dict, model_path: str, n_gpu_layers: int) -> dict:
    t0 = time.perf_counter()
    snippet = req.get("snippet", "")
    block_reason = req.get("block_reason", "")
    budget_ms = int(req.get("predicted_budget_ms", 5000))

    user = (
        f"Static analyzer block_reason: {block_reason}\n"
        f"predicted_budget_ms (Z-axis hint): {budget_ms}\n"
        f"Python loop snippet:\n```python\n{snippet}\n```\n"
        "Propose a 4D-friendly Roma4D kernel or explain why not."
    )

    try:
        from llama_cpp import Llama
    except ImportError as e:
        return {
            "ok": False,
            "error": "llama_cpp_not_installed",
            "detail": str(e),
            "inference_ms": int((time.perf_counter() - t0) * 1000),
        }

    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            verbose=False,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": "model_load_failed",
            "detail": str(e),
            "inference_ms": int((time.perf_counter() - t0) * 1000),
        }

    try:
        comp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        text = comp["choices"][0]["message"]["content"]
    except Exception as e:
        return {
            "ok": False,
            "error": "inference_failed",
            "detail": str(e),
            "inference_ms": int((time.perf_counter() - t0) * 1000),
        }

    inference_ms = int((time.perf_counter() - t0) * 1000)
    parsed = _parse_json_blob(text)
    if parsed is None:
        return {
            "ok": True,
            "suggested_kernel_r4d": text.strip()[:4000],
            "confidence_wxyz": [0.35, 0.35, 0.35, 0.35],
            "rationale": "model returned non-JSON; raw text in suggested_kernel_r4d",
            "liftable_after_refactor": False,
            "inference_ms": inference_ms,
        }

    wxyz = parsed.get("confidence_wxyz")
    if not isinstance(wxyz, list) or len(wxyz) != 4:
        wxyz = [0.5, 0.5, 0.5, 0.5]
    else:
        wxyz = [float(min(1.0, max(0.0, float(x)))) for x in wxyz]

    return {
        "ok": True,
        "suggested_kernel_r4d": str(parsed.get("suggested_kernel_r4d", "")),
        "confidence_wxyz": wxyz,
        "rationale": str(parsed.get("rationale", "")),
        "liftable_after_refactor": bool(parsed.get("liftable_after_refactor", False)),
        "inference_ms": inference_ms,
    }


def main() -> int:
    if len(sys.argv) != 4:
        print(
            json.dumps({"ok": False, "error": "usage", "detail": "gguf_inference.py <request.json> <model.gguf> <n_gpu_layers>"}),
            flush=True,
        )
        return 2
    req_path, model_path, ngl = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        n_gpu_layers = int(ngl)
    except ValueError:
        print(json.dumps({"ok": False, "error": "bad_n_gpu_layers", "detail": ngl}), flush=True)
        return 2
    try:
        with open(req_path, encoding="utf-8") as f:
            req = json.load(f)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "bad_request_json", "detail": str(e)}), flush=True)
        return 1

    out = predict_liftability(req, model_path, n_gpu_layers)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
