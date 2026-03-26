# cli.py
# Copyright RomanAILabs - Daniel Harding
# Christ is King.

from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
import urllib.request
from typing import Annotated, Any, Optional

import typer

from fourdollama import __version__
from fourdollama.config import DEFAULT_PORT, Settings
from fourdollama.engine import ensure_model, stream_engine
from fourdollama.registry import load_registry, normalize_model_name, remove_model

app = typer.Typer(no_args_is_help=True, add_completion=False, help="4dollama — Ollama-like CLI; Roma4D (r4d) engine. Default API :13377 (not Ollama :11434).")


def _http_base(s: Settings) -> str:
    h = s.host
    if h in ("0.0.0.0", "::"):
        h = "127.0.0.1"
    return f"http://{h}:{s.port}"


def _http_json(method: str, path: str, settings: Settings, body: dict[str, Any] | None = None) -> Any:
    url = _http_base(settings) + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=settings.request_timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _try_uvloop() -> None:
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass


@app.command("serve")
def cmd_serve(
    host: str | None = typer.Option(None, "--host", help="bind address"),
    port: int | None = typer.Option(None, "--port", help=f"port (default {DEFAULT_PORT})"),
) -> None:
    s = Settings.load()
    h = host or s.host
    p = port if port is not None else s.port
    _try_uvloop()
    import uvicorn

    print(f'4dollama: listening on http://{h}:{p} (version {__version__}, r4d bridge)', file=sys.stderr)
    uvicorn.run(
        "fourdollama.server:app",
        host=h,
        port=p,
        log_level="info",
        loop="auto",
        access_log=False,
        use_colors=False,
    )


@app.command("version")
def cmd_version() -> None:
    """Print version (same idea as `ollama version`)."""
    typer.echo(__version__)


@app.command("ps")
def cmd_ps() -> None:
    """List running models (Ollama-shaped; empty when nothing is loaded)."""
    s = Settings.load()
    try:
        out = _http_json("GET", "/api/ps", s)
    except urllib.error.URLError as e:
        print(f"4dollama ps: {e}  (start server: 4dollama serve)", file=sys.stderr)
        raise typer.Exit(1) from e
    models = out.get("models") or []
    typer.echo("NAME    ID    SIZE    PROCESSOR    UNTIL")
    for m in models:
        typer.echo(f"{m!s}")


@app.command("pull")
def cmd_pull(
    model: str = typer.Argument(..., help="model name"),
) -> None:
    """GGUF pull + native four_d_engine decode: use the Go `4dollama` from the 4DEngine repo (not this Python r4d bridge)."""
    typer.echo(
        "Pull GGUF with the Go CLI from 4DEngine:  4dollama pull "
        + model
        + "  then  4dollama serve  — inference defaults to native four_d_engine (no hybrid). "
        "Optional hybrid only if you set FOURD_INFERENCE=ollama and OLLAMA_HOST. "
        "This Python package is the Roma4D (r4d) HTTP bridge on :13377.",
        err=True,
    )
    raise typer.Exit(2)


@app.command("show")
def cmd_show(
    model: str = typer.Argument(..., help="model name"),
) -> None:
    s = Settings.load()
    try:
        out = _http_json("POST", "/api/show", s, {"model": model})
    except urllib.error.URLError as e:
        print(f"4dollama show: {e}  (start server: 4dollama serve)", file=sys.stderr)
        raise typer.Exit(1) from e
    typer.echo(json.dumps(out, indent=2))


@app.command("list")
def cmd_list() -> None:
    s = Settings.load()
    reg = load_registry(s)
    typer.echo("NAME                       ID              SIZE      MODIFIED    ")
    for m in sorted(reg.values(), key=lambda x: x.name.lower()):
        sz = f"{m.size:>8}" if m.size else "        —"
        did = (m.digest[:12] + "…") if len(m.digest) > 12 else (m.digest or "—")
        typer.echo(f"{m.name:27}{did:16}{sz}    {m.modified_at}")


@app.command("rm")
def cmd_rm(
    model: str = typer.Argument(..., help="model name to remove"),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    s = Settings.load()
    disp = normalize_model_name(model)
    if not force and not typer.confirm(f"delete '{disp}'?", default=False):
        raise typer.Abort()
    if remove_model(s, model):
        typer.secho(f"deleted '{disp}'", fg="green")
    else:
        typer.secho(f"cannot remove '{disp}' (missing or protected builtin)", fg="red")
        raise typer.Exit(code=1)


def _run_slash(line: str) -> str | None:
    t = line.strip()
    if t in ("/bye", "/exit", "/quit"):
        return "exit"
    if t in ("/help", "/?", "?"):
        print("commands: /clear  /bye", file=sys.stderr)
        return "continue"
    if t == "/clear":
        return "continue"
    return None


@app.command("run")
def cmd_run(
    model: str = typer.Argument(..., metavar="MODEL", help="e.g. qwen2.5 or qwen2.5:latest"),
    prompt: Annotated[Optional[list[str]], typer.Argument()] = None,
    use_server: bool = typer.Option(
        False,
        "--remote",
        help="stream via local HTTP API (127.0.0.1:FOURDOLLAMA_PORT)",
    ),
) -> None:
    s = Settings.load()
    try:
        canonical = ensure_model(s, model)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(1) from e

    one_shot = " ".join(prompt) if prompt else None
    if one_shot is not None and one_shot.strip() != "":
        if use_server:
            _remote_generate_stream(s, canonical, one_shot)
        else:
            asyncio.run(_local_stream(canonical, one_shot, s))
        return

    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            raise typer.Exit(0) from None
        if not line.strip():
            continue
        act = _run_slash(line)
        if act == "exit":
            raise typer.Exit(0)
        if act == "continue":
            continue

        if use_server:
            _remote_generate_stream(s, canonical, line)
        else:
            asyncio.run(_local_stream(canonical, line, s))
        sys.stdout.write("\n")
        sys.stdout.flush()


async def _local_stream(model: str, prompt: str, settings: Settings) -> None:
    try:
        async for chunk in stream_engine(model, prompt, settings=settings):
            sys.stdout.write(chunk)
            sys.stdout.flush()
    except OSError as e:
        print(f"\nengine: {e}", file=sys.stderr)


def _remote_generate_stream(settings: Settings, model: str, prompt: str) -> None:
    h = settings.host
    if h in ("0.0.0.0", "::"):
        h = "127.0.0.1"
    url = f"http://{h}:{settings.port}/api/generate"
    body: dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=settings.request_timeout_sec) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r = obj.get("response", "")
                if r:
                    sys.stdout.write(r)
                    sys.stdout.flush()
                if obj.get("done"):
                    sys.stdout.flush()
    except urllib.error.URLError as e:
        print(f"api: {e}", file=sys.stderr)
        raise typer.Exit(1) from e


def main() -> None:
    app()


if __name__ == "__main__":
    main()
