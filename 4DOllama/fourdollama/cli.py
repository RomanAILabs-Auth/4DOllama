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

from fourdollama.config import Settings
from fourdollama.engine import ensure_model, stream_engine
from fourdollama.registry import load_registry, normalize_model_name, remove_model

app = typer.Typer(no_args_is_help=True, add_completion=False, help="4DOllama — Ollama-shaped CLI over Roma4D (r4d).")


def _try_uvloop() -> None:
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass


@app.command("serve")
def cmd_serve(
    host: str | None = typer.Option(None, "--host", help="bind address"),
    port: int | None = typer.Option(None, "--port", help="port (default 13377)"),
) -> None:
    s = Settings.load()
    h = host or s.host
    p = port if port is not None else s.port
    _try_uvloop()
    import uvicorn

    uvicorn.run(
        "fourdollama.server:app",
        host=h,
        port=p,
        log_level="info",
        loop="auto",
    )


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
        typer.secho(
            "Commands: /help, /bye, /exit, /clear  ·  Same as Ollama: type a message, Enter to run.",
            fg="cyan",
            err=True,
        )
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
        typer.secho(str(e), fg="red", err=True)
        raise typer.Exit(1) from e

    one_shot = " ".join(prompt) if prompt else None
    if one_shot is not None and one_shot.strip() != "":
        if use_server:
            _remote_generate_stream(s, canonical, one_shot)
        else:
            asyncio.run(_local_stream(canonical, one_shot, s))
        return

    typer.secho(f"Using model {canonical}  (Roma4D / r4d backend)", fg="green", err=True)
    typer.secho("", err=True)
    typer.secho(">>> Send a message (/? for help)", fg="white", err=True)

    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("", err=True)
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
        typer.secho(f"\nengine: {e}", fg="red", err=True)


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
        typer.secho(f"api: {e}", fg="red", err=True)
        raise typer.Exit(1) from e


def main() -> None:
    app()


if __name__ == "__main__":
    main()
