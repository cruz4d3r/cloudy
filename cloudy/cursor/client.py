# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Cursor SDK chat client for the Cloudy LLM engine chain.

Uses Node @cursor/sdk helper (Python 3.9 Mac venv) with optional in-process
cursor_sdk when Python >= 3.10.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from cloudy.cursor.config import load_cloud_repo, load_cursor_api_key
from cloudy.paths import ROOT, bundled_node_binary


class CursorChatError(RuntimeError):
    """Cursor client failure (mapped to LLMError in llm.py)."""


class CursorQuotaError(CursorChatError):
    """Rate limit / quota (mapped to LLMQuotaError in llm.py)."""


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "user").strip().upper()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        blocks.append(f"[{role}]\n{content}")
    return "\n\n".join(blocks)


def _node_binary() -> str:
    return bundled_node_binary()


def _via_node(
    *,
    api_key: str,
    messages: list[dict[str, str]],
    model: str,
    runtime: str,
    cwd: str,
    cloud_repo: str,
    cloud_ref: str,
    timeout: int,
) -> str:
    helper = ROOT / ".tools" / "cursor-node" / "cursor_chat.mjs"
    if not helper.is_file():
        raise CursorChatError(
            f"Falta helper Cursor en {helper}. Ejecuta: cd .tools/cursor-node && npm install"
        )

    payload = json.dumps(
        {
            "api_key": api_key,
            "messages": messages,
            "model": model,
            "runtime": runtime,
            "cwd": cwd,
            "cloud_repo": cloud_repo,
            "cloud_ref": cloud_ref,
        },
        ensure_ascii=False,
    )

    proc = subprocess.run(
        [_node_binary(), str(helper), payload],
        capture_output=True,
        text=True,
        timeout=max(30, int(timeout)),
        cwd=str(helper.parent),
        env={**os.environ, "CURSOR_API_KEY": api_key},
    )
    out = (proc.stdout or "").strip()
    if not out:
        err = (proc.stderr or "").strip()[:400]
        raise CursorChatError(f"Cursor helper vacío (exit={proc.returncode}): {err}")

    try:
        data = json.loads(out.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise CursorChatError(f"Cursor helper JSON inválido: {out[:300]}") from exc

    if data.get("error"):
        msg = str(data.get("error"))
        if data.get("retryable") or "429" in msg or "rate" in msg.lower():
            raise CursorQuotaError(msg)
        raise CursorChatError(f"Cursor SDK: {msg}")

    text = str(data.get("text") or "").strip()
    if not text:
        raise CursorChatError("Cursor SDK devolvió respuesta vacía")
    return text


def _via_python_sdk(
    *,
    api_key: str,
    messages: list[dict[str, str]],
    model: str,
    runtime: str,
    cwd: str,
    cloud_repo: str,
    cloud_ref: str,
) -> str:
    from cursor_sdk import Agent, AgentOptions, CloudAgentOptions, LocalAgentOptions  # type: ignore

    prompt = _messages_to_prompt(messages)
    opts: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
    }
    if runtime == "cloud":
        if not cloud_repo:
            raise CursorChatError("cursor cloud_repo no configurado")
        opts["cloud"] = CloudAgentOptions(
            repos=[{"url": cloud_repo, "startingRef": cloud_ref}],
            skip_reviewer_request=True,
        )
    else:
        opts["local"] = LocalAgentOptions(cwd=cwd)

    result = Agent.prompt(prompt, AgentOptions(**opts))
    status = str(getattr(result, "status", "") or "")
    text = str(getattr(result, "result", None) or getattr(result, "text", None) or "").strip()
    if status == "error":
        raise CursorChatError(f"Cursor run status=error: {text[:200]}")
    if not text:
        raise CursorChatError("Cursor SDK devolvió respuesta vacía")
    return text


def cursor_chat(
    messages: list[dict[str, str]],
    *,
    engine: dict[str, Any],
    timeout: int = 120,
) -> str:
    """
    Run a chat completion through Cursor Agent.prompt (one-shot, full history in prompt).
    """
    api_key = load_cursor_api_key(engine)
    if not api_key:
        raise CursorChatError("CURSOR_API_KEY no configurada (config/cursor.json o env)")

    model = str(engine.get("model") or "auto").strip() or "auto"
    runtime = str(engine.get("runtime") or "local").strip().lower()
    cwd = str(engine.get("cwd") or ROOT)
    cloud_repo, cloud_ref = load_cloud_repo(engine)

    if runtime == "cloud" and not cloud_repo:
        raise CursorChatError(
            "cursor cloud_repo vacío — configura engine.cloud_repo o config/cursor-cloud.json"
        )

    helper = ROOT / ".tools" / "cursor-node" / "cursor_chat.mjs"
    if helper.is_file():
        try:
            return _via_node(
                api_key=api_key,
                messages=messages,
                model=model,
                runtime=runtime,
                cwd=str(cwd),
                cloud_repo=cloud_repo,
                cloud_ref=cloud_ref,
                timeout=timeout,
            )
        except CursorChatError:
            pass

    try:
        return _via_python_sdk(
            api_key=api_key,
            messages=messages,
            model=model,
            runtime=runtime,
            cwd=cwd,
            cloud_repo=cloud_repo,
            cloud_ref=cloud_ref,
        )
    except ImportError:
        if helper.is_file():
            return _via_node(
                api_key=api_key,
                messages=messages,
                model=model,
                runtime=runtime,
                cwd=str(cwd),
                cloud_repo=cloud_repo,
                cloud_ref=cloud_ref,
                timeout=timeout,
            )
        raise CursorChatError(
            "cursor-sdk no instalado y falta helper Node en .tools/cursor-node"
        )
    except CursorChatError:
        raise
    except Exception as exc:
        msg = str(exc)
        if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
            raise CursorQuotaError(msg) from exc
        raise CursorChatError(f"Cursor SDK: {msg}") from exc


__all__ = ["cursor_chat"]
