# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Cloud-only LLM chain for Cloudy Edge (VPS nodes without local Ollama).

Skips ``ollama_local`` engines and optionally ``katana`` to avoid calling back
into Laravel when Edge itself is the Katana fallback.
"""
from __future__ import annotations

import os
import time
from typing import Any

from cloudy.bot.config import llm_cooldown_seconds, ollama_config
from cloudy.bot.llm import (
    LLMError,
    _DEFAULT_CHAT_OPTIONS,
    _DEFAULT_KEEP_ALIVE,
    _build_engine_chain,
    _run_chain,
    cooldown_status,
)
from cloudy.edge.config import edge_config


def cloud_only_chain(*, allow_katana: bool | None = None, channel: str = "") -> list[dict[str, Any]]:
    """
    Resolve the engine chain for Edge: cloud providers only.

    Filters out ``ollama_local`` (no GPU on VPS). By default also drops
    ``katana`` to prevent circular fallback Edge → Katana → Edge.
    """
    cfg = edge_config()
    if allow_katana is None:
        allow_katana = bool(cfg.get("allow_katana_fallback", False))

    prev_edge = os.environ.get("CLOUDY_EDGE_RUNTIME")
    os.environ["CLOUDY_EDGE_RUNTIME"] = "1"
    try:
        engines = _build_engine_chain(
            ollama_config(),
            want_cloud=True,
            allow_cloud_fallback=allow_katana,
            channel=channel,
        )
    finally:
        if prev_edge is None:
            os.environ.pop("CLOUDY_EDGE_RUNTIME", None)
        else:
            os.environ["CLOUDY_EDGE_RUNTIME"] = prev_edge

    filtered: list[dict[str, Any]] = []
    for engine in engines:
        etype = str(engine.get("type") or "")
        if etype == "ollama_local":
            continue
        if engine.get("mac_only"):
            continue
        # Cursor SDK requires Node (.tools/... or PATH). VPS Edge has neither.
        if etype == "cursor":
            continue
        filtered.append(engine)
    return filtered


def describe_cloud_chain(channel: str = "") -> list[dict[str, Any]]:
    """Chain labels + cooldown state for diagnostics (``cloudy edge chain``)."""
    cooling = cooldown_status()
    out: list[dict[str, Any]] = []
    for engine in cloud_only_chain(channel=channel):
        label = str(engine.get("label") or engine.get("type") or "engine")
        out.append({
            "label": label,
            "type": engine.get("type"),
            "model": engine.get("model", ""),
            "cooldown_remaining": cooling.get(label, 0),
        })
    return out


def edge_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    channel: str = "edge",
    profile_context: str = "",
    timeout: int = 90,
) -> tuple[str, str, str]:
    """
    Run ``messages`` through the cloud-only chain.

    Returns (content, engine_label, model). Raises LLMError if every engine fails.
    """
    cfg = edge_config()
    options = dict(_DEFAULT_CHAT_OPTIONS)
    options["temperature"] = float(temperature if temperature is not None else cfg.get("temperature", 0.55))
    options["num_predict"] = int(max_tokens if max_tokens is not None else cfg.get("max_tokens", 500))
    options["num_ctx"] = 8192

    engines = cloud_only_chain(channel=channel)
    if not engines:
        raise LLMError("No hay engines cloud habilitados para Edge (revisa config/llm.json)")

    return _run_chain(
        engines,
        messages,
        options=options,
        think=False,
        keep_alive=_DEFAULT_KEEP_ALIVE,
        timeout=timeout,
        fmt=None,
        channel=channel,
        profile_context=profile_context,
        cooldown_seconds=llm_cooldown_seconds(),
    )


def diagnostic_edge_chat(prompt: str, *, timeout: int = 90) -> dict[str, Any]:
    """One-shot test prompt for ``cloudy edge test``."""
    messages = [{"role": "user", "content": prompt}]
    start = time.time()
    content, label, model = edge_chat(messages, timeout=timeout)
    return {
        "engine": label,
        "model": model,
        "elapsed_s": round(time.time() - start, 2),
        "content": content,
        "chain": [str(e.get("label") or e.get("type")) for e in cloud_only_chain(channel="edge")],
    }


__all__ = [
    "cloud_only_chain",
    "describe_cloud_chain",
    "edge_chat",
    "diagnostic_edge_chat",
]
