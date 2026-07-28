# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Ollama client (chat + embeddings) over the local HTTP API.

Model strategy (CPU-friendly, per plan):
  - chat_model (qwen3.5:9b): intent routing and normal replies — low latency.
  - escalation_model (deepseek-r1:14b): complex/technical requests only.
  - embed_model (nomic-embed-text): RAG embeddings.
  - vision_model (moondream): describe inbound images / video frames.

Latency notes (critical for qwen3.5 / reasoning models):
  - Qwen3.5 enables "thinking" by default. Without top-level `"think": false`
    a 1-word reply can burn 400+ tokens (~30–60s) on internal reasoning and
    leave content empty when num_predict is low.
  - We always pass think=False for normal chat/classify/vision; only the
    explicit escalation path (deepseek-r1) keeps thinking enabled.
  - num_ctx / num_predict / keep_alive are capped so the Mac does not thrash
    swapping a 32k context on every webchat turn.

Uses urllib (no external deps) so `bot send`/CLI utilities work even in a
minimal environment; only the webhook server needs fastapi/uvicorn.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from cloudy.bot.config import (
    llm_chain,
    llm_cooldown_seconds,
    llm_profiles,
    ollama_config,
    resolve_llm_profile,
)
from cloudy.bot.runtime_env import is_edge_runtime
from cloudy.paths import ROOT as REPO_ROOT

logger = logging.getLogger("cloudy.bot.llm")

# Per-thread LLM telemetry for the last chain completion (WhatsApp gather threads).
_meta_local = threading.local()


class LLMError(RuntimeError):
    """Raised when Ollama is unreachable or returns an invalid payload."""


class LLMQuotaError(LLMError):
    """
    Raised when an engine reports its free quota is exhausted (HTTP 429, empty
    cloud completion, payment required...). The engine-chain runner puts that
    engine into a cooldown so the next calls skip it instead of hammering it.
    """


def _engine_model(engine: dict[str, Any]) -> str:
    """Resolved model id for telemetry (cursor auto, groq id, ollama tag...)."""
    return str(engine.get("model") or engine.get("type") or "").strip()


def _set_chat_meta(engine: str, model: str) -> None:
    _meta_local.engine = str(engine or "").strip()
    _meta_local.model = str(model or "").strip()


def pop_chat_meta() -> dict[str, str]:
    """Consume engine/model from the last ``chat()`` chain completion in this thread."""
    engine = str(getattr(_meta_local, "engine", "") or "").strip()
    model = str(getattr(_meta_local, "model", "") or "").strip()
    _meta_local.engine = ""
    _meta_local.model = ""
    return {"engine": engine, "model": model}


# Cloudflare (and some provider WAFs) block urllib's default User-Agent, so we
# use an explicit one for OpenAI-compatible providers, mirroring cloudy/katana.py.
_USER_AGENT = "Cloudy/1.0 (+https://1lockers.net)"


# Defaults tuned for Mac + qwen3.5:9b interactive use (WhatsApp + webchat).
_DEFAULT_CHAT_OPTIONS: dict[str, Any] = {
    "temperature": 0.4,
    "num_ctx": 4096,
    "num_predict": 220,
}
_DEFAULT_KEEP_ALIVE = "45m"


def _post(path: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    base = ollama_config()["url"]
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LLMError(f"Ollama unreachable at {base}: {exc}") from exc


def _strip_thinking(content: str) -> str:
    """Remove leaked deepseek-style <think> blocks from visible replies."""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _is_cloud_model(model: str) -> bool:
    """Ollama Cloud models carry a '-cloud'/':cloud' suffix and route to ollama.com."""
    name = str(model).strip().lower()
    return name.endswith("-cloud") or name.endswith(":cloud")


# Reasoning cloud models (gpt-oss) always spend some budget on internal
# reasoning, and those tokens count toward num_predict even with think=False.
# We raise the cap for cloud calls so the visible answer is never starved.
_CLOUD_MIN_NUM_PREDICT = 900


def _chat_once(
    model: str,
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any],
    think: bool,
    keep_alive: str,
    timeout: int,
    fmt: str | None = None,
) -> str:
    """
    One non-streaming /api/chat call against a single model (local or -cloud).
    Raises LLMError on transport failure OR empty content, so callers can fall
    back to the next model in the chain instead of shipping a blank reply.
    """
    call_options = dict(options)
    if _is_cloud_model(model):
        # Only lift an EXISTING small cap (e.g. 220 for chat replies) so the
        # reasoning tokens don't starve the visible answer. If the caller left
        # num_predict unset (e.g. long blog generation), keep it uncapped.
        existing_cap = call_options.get("num_predict")
        if existing_cap is not None and int(existing_cap) > 0:
            call_options["num_predict"] = max(int(existing_cap), _CLOUD_MIN_NUM_PREDICT)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "keep_alive": keep_alive,
        "options": call_options,
    }
    if fmt:
        payload["format"] = fmt
    result = _post("/api/chat", payload, timeout=timeout)
    content = _strip_thinking(str(((result.get("message") or {}).get("content")) or ""))
    if not content:
        # Empty usually means quota/truncation on a cloud model: treat as failure
        # so the fallback chain can try the next engine. On cloud models this is
        # almost always exhausted GPU-time quota -> raise LLMQuotaError so the
        # engine-chain runner puts the provider into cooldown.
        detail = f"Empty content from model '{model}': {str(result)[:180]}"
        if _is_cloud_model(model):
            raise LLMQuotaError(detail)
        raise LLMError(detail)
    return content


def chat_with_fallback(
    models: list[str],
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any],
    think: bool,
    keep_alive: str = _DEFAULT_KEEP_ALIVE,
    timeout: int = 90,
    fmt: str | None = None,
) -> str:
    """
    Try each model in order (cloud-first, then local). On any LLMError — cloud
    quota exhausted, model unreachable, empty content — move to the next model.
    Reusable by the WhatsApp/webchat bot and the social content pipeline so the
    cloud-first policy lives in one place (regla Cloudy: importar de cloudy.*).
    """
    usable = [m for m in models if str(m).strip()]
    if not usable:
        raise LLMError("chat_with_fallback called with no models")
    last_error: LLMError | None = None
    for model in usable:
        try:
            return _chat_once(
                model,
                messages,
                options=options,
                think=think,
                keep_alive=keep_alive,
                timeout=timeout,
                fmt=fmt,
            )
        except LLMError as exc:
            last_error = exc
            continue
    raise last_error or LLMError("All models failed")


def chat_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.4,
    max_tokens: int | None = 900,
    timeout: int = 90,
) -> str:
    """
    One non-streaming completion against any OpenAI Chat-Completions-compatible
    provider (Groq, Google Gemini, OpenRouter, Cerebras...). Uses urllib to keep
    the zero-extra-deps policy of this module.

    Raises LLMQuotaError on HTTP 429/402 (free quota exhausted -> cooldown) and
    LLMError on any other transport/HTTP failure or empty content, so the engine
    chain moves on to the next provider.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    if max_tokens:
        # Reasoning models (Kimi) spend part of the budget on thinking; keep a
        # floor so the visible answer is not starved when callers pass a tiny cap.
        payload["max_tokens"] = max(int(max_tokens), 512)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # noqa: BLE001
            detail = str(exc)
        if exc.code in (429, 402) or "quota" in detail.lower() or "rate limit" in detail.lower():
            raise LLMQuotaError(f"{model} quota/rate limit (HTTP {exc.code}): {detail}") from exc
        raise LLMError(f"{model} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"OpenAI-compatible endpoint unreachable at {base_url}: {exc}") from exc

    choices = data.get("choices") or []
    content = ""
    if choices:
        content = str((choices[0].get("message") or {}).get("content") or "")
    content = _strip_thinking(content)
    if not content:
        raise LLMError(f"Empty content from '{model}': {str(data)[:180]}")
    return content


# --- Engine chain (multi-cloud free -> local -> paid) -----------------------
# In-process cooldown: label -> epoch time until which the engine is skipped.
# Enough for the long-lived `bot serve` process and for CLI runs.
_ENGINE_COOLDOWN: dict[str, float] = {}


def _cooldown_active(label: str) -> bool:
    until = _ENGINE_COOLDOWN.get(label)
    return until is not None and time.time() < until


def _mark_cooldown(label: str, seconds: int) -> None:
    _ENGINE_COOLDOWN[label] = time.time() + max(1, int(seconds))


def cooldown_status() -> dict[str, int]:
    """Remaining cooldown seconds per engine label (for diagnostics)."""
    now = time.time()
    return {
        label: int(until - now)
        for label, until in _ENGINE_COOLDOWN.items()
        if until > now
    }


def _resolve_api_key(engine: dict[str, Any]) -> str:
    """
    Resolve an engine's API key from an inline 'api_key' or an external
    'api_key_file' (path relative to the repo root). The file may contain
    comments (#...) and blank lines; the first real line is used. This keeps the
    secret out of config/llm.json when preferred.
    """
    inline = str(engine.get("api_key") or "").strip()
    if inline and "REEMPLAZA" not in inline and "HERE" not in inline.upper():
        return inline
    key_file = str(engine.get("api_key_file") or "").strip()
    if key_file:
        path = REPO_ROOT / key_file
        if path.is_file():
            raw = path.read_text(encoding="utf-8", errors="replace").strip()
            if raw.startswith("{"):
                try:
                    return str(json.loads(raw).get("api_key") or "").strip()
                except json.JSONDecodeError:
                    pass
            for line in raw.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    return ""


def _run_engine(
    engine: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any],
    think: bool,
    keep_alive: str,
    timeout: int,
    fmt: str | None,
    channel: str,
    profile_context: str,
) -> str:
    """Dispatch one engine descriptor to the right client. Raises LLMError/LLMQuotaError."""
    etype = str(engine.get("type") or "")

    if etype in ("ollama_cloud", "ollama_local"):
        model = str(engine.get("model") or "")
        if not model:
            raise LLMError(f"engine '{engine.get('label')}' sin 'model'")
        return _chat_once(
            model, messages, options=options, think=think,
            keep_alive=keep_alive, timeout=timeout, fmt=fmt,
        )

    if etype == "openai_compatible":
        api_key = _resolve_api_key(engine)
        base_url = str(engine.get("base_url") or "")
        model = str(engine.get("model") or "")
        if not api_key or "REEMPLAZA" in api_key or not base_url or not model:
            raise LLMError(f"engine '{engine.get('label')}' mal configurado (base_url/api_key/model)")
        return chat_openai_compatible(
            base_url, api_key, model, messages,
            temperature=float(options.get("temperature", 0.4)),
            max_tokens=options.get("num_predict") or 900,
            timeout=timeout,
        )

    if etype == "katana":
        from cloudy.bot.katana_chat import cloud_chat

        user_message = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_message = str(m.get("content") or "")
                break
        return cloud_chat(
            messages,
            channel=channel,
            profile_context=profile_context,
            user_message=user_message,
        )

    if etype == "cursor":
        from cloudy.cursor.client import CursorChatError, CursorQuotaError, cursor_chat

        engine_timeout = int(engine.get("timeout") or timeout)
        try:
            return cursor_chat(messages, engine=engine, timeout=engine_timeout)
        except CursorQuotaError as exc:
            raise LLMQuotaError(str(exc)) from exc
        except CursorChatError as exc:
            raise LLMError(str(exc)) from exc
        except (FileNotFoundError, OSError) as exc:
            raise LLMError(f"cursor engine unavailable: {exc}") from exc

    raise LLMError(f"tipo de engine desconocido: '{etype}'")


def _apply_profile_order(engines: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
    """Reorder/filter engines when llm.json defines a profile for this channel."""
    profile_name = resolve_llm_profile(channel)
    if not profile_name:
        return engines
    labels = llm_profiles().get(profile_name)
    if not labels:
        return engines
    by_label = {
        str(engine.get("label") or engine.get("type") or ""): engine
        for engine in engines
    }
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label in labels:
        engine = by_label.get(label)
        if engine is None or label in seen:
            continue
        ordered.append(engine)
        seen.add(label)
    for engine in engines:
        label = str(engine.get("label") or engine.get("type") or "")
        if label not in seen:
            ordered.append(engine)
    return ordered


def _build_engine_chain(
    cfg: dict[str, Any],
    *,
    want_cloud: bool,
    allow_cloud_fallback: bool,
    channel: str = "",
) -> list[dict[str, Any]]:
    """
    Resolve the ordered engine chain for a normal (non-escalation) chat call.

    Uses config/llm.json when present; otherwise synthesises the classic chain
    from the 'ollama' block (Ollama Cloud -> local -> paid) so behaviour is
    unchanged for installs without llm.json.
    """
    configured = llm_chain()
    engines: list[dict[str, Any]] = []

    if configured is not None:
        for engine in configured:
            if not engine.get("enabled", True):
                continue
            etype = str(engine.get("type") or "")
            # Cloud tiers are skipped when the caller wants local-only (classify_json).
            if etype in ("ollama_cloud", "openai_compatible", "cursor") and not want_cloud:
                continue
            # The paid contingency only runs when the caller allows it.
            if etype == "katana" and not allow_cloud_fallback:
                continue
            if etype == "cursor":
                # Cursor Agent runs with full repo cwd — must NOT answer WhatsApp
                # (reads unrelated client work from the monorepo / IDE context).
                if channel in ("whatsapp", "project_whatsapp"):
                    continue
                runtime = str(engine.get("runtime") or "local").strip().lower()
                if is_edge_runtime():
                    # Edge VPS has no bundled Node; cursor-cloud-auto caused 503 FileNotFoundError.
                    continue
                if runtime == "cloud" and not engine.get("allow_on_mac", False):
                    continue
            channels = engine.get("channels")
            if channels and channel:
                allowed = [str(c).strip() for c in channels if str(c).strip()]
                if allowed and channel not in allowed:
                    continue
            engines.append(engine)
        return _apply_profile_order(engines, channel)

    # No llm.json: classic chain from the ollama block.
    if want_cloud and cfg["cloud_enabled"]:
        engines.append({"type": "ollama_cloud", "label": "ollama-cloud", "model": cfg["cloud_chat_model"]})
    engines.append({"type": "ollama_local", "label": "local", "model": cfg["chat_model"]})
    if allow_cloud_fallback:
        engines.append({"type": "katana", "label": "openai-katana"})
    return engines


def _run_chain(
    engines: list[dict[str, Any]],
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any],
    think: bool,
    keep_alive: str,
    timeout: int,
    fmt: str | None,
    channel: str,
    profile_context: str,
    cooldown_seconds: int,
) -> tuple[str, str, str]:
    """
    Walk the engine chain in order and return (content, engine_label, model).

    On quota exhaustion the engine is put into cooldown and skipped; on any other
    failure we simply try the next engine. If every engine is currently cooling
    down we make a second pass ignoring cooldowns (better a slow answer than none).
    """
    last_error: LLMError | None = None

    for respect_cooldown in (True, False):
        tried = 0
        for engine in engines:
            label = str(engine.get("label") or engine.get("type") or "engine")
            if respect_cooldown and _cooldown_active(label):
                continue
            tried += 1
            engine_timeout = int(engine.get("timeout") or timeout)
            try:
                content = _run_engine(
                    engine, messages, options=options, think=think,
                    keep_alive=keep_alive, timeout=engine_timeout, fmt=fmt,
                    channel=channel, profile_context=profile_context,
                )
                model = _engine_model(engine)
                logger.info(
                    "llm chain ok engine=%s model=%s channel=%s",
                    label, model, channel or "-",
                )
                return content, label, model
            except LLMQuotaError as exc:
                _mark_cooldown(label, cooldown_seconds)
                last_error = exc
                continue
            except LLMError as exc:
                last_error = exc
                continue
        # Only do the second (cooldown-ignoring) pass if the first tried nothing.
        if tried > 0:
            break

    raise last_error or LLMError("No hay engines disponibles en la cadena LLM")


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    escalate: bool = False,
    temperature: float = 0.4,
    *,
    think: bool | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    keep_alive: str | None = None,
    channel: str = "whatsapp",
    profile_context: str = "",
    allow_cloud_fallback: bool = True,
    use_ollama_cloud: bool | None = None,
    fmt: str | None = None,
) -> str:
    """
    Single non-streaming chat completion, Ollama-Cloud-first when enabled.

    Model chain (when no explicit `model` is given):
      1. Ollama Cloud model (gpt-oss:120b-cloud) — smarter, zero Mac resources.
      2. Local model (qwen3.5:9b / deepseek-r1:14b) — used when the cloud quota
         is exhausted or the cloud call fails.
      3. OpenAI contingency on 1lockers.net — only if all Ollama calls fail and
         allow_cloud_fallback is True and this is not an escalation call.

    Cloud usage is controlled by config `ollama.cloud_enabled`; `use_ollama_cloud`
    overrides it per call (e.g. classify_json keeps cloud off by default).

    For normal replies think=False (required for qwen3.5 speed). Escalation
    to deepseek-r1 keeps thinking on so the reasoning model can do its job;
    any <think> tags are stripped before the text reaches a client.
    """
    cfg = ollama_config()
    # Escalation models are reasoning models — keep thinking unless caller overrides.
    use_think = bool(escalate) if think is None else bool(think)
    options = dict(_DEFAULT_CHAT_OPTIONS)
    options["temperature"] = temperature
    if escalate:
        options["num_ctx"] = 8192
        options["num_predict"] = 600
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)
    if num_predict is not None:
        options["num_predict"] = int(num_predict)

    want_cloud = cfg["cloud_enabled"] if use_ollama_cloud is None else bool(use_ollama_cloud)

    def _katana_fallback() -> str:
        from cloudy.bot.katana_chat import cloud_chat

        user_message = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_message = str(m.get("content") or "")
                break
        return cloud_chat(
            messages,
            channel=channel,
            profile_context=profile_context,
            user_message=user_message,
        )

    # 1) Explicit model override: respect exactly what the caller asked for
    #    (used by chat_vision, blog generation, etc.). Unchanged behaviour.
    if model:
        try:
            return chat_with_fallback(
                [model], messages, options=options, think=use_think,
                keep_alive=keep_alive or _DEFAULT_KEEP_ALIVE, timeout=90,
            )
        except LLMError:
            if not allow_cloud_fallback or escalate:
                raise
            return _katana_fallback()

    # 2) Escalation stays on the tuned reasoning path (Ollama cloud/local
    #    escalation models with thinking on), not the multi-cloud free chain.
    if escalate:
        local_model = cfg["escalation_model"]
        if want_cloud:
            models = [cfg["cloud_escalation_model"], local_model]
        else:
            models = [local_model]
        return chat_with_fallback(
            models, messages, options=options, think=use_think,
            keep_alive=keep_alive or _DEFAULT_KEEP_ALIVE, timeout=90,
        )

    # 3) Normal chat: walk the multi-cloud engine chain
    #    (free clouds -> local -> paid) with per-engine cooldown.
    engines = _build_engine_chain(
        cfg, want_cloud=want_cloud, allow_cloud_fallback=allow_cloud_fallback,
        channel=channel,
    )
    content, label, model = _run_chain(
        engines, messages, options=options, think=use_think,
        keep_alive=keep_alive or _DEFAULT_KEEP_ALIVE, timeout=90, fmt=fmt,
        channel=channel, profile_context=profile_context,
        cooldown_seconds=llm_cooldown_seconds(),
    )
    _set_chat_meta(label, model)
    return content


def embed(texts: list[str]) -> list[list[float]]:
    """Batch embeddings with nomic-embed-text (Ollama /api/embed)."""
    if not texts:
        return []
    cfg = ollama_config()
    result = _post(
        "/api/embed",
        {
            "model": cfg["embed_model"],
            "input": texts,
            "keep_alive": _DEFAULT_KEEP_ALIVE,
        },
        timeout=120,
    )
    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise LLMError(f"Unexpected embeddings payload from Ollama: {str(result)[:200]}")
    return embeddings


def classify_json(
    system: str,
    user_text: str,
    escalate: bool = False,
    *,
    num_predict: int = 180,
) -> dict[str, Any]:
    """
    Ask the model for a strict JSON object (intent classification, entity
    extraction). Tolerates markdown fences and trailing prose around the JSON.
    Always runs with think=False so classification stays fast.

    Routing stays LOCAL by default (config `ollama.cloud_classify`): these are
    many small per-message calls and are not worth spending cloud quota on.
    """
    cfg = ollama_config()
    raw = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        escalate=escalate,
        temperature=0.1,
        think=False,
        num_predict=max(80, int(num_predict)),
        num_ctx=4096 if not escalate else 8192,
        allow_cloud_fallback=True,
        use_ollama_cloud=cfg["cloud_classify"],
    )
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise LLMError(f"Model did not return JSON: {raw[:200]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"Invalid JSON from model: {raw[:200]}") from exc


def chat_vision(prompt: str, image_b64: str, model: str | None = None) -> str:
    """
    Multimodal chat: prompt + one base64 image (no data: URI prefix).
    Used to turn WhatsApp images/video frames into Spanish descriptions.
    """
    cfg = ollama_config()
    chosen = model or cfg.get("vision_model") or "moondream"
    result = _post(
        "/api/chat",
        {
            "model": chosen,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "think": False,
            "keep_alive": _DEFAULT_KEEP_ALIVE,
            "options": {"temperature": 0.2, "num_ctx": 2048, "num_predict": 180},
        },
        timeout=180,
    )
    content = str(((result.get("message") or {}).get("content")) or "")
    return _strip_thinking(content)


def describe_chain() -> list[dict[str, Any]]:
    """
    Return the resolved engine chain for a normal chat call, annotated with the
    current cooldown state. Used by `cloudy llm chain` for diagnostics.
    """
    cfg = ollama_config()
    engines = _build_engine_chain(cfg, want_cloud=True, allow_cloud_fallback=True)
    cooling = cooldown_status()
    out: list[dict[str, Any]] = []
    for engine in engines:
        label = str(engine.get("label") or engine.get("type") or "engine")
        out.append({
            "label": label,
            "type": engine.get("type"),
            "model": engine.get("model", ""),
            "cooldown_remaining": cooling.get(label, 0),
        })
    return out


def diagnostic_chat(prompt: str, *, timeout: int = 90, channel: str = "diagnostic") -> dict[str, Any]:
    """
    Run one prompt through the engine chain and report which engine answered and
    how long it took. Used by `cloudy llm test` to verify the failover.
    """
    cfg = ollama_config()
    engines = _build_engine_chain(cfg, want_cloud=True, allow_cloud_fallback=True, channel=channel)
    messages = [{"role": "user", "content": prompt}]
    options = {"temperature": 0.4, "num_ctx": 4096, "num_predict": 300}
    start = time.time()
    content, label, model = _run_chain(
        engines, messages, options=options, think=False,
        keep_alive=_DEFAULT_KEEP_ALIVE, timeout=timeout, fmt=None,
        channel="diagnostic", profile_context="",
        cooldown_seconds=llm_cooldown_seconds(),
    )
    return {
        "engine": label,
        "model": model,
        "elapsed_s": round(time.time() - start, 2),
        "content": content,
        "chain": [str(e.get("label") or e.get("type")) for e in engines],
    }


def probe_engines(
    *,
    label: str | None = None,
    channel: str = "",
    timeout: int = 90,
    include_katana: bool = False,
) -> list[dict[str, Any]]:
    """
    Probe each enabled engine individually (no chained fallback).

    Used by `cloudy llm probe` for health checks and hourly monitoring.
    """
    cfg = ollama_config()
    engines = _build_engine_chain(
        cfg,
        want_cloud=True,
        allow_cloud_fallback=include_katana,
        channel=channel,
    )
    if label:
        engines = [
            engine
            for engine in engines
            if str(engine.get("label") or engine.get("type") or "") == label
        ]
    if not include_katana:
        engines = [engine for engine in engines if str(engine.get("type") or "") != "katana"]

    messages = [{"role": "user", "content": "Responde solo: OK"}]
    options = {"temperature": 0.0, "num_ctx": 2048, "num_predict": 32}
    results: list[dict[str, Any]] = []

    for engine in engines:
        engine_label = str(engine.get("label") or engine.get("type") or "engine")
        engine_timeout = int(engine.get("timeout") or timeout)
        started = time.time()
        item: dict[str, Any] = {
            "label": engine_label,
            "type": engine.get("type"),
            "model": engine.get("model", ""),
            "ok": False,
            "latency_ms": 0,
            "error": "",
        }
        try:
            content = _run_engine(
                engine,
                messages,
                options=options,
                think=False,
                keep_alive=_DEFAULT_KEEP_ALIVE,
                timeout=engine_timeout,
                fmt=None,
                channel=channel or "probe",
                profile_context="",
            )
            item["latency_ms"] = int((time.time() - started) * 1000)
            item["ok"] = bool(str(content or "").strip())
            if not item["ok"]:
                item["error"] = "empty content"
        except LLMQuotaError as exc:
            item["latency_ms"] = int((time.time() - started) * 1000)
            item["error"] = f"quota: {exc}"[:200]
        except LLMError as exc:
            item["latency_ms"] = int((time.time() - started) * 1000)
            item["error"] = str(exc)[:200]
        except Exception as exc:  # noqa: BLE001
            item["latency_ms"] = int((time.time() - started) * 1000)
            item["error"] = str(exc)[:200]
        results.append(item)

    return results


def cloud_engines_all_in_cooldown(channel: str = "whatsapp") -> bool:
    """True when every cloud engine in the profile chain is in cooldown."""
    cfg = ollama_config()
    engines = _build_engine_chain(cfg, want_cloud=True, allow_cloud_fallback=False, channel=channel)
    cloud_types = {"ollama_cloud", "openai_compatible", "cursor"}
    cloud_labels = [
        str(engine.get("label") or "")
        for engine in engines
        if str(engine.get("type") or "") in cloud_types
    ]
    if not cloud_labels:
        return False
    cooling = cooldown_status()
    return all(cooling.get(label, 0) > 0 for label in cloud_labels)


def warmup(model: str | None = None) -> None:
    """
    Pre-load the chat model into memory so the first real visitor/client
    does not pay a cold-load penalty. Best-effort: never raises.
    """
    cfg = ollama_config()
    chosen = model or cfg["chat_model"]
    try:
        _post(
            "/api/chat",
            {
                "model": chosen,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "think": False,
                "keep_alive": _DEFAULT_KEEP_ALIVE,
                "options": {"num_ctx": 2048, "num_predict": 1, "temperature": 0.0},
            },
            timeout=180,
        )
    except LLMError:
        return
