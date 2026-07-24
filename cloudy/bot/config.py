# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Bot configuration loader.

Reads config/whatsapp.json (gitignored; template in whatsapp.json.example).
Multi-company by design: each block under "companies" owns its WhatsApp
number, client directory, schedule and RAG collection. The webhook routes
each inbound event to a company by metadata.phone_number_id, so a single
service can attend N numbers without code changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "whatsapp.json"
LLM_CONFIG_PATH = ROOT / "config" / "llm.json"

DEFAULT_GRAPH_VERSION = "v25.0"
DEFAULT_LLM_COOLDOWN_SECONDS = 900


class BotConfigError(RuntimeError):
    """Raised when config/whatsapp.json is missing or incomplete."""


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise BotConfigError(
            f"Missing {CONFIG_PATH}. Copy config/whatsapp.json.example -> config/whatsapp.json"
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def graph_version(config: dict[str, Any] | None = None) -> str:
    config = config or load_config()
    return str(config.get("graph_version") or DEFAULT_GRAPH_VERSION)


def verify_token(config: dict[str, Any] | None = None) -> str:
    config = config or load_config()
    token = str(config.get("verify_token", "")).strip()
    if not token or token.startswith("CHANGE_ME"):
        raise BotConfigError("Set a real verify_token in config/whatsapp.json")
    return token


def _as_bool(value: Any, default: bool = False) -> bool:
    """Tolerant bool reader: accepts real bools, 1/0, 'true'/'false', 'yes'/'no'."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def ollama_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        if CONFIG_PATH.is_file():
            config = load_config()
        elif LLM_CONFIG_PATH.is_file():
            # Cloudy Edge on VPS: config/llm.json drives the engine chain; whatsapp.json
            # is not required (no WA webhook on that node).
            config = {}
        else:
            raise BotConfigError(
                f"Missing {CONFIG_PATH}. Copy config/whatsapp.json.example -> config/whatsapp.json"
            )
    ollama = config.get("ollama") or {}
    local_chat = str(ollama.get("chat_model") or "qwen3.5:9b")
    local_escalation = str(ollama.get("escalation_model") or local_chat)
    return {
        "url": str(ollama.get("url") or "http://127.0.0.1:11434").rstrip("/"),
        "chat_model": local_chat,
        "escalation_model": local_escalation,
        "embed_model": str(ollama.get("embed_model") or "nomic-embed-text"),
        # Vision for inbound images/video frames (Ollama multimodal).
        "vision_model": str(ollama.get("vision_model") or "moondream"),
        # faster-whisper size: tiny | base | small (CPU).
        "whisper_model": str(ollama.get("whisper_model") or "base"),
        # Embeddings RAG stay local. Vision/audio can use cloud-lite fallbacks
        # (Gemini Flash + Katana Whisper) when local moondream/whisper fail.
        "cloud_enabled": _as_bool(ollama.get("cloud_enabled"), False),
        "cloud_vision_enabled": _as_bool(ollama.get("cloud_vision_enabled"), True),
        "cloud_whisper_enabled": _as_bool(ollama.get("cloud_whisper_enabled"), True),
        "cloud_chat_model": str(ollama.get("cloud_chat_model") or "gpt-oss:120b-cloud"),
        "cloud_escalation_model": str(
            ollama.get("cloud_escalation_model")
            or ollama.get("cloud_chat_model")
            or "gpt-oss:120b-cloud"
        ),
        "cloud_social_model": str(
            ollama.get("cloud_social_model")
            or ollama.get("cloud_chat_model")
            or "gpt-oss:120b-cloud"
        ),
        # Enrutamiento (classify_json) por defecto local: son muchas llamadas
        # chicas por mensaje y no vale la pena gastar cuota cloud en eso.
        "cloud_classify": _as_bool(ollama.get("cloud_classify"), False),
    }


def llm_config() -> dict[str, Any] | None:
    """
    Load the multi-cloud engine chain from config/llm.json.

    Returns None when the file is absent so callers can fall back to the classic
    single-cloud behaviour derived from the 'ollama' block of whatsapp.json.
    """
    if not LLM_CONFIG_PATH.is_file():
        return None
    try:
        data = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BotConfigError(f"config/llm.json inválido: {exc}") from exc
    return data if isinstance(data, dict) else None


def llm_chain() -> list[dict[str, Any]] | None:
    """Ordered list of engine descriptors, or None when llm.json is absent."""
    data = llm_config()
    if not data:
        return None
    chain = data.get("chain")
    if not isinstance(chain, list):
        return None
    return [engine for engine in chain if isinstance(engine, dict)]


def llm_cooldown_seconds() -> int:
    """Seconds an exhausted engine is skipped after a quota/429 error."""
    data = llm_config() or {}
    try:
        value = int(data.get("cooldown_seconds", DEFAULT_LLM_COOLDOWN_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_LLM_COOLDOWN_SECONDS
    return value if value > 0 else DEFAULT_LLM_COOLDOWN_SECONDS


def is_listen_only(company: dict[str, Any]) -> bool:
    """True when webhook should capture traffic but never auto-reply."""
    return _as_bool(company.get("listen_only"), False)


def companies(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Enabled companies only, keyed by alias."""
    config = config or load_config()
    result: dict[str, dict[str, Any]] = {}
    for alias, block in (config.get("companies") or {}).items():
        if block.get("enabled"):
            result[alias] = block
    return result


def get_company(alias: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    all_companies = companies(config)
    if alias not in all_companies:
        raise BotConfigError(
            f"Company '{alias}' not found or disabled in {CONFIG_PATH}. "
            f"Available: {', '.join(sorted(all_companies)) or '(none)'}"
        )
    block = all_companies[alias]
    for key in ("phone_number_id", "access_token"):
        if not str(block.get(key, "")).strip() or str(block.get(key, "")).startswith("YOUR_"):
            raise BotConfigError(f"Company '{alias}' missing '{key}' in {CONFIG_PATH}")
    return block


def company_by_phone_number_id(phone_number_id: str, config: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]] | None:
    """Resolve which company owns an inbound webhook event. None if unknown."""
    for alias, block in companies(config).items():
        if str(block.get("phone_number_id")) == str(phone_number_id):
            return alias, block
    return None


def company_by_waba_id(waba_id: str, config: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]] | None:
    """Resolve company by WhatsApp Business Account id (webhook entry.id)."""
    waba_id = str(waba_id or "").strip()
    if not waba_id:
        return None
    for alias, block in companies(config).items():
        if str(block.get("waba_id") or "") == waba_id:
            return alias, block
    return None


def company_display_digits(company: dict[str, Any]) -> str:
    """Normalized business line digits from _meta.display_phone when present."""
    meta = company.get("_meta") or {}
    return normalize_number(str(meta.get("display_phone") or ""))


def company_by_display_digits(display_digits: str, config: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]] | None:
    """Match metadata.display_phone_number to a tenant business line."""
    display_digits = normalize_number(display_digits)
    if not display_digits:
        return None
    for alias, block in companies(config).items():
        if company_display_digits(block) == display_digits:
            return alias, block
    return None


def company_by_business_line_sender(sender_digits: str, config: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]] | None:
    """
    Coexistence: when staff sends from the business app, Meta may echo with
    from=<business line> while phone_number_id points at another WABA.
    """
    sender_digits = normalize_number(sender_digits)
    if not sender_digits:
        return None
    for alias, block in companies(config).items():
        if company_display_digits(block) == sender_digits:
            return alias, block
    return None


def company_by_peer_business_line(
    peer_digits: str,
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """
    When Meta delivers Unlockers phone_number_id but the peer is another tenant's
    business line (e.g. echo to +1 404 Paola), route to the listen-only tenant.
    """
    peer_digits = normalize_number(peer_digits)
    if not peer_digits:
        return None
    for alias, block in companies(config).items():
        if not is_listen_only(block):
            continue
        if company_display_digits(block) == peer_digits:
            return alias, block
    return None


def observe_mirror_client(company: dict[str, Any]) -> str:
    """Fallback client wa_id when Meta inverts echo/inbound (Paola ↔ Unlockers)."""
    meta = company.get("_meta") or {}
    return normalize_number(str(meta.get("observe_mirror_client") or ""))


def resolve_company_from_webhook(
    metadata: dict[str, Any],
    *,
    waba_id: str = "",
    sender_digits: str = "",
    peer_digits: str = "",
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """
    Pick the tenant for a webhook change.value block.

    Meta sometimes misroutes phone_number_id (seen with Paola + Unlockers).
    Priority: entry WABA id → display line → peer/sender business line → phone_number_id.
    """
    phone_number_id = str(metadata.get("phone_number_id") or "")
    display_digits = normalize_number(str(metadata.get("display_phone_number") or ""))
    peer_digits = normalize_number(peer_digits)

    by_waba = company_by_waba_id(waba_id, config)
    by_phone_id = company_by_phone_number_id(phone_number_id, config)
    by_display = company_by_display_digits(display_digits, config)
    by_sender_line = company_by_business_line_sender(sender_digits, config)
    by_peer_line = company_by_peer_business_line(peer_digits, config)

    if by_waba and by_phone_id and by_waba[0] != by_phone_id[0]:
        return by_waba
    if by_display and by_phone_id and by_display[0] != by_phone_id[0]:
        return by_display
    if by_peer_line and by_phone_id and by_peer_line[0] != by_phone_id[0]:
        return by_peer_line
    if by_sender_line and by_phone_id and by_sender_line[0] != by_phone_id[0]:
        return by_sender_line
    if by_waba:
        return by_waba
    if by_phone_id:
        return by_phone_id
    if by_display:
        return by_display
    if by_peer_line:
        return by_peer_line
    if by_sender_line:
        return by_sender_line
    return None


def normalize_number(raw: str) -> str:
    """Keep digits only ('+57 316...' -> '57316...') for stable dict keys."""
    return "".join(ch for ch in str(raw) if ch.isdigit())


def resolve_client(company: dict[str, Any], wa_number: str) -> dict[str, Any] | None:
    """Look up an authorized client by WhatsApp number. None if not registered."""
    directory = company.get("clients") or {}
    number = normalize_number(wa_number)
    for key, info in directory.items():
        if normalize_number(key) == number:
            return {"number": number, **info}
    return None


def is_owner(company: dict[str, Any], wa_number: str) -> bool:
    return normalize_number(company.get("owner_number", "")) == normalize_number(wa_number)
