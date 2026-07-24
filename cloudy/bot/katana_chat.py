# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Fallback chat/completions against 1lockers.net platform OpenAI
(POST /api/v1/integrations/bot/chat) when Ollama is down.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger("cloudy.bot")


def _signed_headers(api_key: str, body: bytes, extra: dict[str, str] | None = None) -> dict[str, str]:
    from cloudy.security.m2m import sign_m2m_headers

    base = {
        "Accept": "application/json",
        "User-Agent": "CloudyBot/1.0 (+https://1lockers.net)",
        **(extra or {}),
    }
    return sign_m2m_headers(base, body, api_key=api_key)


def _katana_chat_cfg() -> dict[str, Any]:
    from cloudy.webchat.config import katana_leads_config, load_config

    leads = katana_leads_config()
    raw = load_config().get("katana_chat") or {}
    base = str(raw.get("url") or "").strip()
    if not base:
        # Derive from leads URL: .../bot/leads -> .../bot/chat
        leads_url = str(leads.get("url") or "")
        if leads_url.endswith("/leads"):
            base = leads_url[: -len("/leads")] + "/chat"
        else:
            base = "https://1lockers.net/api/v1/integrations/bot/chat"
    return {
        "enabled": bool(raw.get("enabled", True)),
        "url": base,
        "api_key": str(raw.get("api_key") or leads.get("api_key") or "").strip(),
        "timeout_seconds": float(raw.get("timeout_seconds") or leads.get("timeout_seconds") or 25),
        "contacts_url": str(
            raw.get("contacts_url")
            or "https://1lockers.net/api/v1/integrations/bot/contacts/upsert"
        ).strip(),
        "media_url": str(
            raw.get("media_url")
            or "https://1lockers.net/api/v1/integrations/bot/media/interpret"
        ).strip(),
        "client_profile_url": str(
            raw.get("client_profile_url")
            or "https://1lockers.net/api/v1/integrations/bot/client-profile"
        ).strip(),
        "appointments_url": str(
            raw.get("appointments_url")
            or "https://1lockers.net/api/v1/integrations/bot/appointments/has-booking"
        ).strip(),
        "latest_booking_url": str(
            raw.get("latest_booking_url")
            or "https://1lockers.net/api/v1/integrations/bot/appointments/latest-booking"
        ).strip(),
    }


def cloud_chat(
    messages: list[dict[str, str]],
    *,
    channel: str = "whatsapp",
    profile_context: str = "",
    user_message: str = "",
) -> str:
    """
    Call Laravel contingency chat. Raises RuntimeError on failure.
    """
    cfg = _katana_chat_cfg()
    if not cfg.get("enabled") or not cfg.get("api_key"):
        raise RuntimeError("katana chat disabled or missing api_key")

    payload = {
        "messages": [
            {"role": m.get("role"), "content": m.get("content")}
            for m in messages
            if m.get("role") in ("user", "assistant", "system") and m.get("content")
        ][-40:],
        "channel": channel,
        "profile_context": profile_context or "",
        "user_message": user_message or "",
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        cfg["url"],
        data=body,
        method="POST",
        headers=_signed_headers(
            cfg["api_key"],
            body,
            {"Content-Type": "application/json"},
        ),
    )
    try:
        with urllib.request.urlopen(req, timeout=float(cfg["timeout_seconds"])) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"katana chat HTTP {exc.code}: {detail}") from exc

    if not data.get("success") or not str(data.get("reply") or "").strip():
        raise RuntimeError(f"katana chat empty: {data}")
    return str(data["reply"]).strip()


def fetch_client_profile(phone: str, *, company_alias: str = "") -> dict[str, Any] | None:
    """
    Resolve known client from Katana (billing + WA directory).
    Returns profile dict when recognized, else None.
    """
    cfg = _katana_chat_cfg()
    if not cfg.get("api_key"):
        return None
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(digits) < 7:
        return None
    url = cfg["client_profile_url"].rstrip("/") + "?" + urllib.parse.urlencode(
        {"phone": digits, **({"company": company_alias} if company_alias else {})}
    )
    req = urllib.request.Request(
        url,
        method="GET",
        headers=_signed_headers(cfg["api_key"], b""),
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        logger.exception("katana client-profile lookup failed phone=%s", digits[:6] + "***")
        return None
    if not data.get("success") or not data.get("recognized"):
        return None
    profile = data.get("profile")
    return profile if isinstance(profile, dict) else None


def upsert_contact(payload: dict[str, Any]) -> dict[str, Any] | None:
    cfg = _katana_chat_cfg()
    if not cfg.get("api_key"):
        return None
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        cfg["contacts_url"],
        data=body,
        method="POST",
        headers=_signed_headers(
            cfg["api_key"],
            body,
            {"Content-Type": "application/json"},
        ),
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        logger.warning("katana contact upsert HTTP %s: %s", exc.code, detail)
        return _upsert_contact_via_leads_fallback(payload, detail)
    except Exception:
        logger.exception("katana contact upsert failed")
        return _upsert_contact_via_leads_fallback(payload, "")


def _upsert_contact_via_leads_fallback(
    payload: dict[str, Any],
    prior_error: str,
) -> dict[str, Any] | None:
    """Fallback when /contacts/upsert fails — uses /bot/leads (richer create path)."""
    from cloudy.webchat.katana_leads import push_commercial_lead

    phone = str(payload.get("phone") or payload.get("contact") or "").strip()
    name = str(payload.get("name") or "Prospecto WhatsApp").strip()
    summary = str(payload.get("summary") or "").strip()
    if not phone:
        return {"success": False, "error": prior_error or "upsert_failed"}
    data = push_commercial_lead(
        name=name,
        contact=phone,
        message=summary,
        source_detail="wa_bot_contacts_fallback",
    )
    if not data or not data.get("success"):
        return {"success": False, "error": prior_error or "upsert_failed"}
    return {
        "success": True,
        "commercial_lead_id": data.get("commercial_lead_id"),
        "created": True,
        "fallback": "bot_leads",
    }


def interpret_media_cloud(kind: str, file_path: str, caption: str = "") -> str | None:
    """
    Multipart upload to Laravel media interpret. Returns text or None.
    """
    cfg = _katana_chat_cfg()
    if not cfg.get("api_key"):
        return None
    import mimetypes
    import uuid

    boundary = "----CloudyBot" + uuid.uuid4().hex
    filename = file_path.rsplit("/", 1)[-1]
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(file_path, "rb") as fh:
        raw = fh.read()

    parts: list[bytes] = []
    def field(name: str, value: str) -> None:
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    field("kind", kind)
    if caption:
        field("caption", caption)
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
        + raw
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    from cloudy.security.m2m import sign_m2m_multipart_headers

    req = urllib.request.Request(
        cfg["media_url"],
        data=body,
        method="POST",
        headers=sign_m2m_multipart_headers(
            {
                "Accept": "application/json",
                "User-Agent": "CloudyBot/1.0 (+https://1lockers.net)",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            api_key=str(cfg["api_key"]),
        ),
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("success") and data.get("text"):
            return str(data["text"])
    except Exception:
        logger.exception("katana media interpret failed")
    return None


def has_booking_since(phone: str, since_iso: str) -> bool:
    """True if Katana has a non-cancelled appointment booking for phone since timestamp."""
    cfg = _katana_chat_cfg()
    if not cfg.get("api_key"):
        return False
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    since = str(since_iso or "").strip()
    if len(digits) < 7 or not since:
        return False
    query = urllib.parse.urlencode({"phone": digits, "since": since})
    url = f"{cfg['appointments_url']}?{query}"
    req = urllib.request.Request(
        url,
        method="GET",
        headers=_signed_headers(cfg["api_key"], b""),
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return bool(data.get("success")) and bool(data.get("has_booking"))
    except Exception:
        logger.exception("katana has-booking check failed phone=%s", digits[-4:])
        return False


def fetch_latest_booking(phone: str, since_iso: str | None = None) -> dict[str, Any] | None:
    """Última cita confirmada en Katana para este teléfono (p. ej. /agendar público)."""
    cfg = _katana_chat_cfg()
    if not cfg.get("api_key"):
        return None
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(digits) < 7:
        return None
    params: dict[str, str] = {"phone": digits}
    if since_iso:
        params["since"] = str(since_iso).strip()
    query = urllib.parse.urlencode(params)
    url = f"{cfg['latest_booking_url']}?{query}"
    req = urllib.request.Request(
        url,
        method="GET",
        headers=_signed_headers(cfg["api_key"], b""),
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data.get("success"):
            return None
        booking = data.get("booking")
        return booking if isinstance(booking, dict) else None
    except Exception:
        logger.exception("katana latest-booking failed phone=%s", digits[-4:])
        return None
