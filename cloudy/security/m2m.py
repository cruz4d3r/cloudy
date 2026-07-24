# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
HMAC + timestamp signing for Cloudy ↔ Katana M2M (same contract as John Duran).

Headers:
  Authorization: Bearer <api_key>
  X-Cloudy-Timestamp: <unix seconds>
  X-Cloudy-Signature: sha256=<hmac>
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Mapping

from cloudy.paths import CONFIG_DIR, ROOT


def m2m_hmac_secret() -> str:
    """Read shared HMAC secret (config/bot-m2m-hmac-secret.txt)."""
    path = CONFIG_DIR / "bot-m2m-hmac-secret.txt"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return ""


def sign_m2m_multipart_headers(
    headers: dict[str, str],
    *,
    api_key: str,
    hmac_secret: str | None = None,
) -> dict[str, str]:
    """
    HMAC for multipart/form-data M2M calls (e.g. Katana media/interpret).

    Laravel's Request::getContent() is empty once multipart is parsed, so Katana
    verifies HMAC over an empty body — not the raw multipart bytes.
    """
    return sign_m2m_headers(headers, b"", api_key=api_key, hmac_secret=hmac_secret)


def sign_m2m_headers(
    headers: dict[str, str],
    body: bytes,
    *,
    api_key: str,
    hmac_secret: str | None = None,
) -> dict[str, str]:
    """Add Bearer + timestamp + HMAC signature headers (in-place copy)."""
    out = dict(headers)
    secret = (hmac_secret if hmac_secret is not None else m2m_hmac_secret()).strip()
    key = (api_key or "").strip()
    if key:
        out["Authorization"] = f"Bearer {key}"
    if not secret:
        return out
    ts = str(int(time.time()))
    payload = f"{ts}.".encode("utf-8") + body
    sig = "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    out["X-Cloudy-Timestamp"] = ts
    out["X-Cloudy-Signature"] = sig
    return out


def sign_json_payload(
    payload: Mapping[str, Any] | list[Any],
    *,
    api_key: str,
    hmac_secret: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[bytes, dict[str, str]]:
    """Serialize JSON and return (body bytes, signed headers)."""
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = sign_m2m_headers(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "CloudyBot/1.0 (+https://1lockers.net)",
            **(extra_headers or {}),
        },
        body,
        api_key=api_key,
        hmac_secret=hmac_secret,
    )
    return body, headers


def mac_proxy_token() -> str:
    path = CONFIG_DIR / "mac-proxy-token.txt"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return ""


def read_token_file(rel: str) -> str:
    path = ROOT / rel if not rel.startswith("/") else __import__("pathlib").Path(rel)
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""
