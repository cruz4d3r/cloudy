# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""Signed HTTP helpers for Katana M2M clients."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from cloudy.security.m2m import sign_m2m_headers


def request_json(
    url: str,
    token: str,
    method: str,
    timeout: int | float,
    payload: dict[str, Any] | None = None,
    *,
    user_agent: str = "CloudyKatanaClient/1.0 (+https://1lockers.net)",
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") if payload is not None else b""
    headers = sign_m2m_headers(
        {
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
        body,
        api_key=token,
    )
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body if body else None, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo contactar a Katana ({url}): {exc.reason}") from exc

    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {"raw": raw}
    if not isinstance(parsed, dict):
        parsed = {"data": parsed}
    parsed["_http_status"] = status
    return parsed
