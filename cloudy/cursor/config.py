# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""Load Cursor API credentials from env or config/cursor.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cloudy.paths import ROOT


def load_cursor_api_key(engine: dict[str, Any] | None = None) -> str:
    """
    Resolve Cursor API key: engine inline → Keychain → api_key_file → env → cursor.json.
    """
    if engine:
        inline = str(engine.get("api_key") or "").strip()
        if inline and "REEMPLAZA" not in inline and "HERE" not in inline.upper():
            return inline
        key_file = str(engine.get("api_key_file") or "").strip()
        if key_file:
            parsed = _read_api_key_file(ROOT / key_file)
            if parsed:
                return parsed

    try:
        from cloudy.secrets.resolver import get_secret_text

        kc = get_secret_text("cursor.api_key")
        if kc:
            return kc
    except Exception:
        pass

    env_key = os.environ.get("CURSOR_API_KEY", "").strip()
    if env_key:
        return env_key

    cfg_path = ROOT / "config" / "cursor.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            key = str(data.get("api_key") or "").strip()
            if key and "HERE" not in key.upper():
                return key
        except json.JSONDecodeError:
            pass

    return ""


def load_cloud_repo(engine: dict[str, Any]) -> tuple[str, str]:
    """Return (repo_url, ref) from engine descriptor or config/cursor-cloud.json."""
    url = str(engine.get("cloud_repo") or "").strip()
    ref = str(engine.get("cloud_ref") or "main").strip() or "main"
    if url:
        return url, ref

    cfg_path = ROOT / "config" / "cursor-cloud.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            url = str(data.get("cloud_repo") or data.get("repo_url") or "").strip()
            ref = str(data.get("cloud_ref") or data.get("ref") or ref).strip() or "main"
        except json.JSONDecodeError:
            pass

    return url, ref


def load_cursor_cloud_config() -> dict[str, Any]:
    """Load config/cursor-cloud.json (repo URL, ref, GitHub push credentials)."""
    cfg_path = ROOT / "config" / "cursor-cloud.json"
    if not cfg_path.is_file():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_github_api_token(cfg: dict[str, Any] | None = None) -> str:
    """GitHub PAT for mirror push (Keychain → env → config/github-api-token.txt)."""
    try:
        from cloudy.secrets.resolver import get_secret_text

        kc = get_secret_text("github.api_token")
        if kc:
            return kc
    except Exception:
        pass

    env_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if env_token:
        return env_token

    data = cfg if cfg is not None else load_cursor_cloud_config()
    token_file = str(data.get("github_token_file") or "config/github-api-token.txt").strip()
    if token_file:
        token = _read_api_key_file(ROOT / token_file)
        if token:
            return token
    return ""


def github_push_url(repo_url: str, username: str, token: str) -> str:
    """Build authenticated HTTPS URL without logging the token."""
    from urllib.parse import urlparse

    parsed = urlparse(repo_url.strip())
    host = parsed.netloc or "github.com"
    path = (parsed.path or "").strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    user = username.strip() or path.split("/")[0] if "/" in path else ""
    return f"https://{user}:{token}@{host}/{path}.git"


def _read_api_key_file(path: Path) -> str:
    if not path.is_file():
        return ""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            return str(json.loads(raw).get("api_key") or "").strip()
        except json.JSONDecodeError:
            return ""
    for line in raw.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""
