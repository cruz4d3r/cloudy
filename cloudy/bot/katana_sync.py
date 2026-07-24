# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Bidirectional sync between Mac CloudyBot and Katana (1lockers.net):

  sync-kb        data/rag/unlockers (+ empresa.md) → clients/.../bot_kb/unlockers
  push-turns     local message_log / explicit turns → POST /bot/turns
  pull-learning  GET /bot/turns → local exports + message_log + optional digest KB

Uses the same Bearer key as katana_leads (BOT_LEADS_API_KEY).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from cloudy.bot.company_paths import (
    bundle_content_hash,
    katana_mirror_dir,
    knowledge_source_paths,
    sync_state_path,
    write_manifest,
)
from cloudy.bot.config import ROOT, get_company

logger = logging.getLogger("cloudy.bot")

_BOT_KB_REL = Path("clients/1lockers/sites/web/storage/app/bot_kb/unlockers")
_RAG_DIR = Path("data/rag/unlockers")
_EMPRESA_MD = Path("clients/1lockers/meta/unlockers-cloud-empresa.md")
_DIGEST_NAME = "19-aprendido-katana.md"
_LEARNING_CURSOR = Path("data/rag/unlockers/.katana_learning_cursor")
_LEARNING_EXPORTS = Path("data/rag/conversaciones/katana_sync")
_STATE_DIR = Path("data/rag/unlockers")


def _katana_cfg() -> dict[str, Any]:
    from cloudy.webchat.config import katana_leads_config

    leads = katana_leads_config()
    base = str(leads.get("url") or "").rstrip("/")
    if base.endswith("/leads"):
        base = base[: -len("/leads")]
    elif not base:
        base = "https://1lockers.net/api/v1/integrations/bot"
    return {
        "base": base,
        "api_key": str(leads.get("api_key") or "").strip(),
        "timeout_seconds": float(leads.get("timeout_seconds") or 30),
    }


def _request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
) -> dict[str, Any]:
    from cloudy.security.katana_http import request_json

    cfg = _katana_cfg()
    if not cfg["api_key"]:
        raise RuntimeError("katana api_key vacío (webchat.json katana_leads)")
    url = cfg["base"].rstrip("/") + "/" + path.lstrip("/")
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return request_json(
        url,
        cfg["api_key"],
        method.upper(),
        float(cfg["timeout_seconds"]),
        payload,
        user_agent="CloudyBot/1.0 (+https://1lockers.net)",
    )


_PAOLA_PHONE_NUMBER_ID = "781388565832078"
_PAOLA_LINE_DIGITS = "14047192771"
_UNLOCKERS_316_DIGITS = "573166248968"


def push_inbox_mirror(
    *,
    phone_number_id: str,
    contact: str,
    direction: str,
    content: str,
    external_id: str = "",
    from_name: str = "",
) -> dict[str, Any]:
    """POST /bot/inbox-mirror → channel_messages en Katana."""
    payload: dict[str, Any] = {
        "phone_number_id": phone_number_id,
        "contact": contact,
        "direction": direction,
        "content": content,
    }
    if external_id:
        payload["external_id"] = external_id
    if from_name:
        payload["from_name"] = from_name
    return _request("POST", "inbox-mirror", payload=payload)


def push_wa_channel_token(*, phone_number_id: str, access_token: str) -> dict[str, Any]:
    """POST /bot/wa-channel-token → guarda permanent_token en contact_channels."""
    return _request(
        "POST",
        "wa-channel-token",
        payload={
            "phone_number_id": phone_number_id,
            "access_token": access_token.strip(),
        },
    )


def maybe_mirror_unlockers_to_paola_inbox(
    empresa: str,
    to_number: str,
    text: str,
    wamid: str,
) -> dict[str, Any] | None:
    """
    Meta no entrega webhook inbound al WABA Paola cuando el 316 envía vía Cloud API.
    Espejamos manualmente al Inbox canal Paola (contacto = línea 316).
    """
    from cloudy.bot.config import normalize_number

    if empresa != "unlockers":
        return None
    if normalize_number(to_number) != _PAOLA_LINE_DIGITS:
        return None
    try:
        return push_inbox_mirror(
            phone_number_id=_PAOLA_PHONE_NUMBER_ID,
            contact=_UNLOCKERS_316_DIGITS,
            direction="inbound",
            content=text.strip(),
            external_id=wamid,
            from_name="Unlockers Cloud (316)",
        )
    except Exception as exc:
        logger.warning("inbox mirror unlockers→paola failed: %s", exc)
        return None


def _kb_markdown_sources(empresa: str) -> list[Path]:
    """Top-level tenant KB files for Katana mirror (no by-contact personas)."""
    company = get_company(empresa)
    files: list[Path] = []
    for src in knowledge_source_paths(empresa, company):
        if src.is_file() and src.suffix == ".md":
            rel = str(src)
            if "/by-contact/" in rel or "/by-alias/" in rel:
                continue
            files.append(src)
        elif src.is_dir():
            if "/by-contact" in str(src) or "/by-alias" in str(src):
                continue
            files.extend(sorted(p for p in src.glob("*.md") if not p.name.startswith(".")))
    # Legacy unlockers flat KB during migration.
    if empresa == "unlockers":
        legacy = ROOT / _RAG_DIR
        if legacy.is_dir():
            for path in sorted(legacy.glob("*.md")):
                if path not in files:
                    files.append(path)
    return files


def sync_kb_local(empresa: str = "unlockers") -> dict[str, Any]:
    """
    Copy Mac RAG markdown into the local Katana bot_kb mirror (UTF-8).
    One isolated folder per tenant: storage/app/bot_kb/{empresa}/.
    """
    from cloudy.bot.prompts import prompts_dir as tenant_prompts_dir

    dest_dir = katana_mirror_dir(empresa)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []

    for path in _kb_markdown_sources(empresa):
        target = dest_dir / path.name
        text = path.read_text(encoding="utf-8")
        if target.is_file() and target.read_text(encoding="utf-8") == text:
            skipped.append(path.name)
            continue
        target.write_text(text, encoding="utf-8")
        copied.append(path.name)

    if empresa == "unlockers":
        empresa_src = ROOT / _EMPRESA_MD
        if empresa_src.is_file():
            target = dest_dir / "unlockers-cloud-empresa.md"
            text = empresa_src.read_text(encoding="utf-8")
            if not target.is_file() or target.read_text(encoding="utf-8") != text:
                target.write_text(text, encoding="utf-8")
                copied.append("unlockers-cloud-empresa.md")
            else:
                skipped.append("unlockers-cloud-empresa.md")

    prompts_src = tenant_prompts_dir(empresa)
    if prompts_src.is_dir():
        for path in sorted(prompts_src.glob("*.md")):
            target = dest_dir / path.name
            text = path.read_text(encoding="utf-8")
            if target.is_file() and target.read_text(encoding="utf-8") == text:
                skipped.append(path.name)
                continue
            target.write_text(text, encoding="utf-8")
            copied.append(path.name)
        handoff_src = prompts_src / "bot-handoff.json"
        if handoff_src.is_file():
            target = dest_dir / "bot-handoff.json"
            text = handoff_src.read_text(encoding="utf-8")
            if not target.is_file() or target.read_text(encoding="utf-8") != text:
                target.write_text(text, encoding="utf-8")
                copied.append("bot-handoff.json")
            else:
                skipped.append("bot-handoff.json")

    if empresa == "unlockers":
        meta_ads = dest_dir / "18-meta-ads-inbound.md"
        rag_meta = ROOT / _RAG_DIR / "18-meta-ads-inbound.md"
        if meta_ads.is_file() and not rag_meta.is_file():
            rag_meta.write_text(meta_ads.read_text(encoding="utf-8"), encoding="utf-8")
            skipped.append("18-meta-ads-inbound.md (mirrored to RAG)")

    write_manifest(empresa)
    return {
        "dest": str(dest_dir.relative_to(ROOT)),
        "copied": copied,
        "unchanged": skipped,
        "files_dest": sorted(p.name for p in dest_dir.glob("*.md")),
    }


def bot_kb_relative_files(empresa: str = "unlockers") -> list[str]:
    dest = katana_mirror_dir(empresa)
    return sorted(
        f"storage/app/bot_kb/{empresa}/{p.name}"
        for p in dest.glob("*.md")
    )


def push_turns(
    turns: list[dict[str, Any]],
    *,
    company: str = "unlockers",
) -> dict[str, Any]:
    """POST turns to Katana canonical log."""
    if not turns:
        return {"success": True, "accepted": 0}
    payload = {"company": company, "turns": turns}
    return _request("POST", "turns", payload=payload)


_KATANA_TURN_COMPANIES = frozenset({"unlockers", "paolapalacio"})


def _push_cursor_path(empresa: str) -> Path:
    if empresa == "paolapalacio":
        return ROOT / "data/rag/paolapalacio-wa-observacion/.katana_push_cursor"
    return ROOT / _STATE_DIR / ".katana_push_cursor"


def push_turn_from_mac(
    *,
    contact: str,
    role: str,
    content: str,
    channel: str = "whatsapp",
    company_alias: str = "unlockers",
    via: str = "mac",
    kind: str = "text",
    llm_engine: str = "",
    llm_model: str = "",
) -> None:
    """Fire-and-forget single turn (best effort; never raises to engine)."""
    body = (content or "").strip()
    if not body or company_alias not in _KATANA_TURN_COMPANIES:
        return
    turn: dict[str, Any] = {
        "contact": "".join(ch for ch in contact if ch.isdigit()),
        "role": role if role in ("user", "assistant") else "assistant",
        "content": body[:4000],
        "channel": channel,
        "via": via,
        "kind": kind or "text",
        "at": datetime.now(timezone.utc).isoformat(),
    }
    if str(llm_engine or "").strip():
        turn["llm_engine"] = str(llm_engine).strip()
    if str(llm_model or "").strip():
        turn["llm_model"] = str(llm_model).strip()
    try:
        push_turns([turn], company=company_alias)
    except Exception:
        logger.exception("push_turn_from_mac failed contact=%s empresa=%s", contact, company_alias)


def pull_learning(
    empresa: str = "unlockers",
    *,
    since: str | None = None,
    ingest: bool = True,
    write_digest: bool = True,
) -> dict[str, Any]:
    """
    Pull new turns from Katana, merge into local message_log + WA-style exports,
    optionally ingest unlockers_conv and write digest markdown.
    """
    company = get_company(empresa)
    cursor_path = ROOT / _LEARNING_CURSOR
    since_iso = since or ""
    if not since_iso and cursor_path.is_file():
        since_iso = cursor_path.read_text(encoding="utf-8").strip()

    query: dict[str, str] = {"company": empresa}
    if since_iso:
        query["since"] = since_iso

    data = _request("GET", "turns", query=query)
    turns = list(data.get("turns") or [])
    newest = since_iso

    from cloudy.bot import store

    export_dir = ROOT / _LEARNING_EXPORTS
    export_dir.mkdir(parents=True, exist_ok=True)
    # Group by contact for export files (WhatsApp-like for ingest_conversations).
    by_contact: dict[str, list[dict[str, Any]]] = {}
    applied = 0
    for turn in turns:
        contact = "".join(ch for ch in str(turn.get("contact") or "") if ch.isdigit())
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").strip()
        at = str(turn.get("at") or datetime.now(timezone.utc).isoformat())
        via = str(turn.get("via") or "openai")
        if not contact or not content or role not in ("user", "assistant"):
            continue
        # Placeholders del log Katana (sin adjunto real) no deben re-ingresarse al Mac.
        if _is_katana_media_placeholder(content):
            if at > (newest or ""):
                newest = at
            continue
        # Prefer learning from OpenAI contingency (Mac already has its own log).
        if via == "mac":
            # Still update cursor; skip duplicate local write for mac-originated.
            if at > (newest or ""):
                newest = at
            continue
        store.log_turn(empresa, contact, role, content, kind=str(turn.get("kind") or "text"))
        by_contact.setdefault(contact, []).append(turn)
        applied += 1
        if at > (newest or ""):
            newest = at

    owner = str(company.get("owner_name") or "Sergio")
    files_written = 0
    conv_rel = str(company.get("conversations_dir") or "data/rag/conversaciones")
    conv_root = ROOT / conv_rel
    conv_root.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    for contact, rows in by_contact.items():
        path = export_dir / f"katana_{contact}.txt"
        conv_path = conv_root / f"katana_{contact}.txt"
        lines: list[str] = []
        for row in rows:
            role = str(row.get("role"))
            content = str(row.get("content") or "").replace("\n", " ")
            at = str(row.get("at") or "")
            stamp = _wa_stamp(at)
            sender = owner if role == "assistant" else "Cliente"
            lines.append(f"{stamp} - {sender}: {content}")
        if lines:
            block = "\n".join(lines) + "\n"
            for target in (path, conv_path):
                prev = target.read_text(encoding="utf-8") if target.is_file() else ""
                target.write_text((prev + ("\n" if prev else "") + block), encoding="utf-8")
            files_written += 1

    digest_path = None
    if write_digest and by_contact:
        digest_path = _write_digest(by_contact, owner)

    ingest_result = None
    if ingest and (files_written > 0 or applied > 0):
        from cloudy.bot.rag import ingest_conversations, ingest_knowledge

        ingest_result = ingest_conversations(empresa, company)
        if digest_path:
            ingest_knowledge(empresa, company)

    if newest:
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        cursor_path.write_text(newest, encoding="utf-8")

    return {
        "fetched": len(turns),
        "applied": applied,
        "export_files": files_written,
        "cursor": newest,
        "digest": str(digest_path.relative_to(ROOT)) if digest_path else None,
        "ingest": ingest_result,
    }


def push_recent_local_turns(
    empresa: str = "unlockers",
    *,
    since: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Push recent Mac message_log turns that are not yet on Katana (best-effort batch)."""
    from cloudy.bot import store

    cursor_path = _push_cursor_path(empresa)
    since_iso = since or ""
    if not since_iso and cursor_path.is_file():
        since_iso = cursor_path.read_text(encoding="utf-8").strip()

    with store.connect() as conn:
        if since_iso:
            rows = conn.execute(
                "SELECT company, contact, role, kind, content, created_at FROM message_log"
                " WHERE company=? AND created_at > ? ORDER BY id ASC LIMIT ?",
                (empresa, since_iso, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT company, contact, role, kind, content, created_at FROM message_log"
                " WHERE company=? ORDER BY id DESC LIMIT ?",
                (empresa, limit),
            ).fetchall()
            rows = list(reversed(rows))

    turns = []
    newest = since_iso
    for row in rows:
        turns.append(
            {
                "contact": row["contact"],
                "role": row["role"],
                "content": row["content"],
                "channel": "whatsapp",
                "via": "mac",
                "kind": row["kind"] or "text",
                "at": row["created_at"],
            }
        )
        if row["created_at"] > (newest or ""):
            newest = row["created_at"]

    result = push_turns(turns, company=empresa) if turns else {"success": True, "accepted": 0}
    if newest:
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        cursor_path.write_text(newest, encoding="utf-8")
    return {"pushed": len(turns), "cursor": newest, "api": result}


def push_kb_remote(empresa: str = "unlockers", *, only_changed: bool | None = None) -> dict[str, Any]:
    """
    Push local bot_kb markdown to Katana via M2M API (no FTP).
    Call after sync_kb_local so production contingency reads the same KB.
    """
    local = sync_kb_local(empresa)
    dest_dir = katana_mirror_dir(empresa)
    files: dict[str, str] = {}
    changed = set(local.get("copied") or [])
    for path in sorted(dest_dir.glob("*.md")):
        if only_changed and path.name not in changed and changed:
            # If there were changes, only upload those; if none, skip upload.
            continue
        files[path.name] = path.read_text(encoding="utf-8")
    if only_changed and not changed:
        return {"success": True, "written": [], "unchanged": local.get("unchanged") or [], "skipped_upload": True}
    if not files:
        # Full push when only_changed=False or first sync
        for path in sorted(dest_dir.glob("*.md")):
            files[path.name] = path.read_text(encoding="utf-8")
    handoff = dest_dir / "bot-handoff.json"
    if handoff.is_file():
        files["bot-handoff.json"] = handoff.read_text(encoding="utf-8")
    result = _request("POST", "kb", payload={"company": empresa, "files": files})
    result["local"] = local
    return result


def run_bidirectional_sync(
    empresa: str = "unlockers",
    *,
    ingest: bool = True,
    push_kb: bool = True,
) -> dict[str, Any]:
    """
    Full automatic cycle while Mac is on:
      1) pull OpenAI/Katana learning → Mac
      2) push recent Mac turns → Katana
      3) sync RAG → bot_kb local + push KB API → production
    """
    out: dict[str, Any] = {"empresa": empresa}
    out["pull"] = pull_learning(empresa, ingest=ingest, write_digest=True)
    out["push_turns"] = push_recent_local_turns(empresa, limit=200)
    if push_kb:
        # Upload all md when digest or rag changed; otherwise only changed files.
        changed = bool((out["pull"] or {}).get("digest")) or bool((out["pull"] or {}).get("applied"))
        out["push_kb"] = push_kb_remote(empresa, only_changed=not changed)
    from cloudy.bot.wa_clients import sync_wa_clients_local

    out["sync_wa_clients"] = sync_wa_clients_local()
    return out


_EDGE_DEPLOY_SCRIPT = Path("scripts/maintenance/deploy_edge_srv01.sh")


def _load_sync_state(empresa: str = "unlockers") -> dict[str, Any]:
    path = sync_state_path(empresa)
    if not path.is_file() and empresa == "unlockers":
        legacy = ROOT / "data/rag/unlockers/.bot_sync_state.json"
        if legacy.is_file():
            path = legacy
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _save_sync_state(state: dict[str, Any], empresa: str = "unlockers") -> None:
    path = sync_state_path(empresa)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def deploy_katana_files(*, force: bool = False) -> dict[str, Any]:
    """FTP deploy bot config files when content hash changed."""
    import hashlib
    import subprocess

    from cloudy.bot.wa_clients import sync_wa_clients_local

    sync_wa_clients_local()
    php_rel = Path("clients/1lockers/sites/web/config/bot_wa_clients.php")
    php_path = ROOT / php_rel
    if not php_path.is_file():
        return {"skipped": True, "reason": "bot_wa_clients.php missing"}

    digest = hashlib.sha256(php_path.read_bytes()).hexdigest()[:16]
    state = _load_sync_state("unlockers")
    if not force and state.get("katana_wa_clients_hash") == digest:
        return {"skipped": True, "reason": "unchanged", "hash": digest}

    py = ROOT / ".venv" / "bin" / "python"
    cmd = [
        str(py),
        str(ROOT / "scripts" / "cloudy.py"),
        "deploy",
        "1lockers",
        "web",
        "--files",
        "config/bot_wa_clients.php",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    ok = proc.returncode == 0
    if ok:
        state["katana_wa_clients_hash"] = digest
        _save_sync_state(state, "unlockers")
    return {
        "success": ok,
        "hash": digest,
        "stdout": (proc.stdout or "")[-500:],
        "stderr": (proc.stderr or "")[-500:] if not ok else "",
    }


def deploy_edge_if_needed(*, force: bool = False, empresa: str = "unlockers") -> dict[str, Any]:
    """Run deploy_edge_srv01.sh when bundle hash changed."""
    import subprocess

    from cloudy.bot.prompts import prompts_bundle_hash, sync_handoff_local

    sync_handoff_local(empresa)
    bundle = prompts_bundle_hash(empresa)
    rag_sig = bundle_content_hash(empresa)
    combined = f"{empresa}:{bundle}:{rag_sig}"
    state = _load_sync_state(empresa)
    if not force and state.get("edge_bundle_hash") == combined:
        return {"skipped": True, "reason": "unchanged", "hash": combined, "empresa": empresa}

    script = ROOT / _EDGE_DEPLOY_SCRIPT
    if not script.is_file():
        return {"skipped": True, "reason": "deploy script missing"}

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    ok = proc.returncode == 0
    if ok:
        state["edge_bundle_hash"] = combined
        _save_sync_state(state, empresa)
    return {
        "success": ok,
        "hash": combined,
        "empresa": empresa,
        "stdout": (proc.stdout or "")[-800:],
        "stderr": (proc.stderr or "")[-800:] if not ok else "",
    }


def run_sync_check(empresa: str = "unlockers", *, all_enabled: bool = False) -> dict[str, Any]:
    """Health + hash diagnostics across Mac, Edge, Katana (per tenant or all)."""
    import urllib.request

    from cloudy.bot.company_paths import enabled_company_aliases, load_manifest
    from cloudy.bot.prompts import prompts_bundle_hash

    if all_enabled:
        tenants: dict[str, Any] = {}
        overall_ok = True
        for alias in enabled_company_aliases():
            one = run_sync_check(alias, all_enabled=False)
            tenants[alias] = one
            if not one.get("ok"):
                overall_ok = False
        return {"ok": overall_ok, "tenants": tenants}

    checks: dict[str, Any] = {"ok": True, "empresa": empresa, "checks": {}}

    def _health(url: str, timeout: float = 8.0, headers: dict[str, str] | None = None) -> bool:
        try:
            hdrs = {"User-Agent": "CloudyBot-SyncCheck/1.0"}
            if headers:
                hdrs.update(headers)
            req = urllib.request.Request(url, method="GET", headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read(400).decode("utf-8", errors="replace")
                if resp.status != 200:
                    return False
                low = body.lower()
                if "ok" in low:
                    return True
                try:
                    data = json.loads(body)
                    return bool(data.get("ok") or data.get("success"))
                except json.JSONDecodeError:
                    return '"success":true' in low.replace(" ", "")
        except Exception as exc:
            checks["checks"][url] = {"ok": False, "error": str(exc)}
            checks["ok"] = False
            return False

    mac_ok = _health("http://127.0.0.1:8080/health")
    if mac_ok:
        checks["checks"]["mac"] = {"ok": True}
    edge_ok = _health("https://edge.1lockers.net/health", timeout=12.0)
    if edge_ok:
        checks["checks"]["edge"] = {"ok": True}

    cfg = _katana_cfg()
    katana_ok = False
    if cfg.get("api_key"):
        try:
            data = _request(
                "GET",
                "client-profile",
                query={"phone": "573005059143"},
            )
            status = int(data.get("_http_status") or 0)
            if status == 401:
                raise RuntimeError("HTTP Error 401: Unauthorized")
            katana_ok = bool(data.get("success"))
            checks["checks"]["katana_client_profile"] = {
                "ok": katana_ok,
                "recognized": bool(data.get("recognized")),
                "http_status": status,
            }
        except Exception as exc:
            checks["checks"]["katana_client_profile"] = {"ok": False, "error": str(exc)}
            checks["ok"] = False
    else:
        checks["checks"]["katana_client_profile"] = {"ok": False, "error": "no api_key"}

    local_kb = len(_kb_markdown_sources(empresa))
    mirror_kb = len(list(katana_mirror_dir(empresa).glob("*.md")))
    checks["checks"]["kb_counts"] = {
        "ok": mirror_kb >= local_kb or local_kb == 0,
        "local_rag": local_kb,
        "bot_kb_mirror": mirror_kb,
    }
    if mirror_kb < local_kb:
        checks["ok"] = False

    manifest = load_manifest(empresa)
    prompt_hash = prompts_bundle_hash(empresa)
    checks["checks"]["prompts_bundle_hash"] = {"hash": prompt_hash, "empresa": empresa}
    checks["checks"]["manifest"] = {
        "bundle_hash": manifest.get("bundle_hash"),
        "persona_contacts": len(manifest.get("persona_contacts") or []),
        "alias_kb": len(manifest.get("alias_kb") or []),
    }
    checks["checks"]["mac_health"] = {"ok": mac_ok}
    checks["checks"]["edge_health"] = {"ok": edge_ok}
    if not mac_ok:
        checks["ok"] = False
    if not edge_ok:
        checks["ok"] = False
    if not katana_ok:
        checks["ok"] = False

    return checks


_SYNC_AUDIT_DIR = Path("data/reports/bot-sync")
_SYNC_LOCK_PATH = _SYNC_AUDIT_DIR / ".sync.lock"
_SYNC_LOCK_STALE_SECONDS = 45 * 60
_SYNC_MAX_RETRIES = 3
_SYNC_RETRY_BACKOFF_SECONDS = (15, 30, 60)


def _safe_step(name: str, fn: Any) -> dict[str, Any]:
    """Run a sync step; never abort the full cycle on a single failure."""
    try:
        result = fn()
        if isinstance(result, dict):
            return result
        return {"success": True, "result": result}
    except Exception as exc:
        logger.exception("sync step %s failed", name)
        return {"success": False, "error": str(exc), "step": name}


def evaluate_sync_result(
    result: dict[str, Any],
    *,
    require_mac_health: bool = False,
    require_edge_health: bool = True,
) -> tuple[bool, list[str]]:
    """
    Strict validation for scheduled / hourly sync.
    Returns (ok, list of human-readable error labels).
    """
    errors: list[str] = []

    handoff = result.get("handoff") or {}
    if not handoff.get("dest"):
        errors.append("handoff")

    swc = result.get("sync_wa_clients") or {}
    if swc.get("success") is False or not swc.get("dest_php"):
        errors.append("sync_wa_clients")

    pull = result.get("pull") or {}
    if pull.get("success") is False:
        errors.append("pull_learning")

    push_turns = result.get("push_turns") or {}
    api = push_turns.get("api") or {}
    if api.get("success") is False:
        errors.append("push_turns")

    push_kb = result.get("push_kb")
    if push_kb is not None and push_kb is not False:
        if push_kb.get("success") is False:
            errors.append("push_kb")
        elif not push_kb.get("skipped_upload") and not (push_kb.get("written") or push_kb.get("files")):
            # API accepted but wrote nothing when upload was expected
            if push_kb.get("error"):
                errors.append("push_kb")

    deploy_k = result.get("deploy_katana") or {}
    if deploy_k and not deploy_k.get("skipped") and deploy_k.get("success") is False:
        errors.append("deploy_katana")

    deploy_e = result.get("deploy_edge") or {}
    if deploy_e and not deploy_e.get("skipped") and deploy_e.get("success") is False:
        errors.append("deploy_edge")

    check = result.get("sync_check") or {}
    if check.get("error"):
        errors.append("sync_check")
    else:
        checks = check.get("checks") or {}
        katana = checks.get("katana_client_profile") or {}
        if katana.get("ok") is False:
            errors.append("katana_client_profile")
        kb_counts = checks.get("kb_counts") or {}
        if kb_counts.get("ok") is False:
            errors.append("kb_counts")
        if require_mac_health and not (checks.get("mac_health") or {}).get("ok"):
            errors.append("mac_health")
        if require_edge_health and not (checks.get("edge_health") or {}).get("ok"):
            errors.append("edge_health")

    return (len(errors) == 0, errors)


@contextmanager
def _sync_file_lock() -> Iterator[bool]:
    """Cross-process lock so startup sync and hourly launchd never overlap."""
    _SYNC_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    lock = ROOT / _SYNC_LOCK_PATH
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < _SYNC_LOCK_STALE_SECONDS:
            yield False
            return
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield False
        return
    os.write(fd, f"{os.getpid()}@{datetime.now(timezone.utc).isoformat()}".encode("ascii"))
    os.close(fd)
    try:
        yield True
    finally:
        lock.unlink(missing_ok=True)


def append_sync_audit(
    *,
    trigger: str,
    ok: bool,
    errors: list[str],
    attempt: int,
    result: dict[str, Any] | None = None,
) -> None:
    """Append one line to data/reports/bot-sync/launchd.log (UTF-8)."""
    log_dir = ROOT / _SYNC_AUDIT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "launchd.log"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "OK" if ok else "FAIL"
    err_txt = ",".join(errors) if errors else "-"
    swc = (result or {}).get("sync_wa_clients") or {}
    push_kb = (result or {}).get("push_kb") or {}
    line = (
        f"[{stamp}] trigger={trigger} attempt={attempt} status={status} "
        f"errors={err_txt} contacts={swc.get('contacts', '?')} "
        f"kb_written={len(push_kb.get('written') or [])} "
        f"katana_deploy={(result or {}).get('deploy_katana', {}).get('success', 'skip')} "
        f"edge_deploy={(result or {}).get('deploy_edge', {}).get('success', 'skip')}\n"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def run_scheduled_bot_sync(
    empresa: str = "unlockers",
    *,
    trigger: str = "manual",
    ingest: bool = True,
    push_kb: bool = True,
    deploy_katana: bool = True,
    deploy_edge: bool = True,
    force_deploy: bool = False,
    max_retries: int = _SYNC_MAX_RETRIES,
    require_mac_health: bool = False,
    require_edge_health: bool = True,
) -> dict[str, Any]:
    """
    Hourly / launchd sync with file lock, retries and strict validation.
    Raises RuntimeError when all attempts fail validation.
    """
    with _sync_file_lock() as acquired:
        if not acquired:
            skipped = {
                "skipped": True,
                "reason": "lock_busy",
                "trigger": trigger,
                "success": True,
            }
            append_sync_audit(
                trigger=trigger, ok=True, errors=["lock_busy"], attempt=0, result=skipped
            )
            return skipped

        last_result: dict[str, Any] = {}
        last_errors: list[str] = []
        attempts = max(1, int(max_retries))

        for attempt in range(1, attempts + 1):
            last_result = run_full_bot_sync(
                empresa,
                ingest=ingest,
                push_kb=push_kb,
                deploy_katana=deploy_katana,
                deploy_edge=deploy_edge,
                force_deploy=force_deploy,
            )
            ok, last_errors = evaluate_sync_result(
                last_result,
                require_mac_health=require_mac_health,
                require_edge_health=require_edge_health,
            )
            last_result["success"] = ok
            last_result["errors"] = last_errors
            last_result["attempt"] = attempt
            last_result["trigger"] = trigger
            append_sync_audit(
                trigger=trigger,
                ok=ok,
                errors=last_errors,
                attempt=attempt,
                result=last_result,
            )
            if ok:
                return last_result
            if attempt < attempts:
                backoff = _SYNC_RETRY_BACKOFF_SECONDS[min(attempt - 1, len(_SYNC_RETRY_BACKOFF_SECONDS) - 1)]
                logger.warning(
                    "scheduled sync attempt %s/%s failed (%s); retry in %ss",
                    attempt,
                    attempts,
                    ",".join(last_errors),
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"bot sync failed after {attempts} attempts: {', '.join(last_errors)}"
        )


def push_cursor_cloud_mirror() -> dict[str, Any]:
    """Best-effort push of prompts/KB mirror for Cursor Cloud Agents."""
    script = ROOT / "scripts" / "maintenance" / "push_cursor_cloud_mirror.sh"
    cfg = ROOT / "config" / "cursor-cloud.json"
    if not cfg.is_file():
        return {"success": True, "skipped": True, "reason": "no cursor-cloud.json"}
    if not script.is_file():
        return {"success": False, "error": f"missing {script}"}
    import subprocess

    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=180,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return {"success": False, "exit": proc.returncode, "stdout": out[-500:], "stderr": err[-500:]}
    return {"success": True, "output": out[-500:]}


def run_full_bot_sync(
    empresa: str = "unlockers",
    *,
    ingest: bool = True,
    push_kb: bool = True,
    deploy_katana: bool = True,
    deploy_edge: bool = True,
    force_deploy: bool = False,
) -> dict[str, Any]:
    """
    Full sync: Katana API + FTP + Edge VPS.
    Mac → Edge → Katana contingency stay aligned on KB, prompts, clients, handoff.
    """
    from cloudy.bot.prompts import sync_handoff_local
    from cloudy.bot.wa_clients import sync_wa_clients_local

    out: dict[str, Any] = {"empresa": empresa, "started_at": datetime.now(timezone.utc).isoformat()}
    out["handoff"] = _safe_step("handoff", sync_handoff_local)
    out["pull"] = _safe_step(
        "pull_learning",
        lambda: pull_learning(empresa, ingest=ingest, write_digest=True),
    )
    out["push_turns"] = _safe_step(
        "push_turns",
        lambda: push_recent_local_turns(empresa, limit=200),
    )
    out["push_leads"] = _safe_step(
        "push_leads_retry",
        lambda: __import__("cloudy.bot.katana_leads", fromlist=["retry_pending_crm"]).retry_pending_crm(empresa, limit=30),
    )
    if push_kb:
        pull_data = out["pull"] if out["pull"].get("success") is not False else {}
        changed = bool(pull_data.get("digest")) or bool(pull_data.get("applied"))

        def _push_kb() -> dict[str, Any]:
            return push_kb_remote(empresa, only_changed=not changed)

        out["push_kb"] = _safe_step("push_kb", _push_kb)
    out["cursor_mirror"] = _safe_step("cursor_mirror", push_cursor_cloud_mirror)
    out["sync_wa_clients"] = _safe_step("sync_wa_clients", sync_wa_clients_local)
    if deploy_katana:
        try:
            out["deploy_katana"] = deploy_katana_files(force=force_deploy)
        except Exception as exc:
            logger.exception("deploy_katana failed")
            out["deploy_katana"] = {"success": False, "error": str(exc)}
    if deploy_edge:
        try:
            out["deploy_edge"] = deploy_edge_if_needed(force=force_deploy, empresa=empresa)
        except Exception as exc:
            logger.exception("deploy_edge failed")
            out["deploy_edge"] = {"success": False, "error": str(exc)}
    try:
        out["sync_check"] = run_sync_check(empresa)
    except Exception as exc:
        out["sync_check"] = {"ok": False, "error": str(exc)}
    return out


def _is_katana_media_placeholder(text: str) -> bool:
    """Skip Katana log lines like ``[Media audio]`` with no real transcription."""
    import re

    t = (text or "").strip()
    if not t:
        return True
    return bool(re.match(r"^\[Media\s+\w+\]$", t, re.IGNORECASE))


def _wa_stamp(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        local = dt.astimezone()
        return f"{local.day}/{local.month}/{str(local.year)[2:]}, {local.hour}:{local.minute:02d}"
    except Exception:
        return "1/1/26, 12:00"


def _write_digest(by_contact: dict[str, list[dict[str, Any]]], owner: str) -> Path:
    """Anonymized learning digest (no phone numbers) for both RAG and bot_kb."""
    path = ROOT / _RAG_DIR / _DIGEST_NAME
    lines = [
        "# Aprendizaje reciente (sync Katana ↔ Mac)",
        "",
        "Hechos y tono capturados mientras el bot corría en contingencia OpenAI.",
        "Sin números de teléfono. Úsalo solo como contexto de estilo/casos.",
        "",
    ]
    pair_q: list[str] = []
    for _contact, rows in list(by_contact.items())[:40]:
        for row in rows:
            role = str(row.get("role"))
            content = str(row.get("content") or "").strip()[:400]
            if role == "user":
                pair_q.append(content)
            elif role == "assistant" and pair_q:
                q = pair_q.pop(0)
                lines.append(f"## Caso")
                lines.append(f"- Cliente: {q}")
                lines.append(f"- {owner}/Cloudy: {content}")
                lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    # Mirror into bot_kb local copy
    dest = katana_mirror_dir("unlockers") / _DIGEST_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path
