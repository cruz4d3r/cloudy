# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Cloudy Bot Autofix — lectura de logs del bot Mac + remedios seguros.

A diferencia de ``heartbeat`` (solo consulta y avisa), autofix aplica
correcciones acotadas al **runtime local** del bot (launchd, espejo repo,
Node empaquetado, secretos M2M). Nunca SSH a producción cPanel/OVH.

Cada regla tiene cooldown para evitar bucles. Se invoca desde
``hourly-bot-sync.sh`` (launchd cada hora) o manualmente:

    .venv/bin/python scripts/cloudy.py bot autofix run
    .venv/bin/python scripts/cloudy.py bot autofix status
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cloudy.paths import CONFIG_DIR, DATA_DIR, ROOT, bundled_node_binary

logger = logging.getLogger("cloudy.bot.autofix")

CONFIG_PATH = CONFIG_DIR / "bot-autofix.json"
STATE_DIR = DATA_DIR / "bot-autofix"
STATE_PATH = STATE_DIR / "state.json"
REPORT_DIR = DATA_DIR / "reports" / "bot-autofix"

BOT_APP = Path.home() / "Library" / "Application Support" / "CloudyBot"
RUNTIME = BOT_APP / "runtime"
LOGS = BOT_APP / "logs"
LAUNCHD_LABEL = "net.1lockers.cloudy-bot"
LAUNCHD_TUNNEL_LABEL = "net.1lockers.cloudflared"
PUBLIC_HEALTH_URL = "https://bot.1lockers.net/health"
SYNC_LOCK_DIR = BOT_APP / "bot-sync.lockdir"
SYNC_LOCK_STALE_SEC = 2700

_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "tail_bytes": 300_000,
    "default_cooldown_min": 30,
    "notify_on_fix": True,
    "notify_on_failure": True,
    "notify_on_manual": True,
    "owner_company": "unlockers",
    "public_health_url": PUBLIC_HEALTH_URL,
    "log_sources": [
        "cloudy-bot.log",
        "bot-autosync.log",
        "bot-autofix.log",
        "cloudflared.log",
    ],
}

_SECRET_FILES = (
    "edge-api-token.txt",
    "media-api-token.txt",
    "mac-proxy-token.txt",
    "bot-m2m-hmac-secret.txt",
)


@dataclass
class AutofixRule:
    """Patrón de error → acción remediativa."""

    rule_id: str
    patterns: list[re.Pattern[str]]
    action: str
    description: str
    cooldown_min: int = 30
    severity: str = "warn"


@dataclass
class MatchResult:
    rule_id: str
    action: str
    description: str
    severity: str
    sample: str


@dataclass
class AutofixReport:
    ran_at: str
    trigger: str
    dry_run: bool
    matches: list[MatchResult] = field(default_factory=list)
    applied: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    log_bytes_scanned: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config() -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    if CONFIG_PATH.is_file():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg.update(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("bot-autofix config inválido: %s", exc)
    return cfg


def _load_state() -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.is_file():
        return {"fixes": {}, "history": []}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"fixes": {}, "history": []}


def _save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _compile_rules(cfg: dict[str, Any]) -> list[AutofixRule]:
    raw_rules = cfg.get("rules")
    if isinstance(raw_rules, list) and raw_rules:
        out: list[AutofixRule] = []
        for item in raw_rules:
            if not isinstance(item, dict):
                continue
            pats = item.get("patterns") or []
            if not pats:
                continue
            out.append(
                AutofixRule(
                    rule_id=str(item.get("id") or "custom"),
                    patterns=[re.compile(p, re.IGNORECASE) for p in pats],
                    action=str(item.get("action") or ""),
                    description=str(item.get("description") or item.get("id") or ""),
                    cooldown_min=int(item.get("cooldown_min") or cfg.get("default_cooldown_min", 30)),
                    severity=str(item.get("severity") or "warn"),
                )
            )
        return out
    return _builtin_rules(cfg)


def _builtin_rules(cfg: dict[str, Any]) -> list[AutofixRule]:
    cd = int(cfg.get("default_cooldown_min", 30))
    return [
        AutofixRule(
            rule_id="cursor_node_missing",
            patterns=[
                re.compile(r"FileNotFoundError.*['\"]node['\"]", re.I),
                re.compile(r"No such file or directory: ['\"]node['\"]", re.I),
                re.compile(r"spawn node ENOENT", re.I),
            ],
            action="sync_bundled_node",
            description="Node empaquetado ausente en runtime (Cursor SDK)",
            cooldown_min=cd,
            severity="alert",
        ),
        AutofixRule(
            rule_id="bot_health_down",
            patterns=[
                re.compile(r"Connection refused.*8080", re.I),
                re.compile(r"urlopen error \[Errno 61\]", re.I),
                re.compile(r"local FAIL", re.I),
                re.compile(r"/health.*FAIL", re.I),
            ],
            action="kickstart_bot",
            description="Bot local no responde en :8080",
            cooldown_min=15,
            severity="alert",
        ),
        AutofixRule(
            rule_id="gather_flush_failed",
            patterns=[re.compile(r"gather flush falló", re.I)],
            action="kickstart_bot",
            description="Fallo en flush de gather (reinicio suave launchd)",
            cooldown_min=20,
            severity="warn",
        ),
        AutofixRule(
            rule_id="chromadb_duplicate_id",
            patterns=[
                re.compile(r"DuplicateIDError", re.I),
                re.compile(r"DuplicateID", re.I),
            ],
            action="mirror_repo_runtime",
            description="Chroma duplicate ID — espejar código RAG corregido al runtime",
            cooldown_min=60,
            severity="warn",
        ),
        AutofixRule(
            rule_id="katana_unauthorized",
            patterns=[
                re.compile(r"HTTP Error 401.*Unauthorized", re.I),
                re.compile(r"katana_client_profile.*401", re.I),
                re.compile(r"sync-check.*ok.:\s*false", re.I),
            ],
            action="sync_secrets_runtime",
            description="Secretos M2M/tokens desactualizados en runtime",
            cooldown_min=45,
            severity="warn",
        ),
        AutofixRule(
            rule_id="bot_sync_failed",
            patterns=[
                re.compile(r"FAIL sync rc=", re.I),
                re.compile(r"bot sync failed", re.I),
            ],
            action="mirror_repo_runtime",
            description="Sync horario falló — refrescar cloudy/ en runtime",
            cooldown_min=45,
            severity="warn",
        ),
        AutofixRule(
            rule_id="runtime_venv_missing",
            patterns=[re.compile(r"runtime venv missing", re.I)],
            action="notify_manual",
            description="Falta venv en runtime — requiere install-autostart.sh",
            cooldown_min=360,
            severity="alert",
        ),
        AutofixRule(
            rule_id="cloudflared_tunnel_glitch",
            patterns=[
                re.compile(r"Serve tunnel error", re.I),
                re.compile(r"Connection terminated", re.I),
                re.compile(r"failed to serve tunnel connection", re.I),
                re.compile(r"Failed to refresh DNS local resolver", re.I),
            ],
            action="kickstart_cloudflared",
            description="Inestabilidad en túnel cloudflared (bot.1lockers.net)",
            cooldown_min=20,
            severity="warn",
        ),
        AutofixRule(
            rule_id="tunnel_public_down",
            patterns=[re.compile(r"__proactive_tunnel__")],  # solo vía health check activo
            action="kickstart_cloudflared",
            description="bot.1lockers.net no responde con bot local sano",
            cooldown_min=15,
            severity="alert",
        ),
        AutofixRule(
            rule_id="llm_all_models_failed",
            patterns=[
                re.compile(r"All models failed", re.I),
                re.compile(r"LLMError.*All models", re.I),
            ],
            action="remediate_llm_chain",
            description="Cadena LLM agotada — Node, secretos y reinicio bot",
            cooldown_min=25,
            severity="alert",
        ),
        AutofixRule(
            rule_id="sync_lock_stale",
            patterns=[
                re.compile(r"skip \(lock busy\)", re.I),
            ],
            action="clear_sync_lock",
            description="Lock de sync horario atascado — liberar si es viejo",
            cooldown_min=60,
            severity="warn",
        ),
        AutofixRule(
            rule_id="edge_health_down",
            patterns=[
                re.compile(r"edge_health.*ok.:\s*false", re.I),
                re.compile(r"errors.*edge_health", re.I),
            ],
            action="notify_manual",
            description="Edge VPS caído — revisar edge.1lockers.net (no auto-fix desde Mac)",
            cooldown_min=120,
            severity="alert",
        ),
        AutofixRule(
            rule_id="whatsapp_token_error",
            patterns=[
                re.compile(r"OAuthException", re.I),
                re.compile(r"Error validating access token", re.I),
                re.compile(r"Session has expired", re.I),
            ],
            action="notify_manual",
            description="Token WhatsApp Meta inválido — renovar en Business Manager",
            cooldown_min=360,
            severity="alert",
        ),
    ]


def _log_paths(cfg: dict[str, Any]) -> list[Path]:
    names = cfg.get("log_sources") or _DEFAULTS["log_sources"]
    paths: list[Path] = []
    for name in names:
        p = LOGS / str(name)
        if p.is_file():
            paths.append(p)
    repo_launchd = DATA_DIR / "reports" / "bot-sync" / "launchd.log"
    if repo_launchd.is_file():
        paths.append(repo_launchd)
    return paths


def _tail_text(path: Path, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
        if size <= max_bytes:
            return path.read_text(encoding="utf-8", errors="replace")
        with path.open("rb") as fh:
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError as exc:
        logger.warning("No se pudo leer %s: %s", path, exc)
        return ""


def scan_logs(cfg: dict[str, Any] | None = None) -> tuple[str, list[MatchResult]]:
    cfg = cfg or load_config()
    tail_bytes = int(cfg.get("tail_bytes", 300_000))
    rules = _compile_rules(cfg)
    combined_parts: list[str] = []
    total_bytes = 0
    for path in _log_paths(cfg):
        chunk = _tail_text(path, tail_bytes)
        total_bytes += len(chunk.encode("utf-8", errors="replace"))
        combined_parts.append(f"\n--- {path.name} ---\n{chunk}")
    haystack = "\n".join(combined_parts)

    matches: list[MatchResult] = []
    seen_rules: set[str] = set()
    for rule in rules:
        if rule.rule_id in seen_rules:
            continue
        for pat in rule.patterns:
            m = pat.search(haystack)
            if m:
                start = max(0, m.start() - 80)
                end = min(len(haystack), m.end() + 120)
                sample = haystack[start:end].replace("\n", " ").strip()[:240]
                matches.append(
                    MatchResult(
                        rule_id=rule.rule_id,
                        action=rule.action,
                        description=rule.description,
                        severity=rule.severity,
                        sample=sample,
                    )
                )
                seen_rules.add(rule.rule_id)
                break
    return haystack, matches


def _cooldown_ok(state: dict[str, Any], rule_id: str, cooldown_min: int) -> bool:
    fixes = state.get("fixes") or {}
    entry = fixes.get(rule_id) or {}
    last = entry.get("last_applied")
    if not last:
        return True
    try:
        then = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - then).total_seconds() / 60.0
        return elapsed >= cooldown_min
    except (ValueError, TypeError):
        return True


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except OSError as exc:
        return 1, str(exc)


def _kickstart_bot() -> tuple[bool, str]:
    ok, detail = _kickstart_launchd(LAUNCHD_LABEL, _plist_path())
    time.sleep(2)
    health_rc, health_out = _run(
        ["curl", "-s", "-m", "5", "http://127.0.0.1:8080/health"],
        timeout=10,
    )
    health_ok = health_rc == 0 and "ok" in health_out.lower()
    detail = f"{detail}; health rc={health_rc} body={health_out[:120]}"
    return health_ok, detail


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _tunnel_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_TUNNEL_LABEL}.plist"


def _kickstart_launchd(label: str, plist: Path) -> tuple[bool, str]:
    uid = os.getuid()
    target = f"gui/{uid}/{label}"
    rc, out = _run(["launchctl", "kickstart", "-k", target], timeout=30)
    if rc != 0:
        rc2, out2 = _run(["launchctl", "bootstrap", f"gui/{uid}", str(plist)], timeout=30)
        if rc2 == 0:
            rc, out = _run(["launchctl", "kickstart", "-k", target], timeout=30)
        else:
            out = f"{out}\n{out2}"
    return rc == 0, out[:400] or f"kickstart {label} rc={rc}"


def _kickstart_cloudflared() -> tuple[bool, str]:
    ok, detail = _kickstart_launchd(LAUNCHD_TUNNEL_LABEL, _tunnel_plist_path())
    time.sleep(3)
    local_ok, _ = _health_check_only()
    public_ok, public_msg = _public_health_check()
    return ok and (public_ok or local_ok), f"{detail}; public={public_msg[:120]}"


def _sync_bundled_node() -> tuple[bool, str]:
    src = ROOT
    runtime = RUNTIME
    if not runtime.is_dir():
        return False, f"runtime missing: {runtime}"
    synced = False
    for node_dir in ("node-v22.16.0-darwin-arm64", "node-v20.20.2-darwin-arm64"):
        src_dir = src / ".tools" / node_dir
        if not src_dir.is_dir():
            continue
        dest = runtime / ".tools" / node_dir
        dest.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("rsync"):
            rc, out = _run(
                ["rsync", "-a", f"{src_dir}/", f"{dest}/"],
                timeout=180,
            )
        else:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src_dir, dest)
            rc, out = 0, f"copied {node_dir}"
        synced = synced or rc == 0
        if rc != 0:
            return False, out
    node_bin = bundled_node_binary()
    if not Path(node_bin).is_file() and node_bin == "node":
        return False, "no bundled node in repo .tools/"
    return synced, f"node synced; binary={node_bin}"


def _mirror_repo_runtime() -> tuple[bool, str]:
    src_cloudy = ROOT / "cloudy"
    dest_cloudy = RUNTIME / "cloudy"
    if not src_cloudy.is_dir():
        return False, f"missing {src_cloudy}"
    dest_cloudy.parent.mkdir(parents=True, exist_ok=True)
    rc, out = _run(
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude",
            "__pycache__",
            "--exclude",
            "*.pyc",
            f"{src_cloudy}/",
            f"{dest_cloudy}/",
        ],
        timeout=180,
    )
    details = [out]
    for rel in ("scripts/cloudy.py", "requirements.txt"):
        s, d = ROOT / rel, RUNTIME / rel
        if s.is_file():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    return rc == 0, "; ".join(x for x in details if x) or "mirrored cloudy/"


def _sync_secrets_runtime() -> tuple[bool, str]:
    copied: list[str] = []
    (RUNTIME / "config").mkdir(parents=True, exist_ok=True)
    for name in _SECRET_FILES:
        src = CONFIG_DIR / name
        if not src.is_file():
            continue
        dest = RUNTIME / "config" / name
        shutil.copy2(src, dest)
        os.chmod(dest, 0o600)
        copied.append(name)
    for name in ("llm.json", "cursor.json", "cursor-cloud.json", "whatsapp.json"):
        src = CONFIG_DIR / name
        if src.is_file():
            dest = RUNTIME / "config" / name
            shutil.copy2(src, dest)
            copied.append(name)
    return bool(copied), f"copied: {', '.join(copied) or 'none'}"


def _notify_manual() -> tuple[bool, str]:
    return True, "manual: bash deploy/macos/install-autostart.sh"


def _health_check_only() -> tuple[bool, str]:
    rc, out = _run(["curl", "-s", "-m", "5", "http://127.0.0.1:8080/health"], timeout=10)
    return rc == 0 and "ok" in out.lower(), out[:200]


def _public_health_check(url: str | None = None) -> tuple[bool, str]:
    target = (url or PUBLIC_HEALTH_URL).strip()
    rc, out = _run(["curl", "-s", "-m", "12", "-o", "/dev/null", "-w", "%{http_code}", target], timeout=15)
    ok = rc == 0 and out.strip() in ("200", "204")
    if not ok:
        _, body = _run(["curl", "-s", "-m", "8", target], timeout=12)
        return False, body[:160] or f"http={out}"
    return True, f"http={out}"


def _clear_sync_lock() -> tuple[bool, str]:
    if not SYNC_LOCK_DIR.is_dir():
        return True, "no lock dir"
    try:
        age = time.time() - SYNC_LOCK_DIR.stat().st_mtime
    except OSError as exc:
        return False, str(exc)
    if age < SYNC_LOCK_STALE_SEC:
        return True, f"lock fresh ({int(age)}s) — no touch"
    try:
        SYNC_LOCK_DIR.rmdir()
        return True, f"removed stale lock ({int(age)}s)"
    except OSError as exc:
        return False, str(exc)


def _remediate_llm_chain() -> tuple[bool, str]:
    steps: list[str] = []
    ok_node, d1 = _sync_bundled_node()
    steps.append(f"node:{d1[:80]}")
    ok_sec, d2 = _sync_secrets_runtime()
    steps.append(f"secrets:{d2[:80]}")
    ok_bot, d3 = _kickstart_bot()
    steps.append(f"bot:{d3[:80]}")
    return ok_node and ok_bot, " | ".join(steps)


_ACTIONS: dict[str, Callable[[], tuple[bool, str]]] = {
    "kickstart_bot": _kickstart_bot,
    "kickstart_cloudflared": _kickstart_cloudflared,
    "sync_bundled_node": _sync_bundled_node,
    "mirror_repo_runtime": _mirror_repo_runtime,
    "sync_secrets_runtime": _sync_secrets_runtime,
    "clear_sync_lock": _clear_sync_lock,
    "remediate_llm_chain": _remediate_llm_chain,
    "notify_manual": _notify_manual,
    "health_check": _health_check_only,
}


def _apply_action(action: str) -> tuple[bool, str]:
    fn = _ACTIONS.get(action)
    if not fn:
        return False, f"acción desconocida: {action}"
    return fn()


def _maybe_notify(report: AutofixReport, cfg: dict[str, Any]) -> None:
    if report.dry_run:
        return
    notify_ok = bool(cfg.get("notify_on_fix"))
    notify_fail = bool(cfg.get("notify_on_failure"))
    notify_manual = bool(cfg.get("notify_on_manual", True))
    if not (notify_ok or notify_fail or notify_manual):
        return

    lines: list[str] = []
    for item in report.applied:
        ok = bool(item.get("ok"))
        action = str(item.get("action") or "")
        if ok and not notify_ok and action != "notify_manual":
            continue
        if ok and action == "notify_manual" and not notify_manual:
            continue
        if not ok and not notify_fail:
            continue
        status = "OK" if ok else "FALLÓ"
        lines.append(f"• {item.get('rule_id')} [{status}]: {str(item.get('detail', ''))[:120]}")

    if not lines:
        return

    try:
        from cloudy.bot.config import get_company
        from cloudy.bot.wa_client import notify_owner

        company = get_company(str(cfg.get("owner_company") or "unlockers"))
        header = f"Cloudy autofix ({report.trigger})"
        if report.errors:
            header += " — hubo errores"
        notify_owner(company, f"{header}:\n" + "\n".join(lines[:6]))
    except Exception as exc:
        logger.warning("notify autofix: %s", exc)


def _rule_still_relevant(match: MatchResult) -> bool:
    """Evita remediar patrones viejos en el tail si el sistema ya está sano."""
    if match.rule_id == "cursor_node_missing":
        for node_dir in ("node-v22.16.0-darwin-arm64", "node-v20.20.2-darwin-arm64"):
            node_bin = RUNTIME / ".tools" / node_dir / "bin" / "node"
            if node_bin.is_file():
                return False
        return True
    if match.rule_id in ("bot_health_down", "gather_flush_failed"):
        ok, _ = _health_check_only()
        return not ok
    if match.rule_id in ("cloudflared_tunnel_glitch", "tunnel_public_down"):
        local_ok, _ = _health_check_only()
        public_ok, _ = _public_health_check()
        return local_ok and not public_ok
    if match.rule_id == "sync_lock_stale":
        if not SYNC_LOCK_DIR.is_dir():
            return False
        try:
            age = time.time() - SYNC_LOCK_DIR.stat().st_mtime
            return age >= SYNC_LOCK_STALE_SEC
        except OSError:
            return False
    if match.rule_id in ("bot_sync_failed", "chromadb_duplicate_id"):
        autosync = LOGS / "bot-autosync.log"
        if autosync.is_file():
            tail = _tail_text(autosync, 8000)
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            recent = lines[-8:] if lines else []
            if recent and any("done trigger=" in ln for ln in recent[-3:]):
                if not any("FAIL sync" in ln for ln in recent[-5:]):
                    return False
    return True


def run_autofix(
    *,
    dry_run: bool = False,
    trigger: str = "cli",
    force_rule: str = "",
) -> AutofixReport:
    cfg = load_config()
    report = AutofixReport(ran_at=_utc_now(), trigger=trigger, dry_run=dry_run)
    if not cfg.get("enabled", True):
        report.skipped.append({"reason": "disabled"})
        return report

    _, matches = scan_logs(cfg)
    report.matches = matches
    report.log_bytes_scanned = int(cfg.get("tail_bytes", 300_000))

    # Si /health falla ahora, añadir match aunque no esté en logs recientes
    health_ok, health_msg = _health_check_only()
    public_url = str(cfg.get("public_health_url") or PUBLIC_HEALTH_URL)
    public_ok, public_msg = _public_health_check(public_url)
    if not health_ok and not any(m.rule_id == "bot_health_down" for m in matches):
        matches.append(
            MatchResult(
                rule_id="bot_health_down",
                action="kickstart_bot",
                description="Health check activo falló",
                severity="alert",
                sample=health_msg,
            )
        )
    if health_ok and not public_ok and not any(
        m.rule_id in ("cloudflared_tunnel_glitch", "tunnel_public_down") for m in matches
    ):
        matches.append(
            MatchResult(
                rule_id="tunnel_public_down",
                action="kickstart_cloudflared",
                description=f"Túnel público caído ({public_url}) con bot local OK",
                severity="alert",
                sample=public_msg,
            )
        )
    report.matches = matches

    if force_rule:
        matches = [m for m in matches if m.rule_id == force_rule]
        report.matches = matches

    matches = [m for m in matches if _rule_still_relevant(m)]
    report.matches = matches

    rules_by_id = {r.rule_id: r for r in _compile_rules(cfg)}
    state = _load_state()
    state["last_run"] = report.ran_at

    for match in matches:
        rule = rules_by_id.get(match.rule_id)
        cooldown = rule.cooldown_min if rule else int(cfg.get("default_cooldown_min", 30))
        if not force_rule and not _cooldown_ok(state, match.rule_id, cooldown):
            report.skipped.append({
                "rule_id": match.rule_id,
                "reason": "cooldown",
                "cooldown_min": cooldown,
            })
            continue

        if dry_run:
            report.applied.append({
                "rule_id": match.rule_id,
                "action": match.action,
                "ok": True,
                "dry_run": True,
                "detail": match.description,
            })
            continue

        ok, detail = _apply_action(match.action)
        entry = {
            "rule_id": match.rule_id,
            "action": match.action,
            "ok": ok,
            "detail": detail[:500],
            "at": report.ran_at,
        }
        report.applied.append(entry)
        if not ok:
            report.errors.append(f"{match.rule_id}: {detail[:200]}")

        fixes = state.setdefault("fixes", {})
        fix_rec = fixes.setdefault(match.rule_id, {"count": 0})
        fix_rec["last_applied"] = report.ran_at
        fix_rec["count"] = int(fix_rec.get("count") or 0) + 1
        fix_rec["last_ok"] = ok

        history = state.setdefault("history", [])
        history.append(entry)
        state["history"] = history[-50:]

    _save_state(state)
    _write_report(report)
    _maybe_notify(report, cfg)
    return report


def autofix_status() -> dict[str, Any]:
    cfg = load_config()
    state = _load_state()
    _, matches = scan_logs(cfg)
    health_ok, health_msg = _health_check_only()
    actionable = [m for m in matches if _rule_still_relevant(m)]
    public_ok, public_msg = _public_health_check(str(cfg.get("public_health_url") or PUBLIC_HEALTH_URL))
    return {
        "enabled": cfg.get("enabled", True),
        "notify_on_fix": cfg.get("notify_on_fix"),
        "last_run": state.get("last_run"),
        "runtime": str(RUNTIME),
        "logs_dir": str(LOGS),
        "health_ok": health_ok,
        "health": health_msg,
        "public_health_ok": public_ok,
        "public_health": public_msg,
        "log_matches": len(matches),
        "actionable_matches": [
            {
                "rule_id": m.rule_id,
                "action": m.action,
                "severity": m.severity,
                "sample": m.sample[:160],
            }
            for m in actionable
        ],
        "fixes": state.get("fixes") or {},
        "recent_history": (state.get("history") or [])[-10:],
    }


def _write_report(report: AutofixReport) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report.ran_at.replace(":", "").replace("-", "")
    path = REPORT_DIR / f"autofix-{stamp}.json"
    payload = {
        "ran_at": report.ran_at,
        "trigger": report.trigger,
        "dry_run": report.dry_run,
        "matches": [m.__dict__ for m in report.matches],
        "applied": report.applied,
        "skipped": report.skipped,
        "errors": report.errors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
