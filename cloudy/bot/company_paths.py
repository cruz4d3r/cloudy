# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Tenant-scoped paths for multi-company bot personalization.

Every WABA is an isolated tenant. All persona, KB, style and sync state
live under data/rag/companies/{company_alias}/ — never in a shared global folder.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloudy.bot.config import ROOT, companies, get_company

COMPANIES_ROOT = ROOT / "data" / "rag" / "companies"

# Legacy unlockers KB (migrated gradually; still indexed during transition).
_LEGACY_UNLOCKERS_KB = ROOT / "data" / "rag" / "unlockers"
_LEGACY_UNLOCKERS_PROMPTS = _LEGACY_UNLOCKERS_KB / "prompts"

# Katana mirror root (one subfolder per tenant).
BOT_KB_ROOT = ROOT / "clients/1lockers/sites/web/storage/app/bot_kb"


def require_company_alias(company_alias: str, company: dict[str, Any] | None = None) -> str:
    """Fail fast when company is missing — never default silently to unlockers."""
    alias = str(company_alias or "").strip()
    if not alias:
        raise ValueError("company_alias is required for tenant-scoped bot operations")
    if company is not None:
        resolved = str(company.get("alias") or company.get("rag_collection") or "").strip()
        if resolved and resolved != alias:
            raise ValueError(f"company_alias mismatch: expected {alias}, got {resolved}")
    return alias


def company_root(company_alias: str) -> Path:
    return COMPANIES_ROOT / require_company_alias(company_alias)


def kb_dir(company_alias: str) -> Path:
    return company_root(company_alias) / "kb"


def by_contact_dir(company_alias: str, contact: str) -> Path:
    digits = "".join(ch for ch in str(contact) if ch.isdigit())
    return company_root(company_alias) / "by-contact" / digits


def by_alias_dir(company_alias: str, client_alias: str) -> Path:
    safe = str(client_alias or "").strip().lower().replace("/", "-")
    return company_root(company_alias) / "by-alias" / safe


def style_path(company_alias: str) -> Path:
    return company_root(company_alias) / "estilo-fewshot.md"


def prompts_dir(company_alias: str) -> Path:
    tenant = company_root(company_alias) / "prompts"
    if tenant.is_dir():
        return tenant
    if company_alias == "unlockers" and _LEGACY_UNLOCKERS_PROMPTS.is_dir():
        return _LEGACY_UNLOCKERS_PROMPTS
    return tenant


def manifest_path(company_alias: str) -> Path:
    return company_root(company_alias) / "manifest.json"


def sync_state_path(company_alias: str) -> Path:
    return company_root(company_alias) / ".bot_sync_state.json"


def observacion_dir(company_alias: str, company: dict[str, Any] | None = None) -> Path | None:
    co = company or get_company(company_alias)
    rel = str(co.get("observe_dir") or "").strip()
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    return path


def ensure_company_layout(company_alias: str) -> Path:
    root = company_root(company_alias)
    for sub in ("kb", "by-contact", "by-alias", "prompts"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def enabled_company_aliases() -> list[str]:
    return sorted(companies().keys())


def client_cloudy_aliases(client: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for raw in client.get("cloudy_aliases") or []:
        name = str(raw).strip().lower()
        if name and name not in aliases:
            aliases.append(name)
    primary = str(client.get("alias") or "").strip().lower()
    if primary and primary not in aliases and not primary.startswith("_"):
        aliases.insert(0, primary)
    return aliases


def alias_belongs_to_company(company_alias: str, client_alias: str, company: dict[str, Any] | None = None) -> bool:
    """True when client_alias is registered under this WABA tenant."""
    co = company or get_company(company_alias)
    target = str(client_alias or "").strip().lower()
    if not target or target.startswith("_"):
        return False
    for client in (co.get("clients") or {}).values():
        if not isinstance(client, dict):
            continue
        if target in client_cloudy_aliases(client):
            return True
        if str(client.get("alias") or "").strip().lower() == target:
            return True
    # Tenant's own brand alias (paolapalacio on paolapalacio WABA).
    if target == company_alias:
        return True
    return False


def validate_alias_scope(company_alias: str, client_alias: str, company: dict[str, Any] | None = None) -> None:
    if not alias_belongs_to_company(company_alias, client_alias, company):
        raise ValueError(
            f"alias '{client_alias}' does not belong to tenant '{company_alias}' — cross-tenant write blocked"
        )


def knowledge_source_paths(company_alias: str, company: dict[str, Any] | None = None) -> list[Path]:
    """All markdown KB roots for ingest (tenant kb/ + legacy rag_sources)."""
    co = company or get_company(company_alias)
    require_company_alias(company_alias, co)
    paths: list[Path] = []
    tenant_kb = kb_dir(company_alias)
    if tenant_kb.is_dir():
        paths.append(tenant_kb)
    for raw in co.get("rag_sources") or []:
        path = (ROOT / raw).resolve() if not str(raw).startswith("/") else Path(raw)
        if path.is_file() or path.is_dir():
            if path not in paths:
                paths.append(path)
    return paths


def persona_path(company_alias: str, contact: str) -> Path:
    return by_contact_dir(company_alias, contact) / "persona.md"


def corpus_path(company_alias: str, contact: str) -> Path:
    return by_contact_dir(company_alias, contact) / "corpus.txt"


def bot_kb_path(company_alias: str, client_alias: str) -> Path:
    return by_alias_dir(company_alias, client_alias) / "bot-kb.md"


def katana_mirror_dir(company_alias: str) -> Path:
    return BOT_KB_ROOT / require_company_alias(company_alias)


def bundle_content_hash(company_alias: str, company: dict[str, Any] | None = None) -> str:
    """Hash of KB + personas + style for manifest / sync-check."""
    parts: list[str] = []
    for path in sorted(knowledge_source_paths(company_alias, company)):
        if path.is_file() and path.suffix == ".md":
            parts.append(hashlib.sha256(path.read_bytes()).hexdigest())
        elif path.is_dir():
            for md in sorted(path.rglob("*.md")):
                if md.name.startswith("."):
                    continue
                parts.append(hashlib.sha256(md.read_bytes()).hexdigest())
    style = style_path(company_alias)
    if style.is_file():
        parts.append(hashlib.sha256(style.read_bytes()).hexdigest())
    for prompt in sorted(prompts_dir(company_alias).glob("*.md")):
        parts.append(hashlib.sha256(prompt.read_bytes()).hexdigest())
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:16]


def list_persona_contacts(company_alias: str) -> list[str]:
    base = company_root(company_alias) / "by-contact"
    if not base.is_dir():
        return []
    return sorted(
        p.name for p in base.iterdir() if p.is_dir() and (p / "persona.md").is_file()
    )


def list_alias_kb(company_alias: str) -> list[str]:
    base = company_root(company_alias) / "by-alias"
    if not base.is_dir():
        return []
    return sorted(
        p.name for p in base.iterdir() if p.is_dir() and (p / "bot-kb.md").is_file()
    )


def chroma_collections(company_alias: str, company: dict[str, Any] | None = None) -> list[str]:
    co = company or get_company(company_alias)
    base = str(co.get("rag_collection") or company_alias)
    return [base, f"{base}_conv", f"{base}_client"]


def write_manifest(company_alias: str, company: dict[str, Any] | None = None) -> dict[str, Any]:
    co = company or get_company(company_alias)
    ensure_company_layout(company_alias)
    payload = {
        "company": company_alias,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "persona_contacts": list_persona_contacts(company_alias),
        "alias_kb": list_alias_kb(company_alias),
        "bundle_hash": bundle_content_hash(company_alias, co),
        "chroma_collections": chroma_collections(company_alias, co),
    }
    path = manifest_path(company_alias)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_manifest(company_alias: str) -> dict[str, Any]:
    path = manifest_path(company_alias)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


__all__ = [
    "BOT_KB_ROOT",
    "COMPANIES_ROOT",
    "alias_belongs_to_company",
    "bot_kb_path",
    "bundle_content_hash",
    "by_alias_dir",
    "by_contact_dir",
    "chroma_collections",
    "client_cloudy_aliases",
    "company_root",
    "corpus_path",
    "enabled_company_aliases",
    "ensure_company_layout",
    "katana_mirror_dir",
    "kb_dir",
    "knowledge_source_paths",
    "list_alias_kb",
    "list_persona_contacts",
    "load_manifest",
    "manifest_path",
    "observacion_dir",
    "persona_path",
    "prompts_dir",
    "require_company_alias",
    "style_path",
    "sync_state_path",
    "validate_alias_scope",
    "write_manifest",
]
