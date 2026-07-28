# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
RAG: ingestion + retrieval on a persistent Chroma index (data/rag/index/).

Isolation model per tenant (WABA):
  <rag_collection>         knowledge (markdown sources)
  <rag_collection>_conv    real Q/A pairs from WhatsApp exports
  <rag_collection>_client  personas + bot-kb scoped by contact/alias

retrieve_for_contact NEVER crosses tenants — company metadata filter on every query.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from cloudy.bot.company_paths import (
    client_cloudy_aliases,
    company_root,
    knowledge_source_paths,
    require_company_alias,
    style_path,
)
from cloudy.bot.config import ROOT
from cloudy.bot.llm import embed

INDEX_DIR = ROOT / "data" / "rag" / "index"

_CHUNK_CHARS = 900
_EMBED_BATCH = 16

# Chunks de otros clientes/proyectos que no deben contaminar leads comerciales genéricos.
_COMMERCIAL_BLEED_RE = re.compile(
    r"paolapalacio|fecoljudo|decolombiajoyas|mixcoco|neogestion|lantonella|"
    r"by-contact/|bot-kb\.md|Persona —|Tenant:.*paolapalacio",
    re.I,
)

_COMMERCIAL_KB_PRIORITY = (
    "faqs-servicios",
    "02-leads-correo",
    "20-campana-correo",
    "01-cta-y-citas",
    "10-diseno-web",
    "11-hosting",
    "12-tienda",
    "13-marketing",
    "16-rescate",
)


def _client():
    import chromadb

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(INDEX_DIR))


def _chunk_markdown(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 > _CHUNK_CHARS and current:
            chunks.append(current.strip())
            current = ""
        current += ("\n\n" if current else "") + paragraph
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _collect_paths(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
        elif path.is_file() and path.suffix == ".md":
            files.append(path)
    return [f for f in files if not f.name.startswith(".")]


def _chunk_id(document: str, metadata: dict[str, str]) -> str:
    """
    Stable Chroma id from document body + metadata.

    Content-only hashes collide when the same Q/A is exported twice (pull_learning
    appends katana_*.txt) or when two KB chunks share identical text.
    """
    parts = [document]
    for key in sorted(metadata.keys()):
        parts.append(f"{key}={metadata[key]}")
    payload = "\x1e".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _upsert(collection: Any, documents: list[str], metadatas: list[dict[str, str]]) -> int:
    if not documents:
        return 0
    if len(documents) != len(metadatas):
        raise ValueError("documents and metadatas length mismatch")

    deduped_docs: list[str] = []
    deduped_metas: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for doc, meta in zip(documents, metadatas):
        cid = _chunk_id(doc, meta)
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        deduped_docs.append(doc)
        deduped_metas.append(meta)

    total = 0
    for start in range(0, len(deduped_docs), _EMBED_BATCH):
        docs = deduped_docs[start : start + _EMBED_BATCH]
        metas = deduped_metas[start : start + _EMBED_BATCH]
        ids = [_chunk_id(d, m) for d, m in zip(docs, metas)]
        collection.upsert(ids=ids, documents=docs, embeddings=embed(docs), metadatas=metas)
        total += len(docs)
    return total


def _query_collection(
    client: Any,
    name: str,
    query_embedding: list[list[float]],
    *,
    company_alias: str,
    n_results: int,
    extra_where: dict[str, str] | None = None,
) -> list[str]:
    try:
        collection = client.get_collection(name)
    except Exception:
        return []
    where: dict[str, Any] = {"company": company_alias}
    if extra_where:
        where.update(extra_where)
    try:
        result = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
        )
    except Exception:
        # Fallback without where if legacy chunks lack company metadata.
        result = collection.query(query_embeddings=query_embedding, n_results=n_results)
    passages: list[str] = []
    for doc in (result.get("documents") or [[]])[0]:
        if doc:
            passages.append(doc)
    return passages


def ingest_knowledge(company_alias: str, company: dict[str, Any]) -> dict[str, int]:
    """(Re)index every markdown source for the tenant."""
    require_company_alias(company_alias, company)
    collection = _client().get_or_create_collection(
        str(company.get("rag_collection") or company_alias)
    )
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    paths = _collect_paths(knowledge_source_paths(company_alias, company))
    for path in paths:
        # Skip persona/corpus paths — those go to _client collection.
        rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
        if "/by-contact/" in rel and path.name in ("persona.md", "corpus.txt"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for chunk in _chunk_markdown(text):
            documents.append(chunk)
            metadatas.append({"source": path.name, "company": company_alias, "kind": "kb"})
    count = _upsert(collection, documents, metadatas) if documents else 0
    return {"files": len(paths), "chunks": count}


def ingest_conversations(company_alias: str, company: dict[str, Any]) -> dict[str, int]:
    from cloudy.bot.conversations import build_style_examples, parse_exports_dir

    require_company_alias(company_alias, company)
    conv_dir_raw = str(company.get("conversations_dir") or "")
    if not conv_dir_raw:
        return {"files": 0, "pairs": 0}
    conv_dir = ROOT / conv_dir_raw
    pairs, files_parsed = parse_exports_dir(conv_dir, owner_name=str(company.get("owner_name") or ""))
    if pairs:
        collection = _client().get_or_create_collection(
            f"{company.get('rag_collection') or company_alias}_conv"
        )
        documents = [
            f"Cliente: {q}\nRespuesta de {company.get('owner_name') or 'el asesor'}: {a}"
            for q, a in pairs
        ]
        metadatas = [
            {
                "company": company_alias,
                "kind": "conversation",
                "pair_id": hashlib.sha256(f"{q}\x1f{a}".encode("utf-8")).hexdigest()[:16],
            }
            for q, a in pairs
        ]
        _upsert(collection, documents, metadatas)

    tenant_style = style_path(company_alias)
    if pairs:
        tenant_style.parent.mkdir(parents=True, exist_ok=True)
        tenant_style.write_text(build_style_examples(pairs), encoding="utf-8")
    elif str(company.get("style_file") or "").strip() and pairs:
        style_path_legacy = ROOT / str(company.get("style_file"))
        style_path_legacy.parent.mkdir(parents=True, exist_ok=True)
        style_path_legacy.write_text(build_style_examples(pairs), encoding="utf-8")

    return {"files": files_parsed, "pairs": len(pairs)}


def ingest_client_rag(company_alias: str, company: dict[str, Any]) -> dict[str, int]:
    """
    Index persona.md and bot-kb.md under data/rag/companies/{empresa}/.
    Collection: {rag_collection}_client with metadata company + contact + client_alias.
    """
    require_company_alias(company_alias, company)
    base = str(company.get("rag_collection") or company_alias)
    collection = _client().get_or_create_collection(f"{base}_client")
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    root = company_root(company_alias)
    if not root.is_dir():
        return {"files": 0, "chunks": 0}

    for persona in sorted((root / "by-contact").glob("*/persona.md")):
        contact = persona.parent.name
        text = persona.read_text(encoding="utf-8", errors="replace")
        for chunk in _chunk_markdown(text):
            documents.append(chunk)
            metadatas.append(
                {
                    "company": company_alias,
                    "contact": contact,
                    "client_alias": "",
                    "kind": "persona",
                    "source": f"by-contact/{contact}/persona.md",
                }
            )

    for kb in sorted((root / "by-alias").glob("*/bot-kb.md")):
        alias = kb.parent.name
        text = kb.read_text(encoding="utf-8", errors="replace")
        for chunk in _chunk_markdown(text):
            documents.append(chunk)
            metadatas.append(
                {
                    "company": company_alias,
                    "contact": "",
                    "client_alias": alias,
                    "kind": "kb",
                    "source": f"by-alias/{alias}/bot-kb.md",
                }
            )

    count = _upsert(collection, documents, metadatas) if documents else 0
    return {"files": len(documents), "chunks": count}


def ingest_learning_turn(
    company_alias: str,
    company: dict[str, Any],
    contact: str,
    *,
    user_text: str = "",
    staff_text: str = "",
) -> dict[str, int]:
    """
    Incremental Q/A upsert after each observed staff reply.
    Writes to {rag_collection}_conv and refreshes persona chunk in _client.
    """
    require_company_alias(company_alias, company)
    base = str(company.get("rag_collection") or company_alias)
    client = _client()
    conv = client.get_or_create_collection(f"{base}_conv")
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    digits = "".join(ch for ch in str(contact) if ch.isdigit())

    user_text = (user_text or "").strip()
    staff_text = (staff_text or "").strip()
    if user_text and staff_text:
        qa_doc = f"Pregunta del cliente:\n{user_text}\n\nRespuesta:\n{staff_text}"
        documents.append(qa_doc)
        metadatas.append(
            {
                "company": company_alias,
                "contact": digits,
                "client_alias": "",
                "kind": "qa",
                "source": f"live/{digits}",
            }
        )

    if staff_text and not user_text:
        documents.append(f"Respuesta de referencia:\n{staff_text}")
        metadatas.append(
            {
                "company": company_alias,
                "contact": digits,
                "client_alias": "",
                "kind": "staff_line",
                "source": f"live/{digits}",
            }
        )

    conv_chunks = _upsert(conv, documents, metadatas) if documents else 0

    persona_chunks = 0
    from cloudy.bot.company_paths import persona_path

    persona_file = persona_path(company_alias, contact)
    if persona_file.is_file():
        pcol = client.get_or_create_collection(f"{base}_client")
        text = persona_file.read_text(encoding="utf-8", errors="replace")
        for chunk in _chunk_markdown(text):
            documents_p = [chunk]
            metas_p = [
                {
                    "company": company_alias,
                    "contact": digits,
                    "client_alias": "",
                    "kind": "persona",
                    "source": f"by-contact/{digits}/persona.md",
                }
            ]
            persona_chunks += _upsert(pcol, documents_p, metas_p)

    return {"conv_chunks": conv_chunks, "persona_chunks": persona_chunks}


def load_contact_persona(company_alias: str, contact: str) -> str:
    from cloudy.bot.company_paths import persona_path

    path = persona_path(company_alias, contact)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def load_alias_kb(company_alias: str, client_alias: str) -> str:
    from cloudy.bot.company_paths import bot_kb_path

    path = bot_kb_path(company_alias, client_alias)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _is_commercial_prospect(client: dict[str, Any]) -> bool:
    if not client.get("prospect"):
        return False
    if client.get("sites"):
        return False
    if client_cloudy_aliases(client):
        return False
    return True


def _filter_commercial_passages(passages: list[str]) -> list[str]:
    filtered = [p for p in passages if not _COMMERCIAL_BLEED_RE.search(p[:800])]
    return filtered if filtered else passages


def _prioritize_commercial_kb(passages: list[str]) -> list[str]:
    priority: list[str] = []
    rest: list[str] = []
    for passage in passages:
        head = passage[:400].lower()
        if any(token in head for token in _COMMERCIAL_KB_PRIORITY):
            priority.append(passage)
        else:
            rest.append(passage)
    return priority + rest


def load_style_examples(company: dict[str, Any], company_alias: str = "") -> str:
    alias = company_alias or str(company.get("alias") or company.get("rag_collection") or "")
    if alias:
        tenant = style_path(alias)
        if tenant.is_file():
            return tenant.read_text(encoding="utf-8", errors="replace").strip()
    style_raw = str(company.get("style_file") or "")
    if not style_raw:
        return ""
    path = ROOT / style_raw
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def retrieve(company_alias: str, company: dict[str, Any], query: str, k: int = 4) -> list[str]:
    """Top-k context passages: knowledge + conversations (tenant-scoped)."""
    require_company_alias(company_alias, company)
    client = _client()
    base = str(company.get("rag_collection") or company_alias)
    passages: list[str] = []
    query_embedding = embed([query])
    for name, take in ((base, k), (f"{base}_conv", max(2, k // 2))):
        passages.extend(
            _query_collection(
                client, name, query_embedding, company_alias=company_alias, n_results=take
            )
        )
    return passages


def retrieve_for_contact(
    company_alias: str,
    company: dict[str, Any],
    contact: str,
    client: dict[str, Any],
    query: str,
    k: int = 4,
) -> list[str]:
    """
    Tenant-scoped retrieval with contact/alias priority:
      1) persona chunks for this contact
      2) bot-kb for client's cloudy aliases
      3) general KB + conversations of SAME tenant only
    """
    require_company_alias(company_alias, company)
    client_ch = _client()
    base = str(company.get("rag_collection") or company_alias)
    query_embedding = embed([query])
    passages: list[str] = []
    digits = "".join(ch for ch in str(contact) if ch.isdigit())
    commercial = _is_commercial_prospect(client)

    if commercial:
        passages.extend(
            _query_collection(
                client_ch,
                base,
                query_embedding,
                company_alias=company_alias,
                n_results=max(k + 2, 6),
            )
        )
        passages = _prioritize_commercial_kb(_filter_commercial_passages(passages))
        seen: set[str] = set()
        unique: list[str] = []
        for p in passages:
            key = p[:200]
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique[: k + 2]

    # Priority 1: contact persona
    if digits:
        passages.extend(
            _query_collection(
                client_ch,
                f"{base}_client",
                query_embedding,
                company_alias=company_alias,
                n_results=max(2, k // 2),
                extra_where={"contact": digits, "kind": "persona"},
            )
        )

    # Priority 2: alias KB
    for alias in client_cloudy_aliases(client):
        alias_passages = _query_collection(
            client_ch,
            f"{base}_client",
            query_embedding,
            company_alias=company_alias,
            n_results=2,
            extra_where={"client_alias": alias, "kind": "kb"},
        )
        passages.extend(alias_passages)

    # Priority 3: general tenant KB
    for name, take in ((base, k), (f"{base}_conv", max(2, k // 2))):
        passages.extend(
            _query_collection(
                client_ch, name, query_embedding, company_alias=company_alias, n_results=take
            )
        )

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in passages:
        key = p[:200]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[: k + 4]


__all__ = [
    "ingest_client_rag",
    "ingest_conversations",
    "ingest_knowledge",
    "ingest_learning_turn",
    "load_alias_kb",
    "load_contact_persona",
    "load_style_examples",
    "retrieve",
    "retrieve_for_contact",
]
