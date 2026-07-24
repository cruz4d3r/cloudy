# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
SQLite persistence for the bot (data/rag/bot.sqlite).

Every table carries a `company` column so one database serves N companies
(multi-tenant isolation without extra infrastructure). This file — together
with data/rag/index/ — is the only state to copy when migrating from the
Mac to the AlmaLinux dedicated server.

Tables:
  sessions      - one row per (company, contact): paused flag + history JSON
  meetings      - scheduled meetings (client, when, topic, status)
  requests      - change/improvement requests (site, description, priority, status)
  sent_messages - wamids the bot sent; an outbound echo NOT in here = human
                  wrote from phone/WhatsApp Web -> pause that conversation
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "rag" / "bot.sqlite"

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    company        TEXT NOT NULL,
    contact        TEXT NOT NULL,
    client_alias   TEXT DEFAULT '',
    client_name    TEXT DEFAULT '',
    paused_until   TEXT DEFAULT NULL,
    paused_reason  TEXT DEFAULT '',
    history        TEXT DEFAULT '[]',
    pending        TEXT DEFAULT '{}',
    last_inbound   TEXT DEFAULT NULL,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (company, contact)
);
CREATE TABLE IF NOT EXISTS meetings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    company        TEXT NOT NULL,
    contact        TEXT NOT NULL,
    client_alias   TEXT DEFAULT '',
    client_name    TEXT DEFAULT '',
    scheduled_at   TEXT NOT NULL,
    duration_min   INTEGER DEFAULT 45,
    topic          TEXT DEFAULT '',
    status         TEXT DEFAULT 'confirmada',
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    company        TEXT NOT NULL,
    contact        TEXT NOT NULL,
    client_alias   TEXT DEFAULT '',
    client_name    TEXT DEFAULT '',
    site           TEXT DEFAULT '',
    description    TEXT NOT NULL,
    priority       TEXT DEFAULT 'normal',
    status         TEXT DEFAULT 'nueva',
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sent_messages (
    wamid          TEXT PRIMARY KEY,
    company        TEXT NOT NULL,
    contact        TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS processed_events (
    wamid          TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS message_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    company        TEXT NOT NULL,
    contact        TEXT NOT NULL,
    role           TEXT NOT NULL,
    kind           TEXT DEFAULT 'text',
    content        TEXT NOT NULL,
    wamid          TEXT DEFAULT '',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_message_log_lookup
    ON message_log (company, contact, created_at);
CREATE TABLE IF NOT EXISTS agenda_nudges (
    company        TEXT NOT NULL,
    contact        TEXT NOT NULL,
    client_name    TEXT DEFAULT '',
    topic          TEXT DEFAULT '',
    offered_at     TEXT NOT NULL,
    nudge_at       TEXT NOT NULL,
    status         TEXT DEFAULT 'pending',
    PRIMARY KEY (company, contact)
);
CREATE INDEX IF NOT EXISTS idx_agenda_nudges_due
    ON agenda_nudges (status, nudge_at);
CREATE TABLE IF NOT EXISTS crm_sync (
    company              TEXT NOT NULL,
    contact              TEXT NOT NULL,
    commercial_lead_id   INTEGER DEFAULT NULL,
    crm_synced_at        TEXT DEFAULT NULL,
    crm_pending          INTEGER DEFAULT 0,
    crm_attempts         INTEGER DEFAULT 0,
    PRIMARY KEY (company, contact)
);
"""

# Keep ~1 week of conversation (text / voice transcripts / images / human echoes).
# Soft cap avoids unbounded JSON in sessions.history; durable copy is message_log.
_HISTORY_DAYS = 7
_MAX_HISTORY = 400
_LLM_HISTORY_LIMIT = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def trim_history(history: list[Any], *, days: int = _HISTORY_DAYS, max_turns: int = _MAX_HISTORY) -> list[dict[str, Any]]:
    """Keep turns from the last N days (and a soft max length)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out: list[dict[str, Any]] = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        ts = _parse_ts(turn.get("ts"))
        if ts is not None and ts < cutoff:
            continue
        entry = {
            "role": turn.get("role") or "user",
            "content": content[:4000],
            "ts": turn.get("ts") or _now(),
        }
        if turn.get("kind"):
            entry["kind"] = turn.get("kind")
        out.append(entry)
    if len(out) > max_turns:
        out = out[-max_turns:]
    return out


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _migrate_message_log(conn)
    _migrate_sessions(conn)
    return conn


def _migrate_message_log(conn: sqlite3.Connection) -> None:
    """Add LLM telemetry columns to message_log on existing databases."""
    cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(message_log)")}
    if "llm_engine" not in cols:
        conn.execute("ALTER TABLE message_log ADD COLUMN llm_engine TEXT DEFAULT ''")
    if "llm_model" not in cols:
        conn.execute("ALTER TABLE message_log ADD COLUMN llm_model TEXT DEFAULT ''")
    conn.commit()


def _migrate_sessions(conn: sqlite3.Connection) -> None:
    """Add contact country columns for market-aware replies."""
    cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(sessions)")}
    if "contact_country" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN contact_country TEXT DEFAULT ''")
    if "country_source" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN country_source TEXT DEFAULT ''")
    if "country_asked" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN country_asked INTEGER DEFAULT 0")
    conn.commit()


# ---------------------------------------------------------------- sessions

def get_session(company: str, contact: str) -> dict[str, Any]:
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE company=? AND contact=?", (company, contact)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO sessions (company, contact, updated_at) VALUES (?,?,?)",
                (company, contact, _now()),
            )
            return {
                "company": company, "contact": contact, "client_alias": "",
                "client_name": "", "paused_until": None, "paused_reason": "",
                "history": [], "pending": {}, "last_inbound": None,
                "contact_country": "", "country_source": "", "country_asked": 0,
            }
        data = dict(row)
        data["history"] = json.loads(data.get("history") or "[]")
        data["pending"] = json.loads(data.get("pending") or "{}")
        return data


def save_session(company: str, contact: str, **fields: Any) -> None:
    """Persist selected session fields (history/pending are JSON-encoded)."""
    allowed = {
        "client_alias", "client_name", "paused_until", "paused_reason",
        "history", "pending", "last_inbound",
        "contact_country", "country_source", "country_asked",
    }
    updates: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Unknown session field: {key}")
        if key in ("history", "pending"):
            if key == "history" and isinstance(value, list):
                value = trim_history(value)
            value = json.dumps(value, ensure_ascii=False)
        updates[key] = value
    if not updates:
        return
    updates["updated_at"] = _now()
    columns = ", ".join(f"{k}=?" for k in updates)
    with _lock, connect() as conn:
        conn.execute(
            f"UPDATE sessions SET {columns} WHERE company=? AND contact=?",
            (*updates.values(), company, contact),
        )


def log_turn(
    company: str,
    contact: str,
    role: str,
    content: str,
    *,
    kind: str = "text",
    wamid: str = "",
    push_katana: bool = True,
    llm_engine: str = "",
    llm_model: str = "",
) -> None:
    """Append one turn to durable message_log (kept ~14 days)."""
    body = (content or "").strip()
    if not body:
        return
    ts = _now()
    role_norm = role if role in ("user", "assistant", "system") else "assistant"
    engine_tag = str(llm_engine or "").strip()
    model_tag = str(llm_model or "").strip()
    with _lock, connect() as conn:
        conn.execute(
            "INSERT INTO message_log"
            " (company, contact, role, kind, content, wamid, created_at, llm_engine, llm_model)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                company, contact, role_norm, kind or "text", body[:4000],
                wamid or "", ts, engine_tag, model_tag,
            ),
        )
        cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        conn.execute("DELETE FROM message_log WHERE created_at < ?", (cutoff,))
    if push_katana and company == "unlockers" and role_norm in ("user", "assistant"):
        try:
            from cloudy.bot.katana_sync import push_turn_from_mac

            push_turn_from_mac(
                contact=contact,
                role=role_norm,
                content=body,
                channel="whatsapp",
                company_alias=company,
                via="mac",
                kind=kind or "text",
                llm_engine=engine_tag,
                llm_model=model_tag,
            )
        except Exception:
            pass


def count_user_messages(company: str, contact: str) -> int:
    """How many inbound user turns we already logged (for first-contact CRM)."""
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM message_log WHERE company=? AND contact=? AND role='user'",
            (company, contact),
        ).fetchone()
    return int(row["n"] if row else 0)


def last_user_message_at(company: str, contact: str) -> str | None:
    """ISO timestamp of the latest inbound user turn (for nurture eligibility)."""
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT created_at FROM message_log"
            " WHERE company=? AND contact=? AND role='user'"
            " ORDER BY created_at DESC LIMIT 1",
            (company, contact),
        ).fetchone()
    if not row or not row["created_at"]:
        return None
    return str(row["created_at"])


def recent_turns(
    company: str,
    contact: str,
    *,
    days: int = _HISTORY_DAYS,
    limit: int = _LLM_HISTORY_LIMIT,
) -> list[dict[str, str]]:
    """Last N days of turns for the LLM (prefers durable message_log)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _lock, connect() as conn:
        rows = conn.execute(
            "SELECT role, content, kind, created_at FROM message_log"
            " WHERE company=? AND contact=? AND created_at >= ?"
            " ORDER BY id DESC LIMIT ?",
            (company, contact, cutoff, max(1, limit)),
        ).fetchall()
    if rows:
        ordered = list(reversed(rows))
        out: list[dict[str, str]] = []
        for row in ordered:
            role = str(row["role"] or "user")
            if role not in ("user", "assistant", "system"):
                role = "assistant"
            content = str(row["content"] or "").strip()
            if content:
                out.append({
                    "role": role,
                    "content": content[:4000],
                    "created_at": str(row["created_at"] or ""),
                    "kind": str(row["kind"] or "text"),
                })
        return out

    # Fallback: session JSON history (legacy rows before message_log).
    session = get_session(company, contact)
    trimmed = trim_history(list(session.get("history") or []), days=days, max_turns=limit)
    out2: list[dict[str, str]] = []
    for turn in trimmed:
        role = str(turn.get("role") or "user")
        if role not in ("user", "assistant", "system"):
            role = "assistant"
        content = str(turn.get("content") or "").strip()
        if content:
            out2.append({"role": role, "content": content[:4000]})
    return out2[-limit:]


def pause_session(company: str, contact: str, hours: float, reason: str) -> None:
    until = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
    save_session(company, contact, paused_until=until, paused_reason=reason)


def resume_session(company: str, contact: str) -> None:
    save_session(company, contact, paused_until=None, paused_reason="")


def is_paused(session: dict[str, Any]) -> bool:
    until = session.get("paused_until")
    if not until:
        return False
    try:
        return datetime.fromisoformat(str(until)) > datetime.now(timezone.utc)
    except ValueError:
        return False


def list_paused(company: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM sessions WHERE paused_until IS NOT NULL"
    params: tuple[Any, ...] = ()
    if company:
        query += " AND company=?"
        params = (company,)
    with _lock, connect() as conn:
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    return [r for r in rows if is_paused(r)]


# ---------------------------------------------------------------- meetings

def add_meeting(
    company: str, contact: str, client_alias: str, client_name: str,
    scheduled_at: str, duration_min: int, topic: str,
) -> int:
    with _lock, connect() as conn:
        cur = conn.execute(
            "INSERT INTO meetings (company, contact, client_alias, client_name,"
            " scheduled_at, duration_min, topic, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (company, contact, client_alias, client_name, scheduled_at, duration_min, topic, _now()),
        )
        return int(cur.lastrowid)


def upcoming_meetings(company: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM meetings WHERE status != 'cancelada' AND scheduled_at >= ?"
    params: list[Any] = [datetime.now(timezone.utc).isoformat()]
    if company:
        query += " AND company=?"
        params.append(company)
    query += " ORDER BY scheduled_at"
    with _lock, connect() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def latest_meeting_for_contact(
    company: str,
    contact: str,
    *,
    include_past_hours: int = 72,
) -> dict[str, Any] | None:
    """Latest non-cancelled meeting for a contact (upcoming or recent past)."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=max(1, include_past_hours))
    ).isoformat()
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT * FROM meetings WHERE company=? AND contact=?"
            " AND status != 'cancelada' AND scheduled_at >= ?"
            " ORDER BY scheduled_at DESC LIMIT 1",
            (company, contact, cutoff),
        ).fetchone()
    return dict(row) if row else None


def upcoming_meeting_for_contact(company: str, contact: str) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT * FROM meetings WHERE company=? AND contact=?"
            " AND status != 'cancelada' AND scheduled_at >= ?"
            " ORDER BY scheduled_at ASC LIMIT 1",
            (company, contact, now),
        ).fetchone()
    return dict(row) if row else None


def meetings_between(company: str, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
    """Confirmed meetings overlapping a window (for free-slot computation)."""
    with _lock, connect() as conn:
        rows = conn.execute(
            "SELECT * FROM meetings WHERE company=? AND status != 'cancelada'"
            " AND scheduled_at >= ? AND scheduled_at < ? ORDER BY scheduled_at",
            (company, start_iso, end_iso),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- requests

def add_request(
    company: str, contact: str, client_alias: str, client_name: str,
    site: str, description: str, priority: str,
) -> int:
    with _lock, connect() as conn:
        cur = conn.execute(
            "INSERT INTO requests (company, contact, client_alias, client_name,"
            " site, description, priority, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (company, contact, client_alias, client_name, site, description, priority, _now()),
        )
        return int(cur.lastrowid)


def list_requests(company: str | None = None, client_alias: str = "", status: str = "") -> list[dict[str, Any]]:
    query = "SELECT * FROM requests WHERE 1=1"
    params: list[Any] = []
    if company:
        query += " AND company=?"
        params.append(company)
    if client_alias:
        query += " AND client_alias=?"
        params.append(client_alias)
    if status:
        query += " AND status=?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    with _lock, connect() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def set_request_status(request_id: int, status: str) -> bool:
    with _lock, connect() as conn:
        cur = conn.execute("UPDATE requests SET status=? WHERE id=?", (status, request_id))
        return cur.rowcount > 0


# ------------------------------------------------------------ sent wamids

def record_sent(wamid: str, company: str, contact: str) -> None:
    if not wamid:
        return
    with _lock, connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sent_messages (wamid, company, contact, created_at) VALUES (?,?,?,?)",
            (wamid, company, contact, _now()),
        )


def was_sent_by_bot(wamid: str) -> bool:
    with _lock, connect() as conn:
        row = conn.execute("SELECT 1 FROM sent_messages WHERE wamid=?", (wamid,)).fetchone()
        return row is not None


def recently_sent(company: str, contact: str, seconds: float) -> bool:
    """
    True if the bot sent *something* to this contact within the last
    `seconds`. Coexistence sometimes echoes back a bot-sent message under a
    different wamid than the one Cloud API returned on send, which would
    otherwise look like a human reply and trigger a false pause. This is a
    time-window fallback for that exact wamid mismatch.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM sent_messages WHERE company=? AND contact=? AND created_at >= ?"
            " ORDER BY created_at DESC LIMIT 1",
            (company, contact, cutoff),
        ).fetchone()
        return row is not None


# -------------------------------------------------------- dedup of events

def mark_processed(wamid: str) -> bool:
    """
    Meta retries webhook deliveries; True = first time we see this wamid,
    False = duplicate (skip). Uses INSERT OR IGNORE for atomicity.
    """
    if not wamid:
        return True
    with _lock, connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO processed_events (wamid, created_at) VALUES (?,?)",
            (wamid, _now()),
        )
        return cur.rowcount > 0


# -------------------------------------------------------- agenda nudges

def upsert_agenda_nudge(
    company: str,
    contact: str,
    *,
    client_name: str = "",
    topic: str = "",
    delay_minutes: int = 20,
) -> dict[str, Any]:
    """Schedule (or reschedule) a reminder if booking was not completed."""
    offered = datetime.now(timezone.utc)
    nudge_at = offered + timedelta(minutes=max(5, int(delay_minutes)))
    row = {
        "company": company,
        "contact": contact,
        "client_name": client_name,
        "topic": topic,
        "offered_at": offered.isoformat(),
        "nudge_at": nudge_at.isoformat(),
        "status": "pending",
    }
    with _lock, connect() as conn:
        conn.execute(
            """
            INSERT INTO agenda_nudges (company, contact, client_name, topic, offered_at, nudge_at, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(company, contact) DO UPDATE SET
                client_name=excluded.client_name,
                topic=excluded.topic,
                offered_at=excluded.offered_at,
                nudge_at=excluded.nudge_at,
                status='pending'
            """,
            (company, contact, client_name, topic, row["offered_at"], row["nudge_at"]),
        )
    return row


def cancel_agenda_nudge(company: str, contact: str) -> None:
    with _lock, connect() as conn:
        conn.execute(
            "UPDATE agenda_nudges SET status='cancelled' WHERE company=? AND contact=? AND status='pending'",
            (company, contact),
        )


def get_agenda_nudge(company: str, contact: str) -> dict[str, Any] | None:
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT * FROM agenda_nudges WHERE company=? AND contact=? AND status='pending'",
            (company, contact),
        ).fetchone()
    return dict(row) if row else None


def mark_agenda_nudge_sent(company: str, contact: str) -> None:
    with _lock, connect() as conn:
        conn.execute(
            "UPDATE agenda_nudges SET status='sent' WHERE company=? AND contact=?",
            (company, contact),
        )


def list_due_agenda_nudges(company: str | None = None) -> list[dict[str, Any]]:
    now = _now()
    query = "SELECT * FROM agenda_nudges WHERE status='pending' AND nudge_at <= ?"
    params: list[Any] = [now]
    if company:
        query += " AND company=?"
        params.append(company)
    with _lock, connect() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def list_pending_agenda_nudges(company: str | None = None) -> list[dict[str, Any]]:
    """All pending nudges (due or still waiting for their timer)."""
    query = "SELECT * FROM agenda_nudges WHERE status='pending'"
    params: list[Any] = []
    if company:
        query += " AND company=?"
        params.append(company)
    with _lock, connect() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def has_meeting_since(company: str, contact: str, since_iso: str) -> bool:
    with _lock, connect() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM meetings
            WHERE company=? AND contact=? AND status != 'cancelada'
              AND created_at >= ?
            LIMIT 1
            """,
            (company, contact, since_iso),
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------- CRM sync (Katana CommercialLead)

def get_crm_sync(company: str, contact: str) -> dict[str, Any]:
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT * FROM crm_sync WHERE company=? AND contact=?",
            (company, contact),
        ).fetchone()
    if row is None:
        return {
            "company": company,
            "contact": contact,
            "commercial_lead_id": None,
            "crm_synced_at": None,
            "crm_pending": 0,
            "crm_attempts": 0,
        }
    return dict(row)


def mark_crm_synced(company: str, contact: str, commercial_lead_id: int | None) -> None:
    with _lock, connect() as conn:
        conn.execute(
            """
            INSERT INTO crm_sync (company, contact, commercial_lead_id, crm_synced_at, crm_pending, crm_attempts)
            VALUES (?, ?, ?, ?, 0, 0)
            ON CONFLICT(company, contact) DO UPDATE SET
                commercial_lead_id=excluded.commercial_lead_id,
                crm_synced_at=excluded.crm_synced_at,
                crm_pending=0
            """,
            (company, contact, commercial_lead_id, _now()),
        )


def mark_crm_pending(company: str, contact: str) -> None:
    with _lock, connect() as conn:
        conn.execute(
            """
            INSERT INTO crm_sync (company, contact, crm_pending, crm_attempts)
            VALUES (?, ?, 1, 1)
            ON CONFLICT(company, contact) DO UPDATE SET
                crm_pending=1,
                crm_attempts=crm_attempts + 1
            """,
            (company, contact),
        )


def begin_crm_upsert(company: str, contact: str) -> tuple[bool, bool]:
    """
    Reserve a CRM upsert slot (anti-race between parallel webhooks).

    Returns (proceed, is_first_contact):
      proceed=False — lead already synced or another thread is upserting.
      is_first_contact=True — notify Katana only on genuine first capture.
    """
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT commercial_lead_id, crm_pending FROM crm_sync WHERE company=? AND contact=?",
            (company, contact),
        ).fetchone()
        if row is not None:
            if row["commercial_lead_id"]:
                return False, False
            if int(row["crm_pending"] or 0) == 2:
                return False, False
        is_first_contact = row is None
        conn.execute(
            """
            INSERT INTO crm_sync (company, contact, crm_pending, crm_attempts)
            VALUES (?, ?, 2, 0)
            ON CONFLICT(company, contact) DO UPDATE SET
                crm_pending=2
            """,
            (company, contact),
        )
        return True, is_first_contact


def list_crm_pending(company: str, *, limit: int = 50) -> list[dict[str, Any]]:
    with _lock, connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM crm_sync
            WHERE company=? AND crm_pending=1
            ORDER BY crm_attempts ASC, contact ASC
            LIMIT ?
            """,
            (company, max(1, limit)),
        ).fetchall()
    return [dict(r) for r in rows]

