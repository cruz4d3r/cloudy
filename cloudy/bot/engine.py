# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Conversation engine: everything between an inbound webhook event and the
reply sent through the Cloud API.

Flow per inbound message:
  dedup -> owner commands -> client authorization -> human-pause check ->
  pending state machine (slot pick / confirmations) -> intent routing
  (agendar | solicitud_cambio | consulta | humano) -> reply + persistence.

Human takeover: handle_outbound_echo() receives every outbound message echo
from Meta; if its wamid is not in our sent_messages registry, a human wrote
it from the phone/WhatsApp Web and the conversation is paused automatically.
"""
from __future__ import annotations

import logging
import re
import threading
import unicodedata
from datetime import datetime, timezone
from typing import Any

from cloudy.bot import agenda_nudge, meeting_awareness, scheduler, store
from cloudy.bot.config import is_owner, normalize_number, resolve_client
from cloudy.bot.llm import LLMError, chat, classify_json, pop_chat_meta
from cloudy.bot.wa_client import WhatsAppError, notify_owner, send_text

logger = logging.getLogger("cloudy.bot")

# Words that force an immediate handoff to a human, no LLM involved.
_HANDOFF_WORDS = ("asesor", "humano", "persona real", "hablar con sergio")

# Default phrases Sergio can type INTO a client chat (from his phone or
# WhatsApp Web) to silence the bot in that single conversation for a full day.
# Matched accent-insensitively as a substring, so a natural greeting like
# "Hola, te saluda Sergio Martínez" also triggers it. Overridable per company
# via the config key "takeover_keywords".
_TAKEOVER_WORDS = (
    "sergio martinez",
    "te saluda sergio",
    "voy a tomar el chat",
    "tomo el chat",
    "yo sigo desde aca",
    "yo continuo por aqui",
)

# Pause applied when a takeover keyword is detected (hours ~= 1 day).
# Overridable per company via the config key "takeover_pause_hours".
_TAKEOVER_PAUSE_HOURS = 24.0

# Appended to every LLM system prompt that talks to clients/leads.
_CLIENT_VOICE_GUARDRAIL = """
TONO INNEGOCIABLE (clientes y leads):
- Habla como Sergio en WhatsApp real: cercano, cálido, humano; tú/usted, NUNCA tercera persona.
- PROHIBIDO sonar a máquina, ticket o nota interna: "el cliente aclara/indica/menciona",
  "ya lo anoté", "quedó registrado", "en cola", "procesando su solicitud",
  "¿en qué puedo asistirte?", "Sergio lo revisa y te escribe".
- No resumas al cliente lo que él mismo acaba de decir como si fueras un informe.
- Máximo 3-4 frases cortas; una sola pregunta si hace falta.
"""

def _sync_contact_country(
    company_alias: str,
    contact: str,
    session: dict[str, Any],
    text: str,
) -> Any:
    """Resolve market from phone/text/session and persist on the WA session."""
    from cloudy.bot.contact_country import resolve_country, session_country_fields

    user_msgs = store.count_user_messages(company_alias, contact)
    resolved = resolve_country(contact, session, text, user_message_count=user_msgs)
    fields = session_country_fields(resolved)
    if (
        str(session.get("contact_country") or "") != fields["contact_country"]
        or str(session.get("country_source") or "") != fields["country_source"]
    ):
        store.save_session(company_alias, contact, **fields)
        session.update(fields)
    return resolved


def _finalize_reply_country(
    company_alias: str,
    contact: str,
    session: dict[str, Any],
    reply: str,
    text: str,
) -> str:
    """Append one-time country question when prefix is ambiguous."""
    from cloudy.bot.contact_country import append_country_ask, resolve_country

    user_msgs = store.count_user_messages(company_alias, contact)
    resolved = resolve_country(contact, session, text, user_message_count=user_msgs)
    out = append_country_ask(reply, resolved)
    if resolved.should_ask and out != reply and not int(session.get("country_asked") or 0):
        store.save_session(company_alias, contact, country_asked=1)
    return out

# Outbound gate: if the text matches, never send it — use warm fallback instead.
_MACHINE_REPLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(el|la)\s+cliente\b", re.I),
    re.compile(r"\b(el\s+contacto|el\s+usuario|el\s+lead)\b", re.I),
    re.compile(r"\b(aclara|indica|menciona|explica|solicita|comenta)\s+que\b", re.I),
    re.compile(r"\bya\s+lo\s+anot[eé]\b", re.I),
    re.compile(r"\bqued[oó]\s+registrad[oa]\b", re.I),
    re.compile(r"\bnota\s+interna\b", re.I),
    re.compile(r"\ben\s+cola\b", re.I),
    re.compile(r"\bprocesando\s+su\s+solicitud\b", re.I),
    re.compile(r"\b¿en\s+qué\s+puedo\s+asist", re.I),
    re.compile(r"\blo\s+revisa\s+y\s+te\s+escribe\b", re.I),
    re.compile(r"\bcomo\s+asistente\b", re.I),
    re.compile(r"\bsu\s+solicitud\s+ha\s+sido\b", re.I),
)


def _strip_accents(text: str) -> str:
    """Lowercase + drop diacritics so keyword matching ignores accents/case."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()

# Grace window after a bot-sent message during which an unmatched echo is
# treated as a delayed/renamed copy of our own message, not a human reply.
# Coexistence (phone app + Cloud API on the same number) can echo back a
# Cloud-API-sent message under a *different* wamid than the one returned by
# the /messages call, which would otherwise look exactly like a human takeover.
_ECHO_GRACE_SECONDS = 30.0


def _detect_inbound_attribution(text: str) -> dict[str, Any]:
    """Infer UTM-ish source from first WA message (ads, email, Meta)."""
    t = (text or "").lower()
    out: dict[str, Any] = {
        "utm_source": "",
        "utm_medium": "whatsapp",
        "utm_campaign": "",
        "utm_content": "",
        "interested_services": None,
    }
    if any(w in t for w in ("facebook", "instagram", "meta", "anuncio", "publicidad", "pauta", "ads")):
        out["utm_source"] = "meta"
        out["utm_medium"] = "paid_social"
        out["utm_campaign"] = "meta_ads_inbound"
        out["interested_services"] = ["Proyecto desde Meta Ads"]
    elif any(w in t for w in ("correo", "email", "mail", "vi el correo")):
        out["utm_source"] = "cold_email"
        out["utm_medium"] = "email"
        out["utm_campaign"] = "cold_email_reply"
        out["interested_services"] = ["Respuesta correo frío"]
    elif any(w in t for w in ("google", "busqueda", "búsqueda", "anuncio de google")):
        out["utm_source"] = "google"
        out["utm_medium"] = "cpc"
        out["utm_campaign"] = "google_ads_inbound"
        out["interested_services"] = ["Proyecto desde Google Ads"]
    elif "web" in t or "página" in t or "pagina" in t:
        out["interested_services"] = ["Sitio web"]
    elif "tienda" in t or "woocommerce" in t or "ecommerce" in t:
        out["interested_services"] = ["Tienda online"]
    return out


def _enrich_client(
    contact: str,
    client: dict[str, Any] | None,
    *,
    company_alias: str = "",
) -> dict[str, Any]:
    """
    Merge Katana client profile when the number is not in local whatsapp.json.
    Keeps Mac bot aligned with contingency when billing/directory knows the contact.
    """
    if client and not client.get("prospect"):
        return client
    try:
        from cloudy.bot.katana_chat import fetch_client_profile

        profile = fetch_client_profile(contact, company_alias=company_alias)
        if not profile or not profile.get("recognized"):
            return client or {
                "alias": f"prospect_{contact}",
                "name": "Cliente",
                "sites": [],
                "prospect": True,
            }
        return {
            "alias": str(profile.get("alias") or "").strip() or f"client_{contact}",
            "name": str(profile.get("name") or profile.get("client_name") or "Cliente").strip(),
            "sites": list(profile.get("sites") or []),
            "notes": str(profile.get("notes") or "").strip(),
            "prospect": False,
            "from_katana": True,
        }
    except Exception:
        logger.debug("katana client enrich skipped contact=%s", contact, exc_info=True)
        return client or {
            "alias": f"prospect_{contact}",
            "name": "Cliente",
            "sites": [],
            "prospect": True,
        }


_INTENT_SYSTEM = """Eres un clasificador de intenciones para el asistente de WhatsApp de una agencia de desarrollo web.
Clasifica el mensaje del cliente en UNA de estas intenciones:
- "agendar": quiere una reunión, cita, llamada o videollamada, o propone/ajusta horario (almuerzo, noche, tarde, "me retrasé", "te escribo luego").
- "solicitud_cambio": pide un cambio, mejora, ajuste, corrección o funcionalidad nueva en su sitio o proyecto.
- "consulta": pregunta general (precios, servicios, estado, dudas) O cualquier mensaje con contexto/disculpa/logística que no sea solo un hola vacío. Ante duda, usa "consulta".
- "humano": pide explícitamente hablar con una persona.
- "saludo": SOLO si el mensaje es únicamente un saludo o gracias cortos (hola, buenos días, gracias) SIN otro contenido. Si hay historial y el cliente continúa el hilo, NUNCA uses "saludo".

IMPORTANTE — NO es "solicitud_cambio" si el cliente solo:
- aclara cómo funciona algo hoy ("no lo tenemos", "así es el flujo", "ellos solo recogen");
- corrige un malentendido o responde una pregunta tuya sin pedir un cambio nuevo;
- explica limitaciones actuales sin pedir implementar/integrar/solicitar algo.
En esos casos usa "consulta".

Responde SOLO un objeto JSON: {"intent": "...", "topic": "tema breve o vacío", "date_pref": "YYYY-MM-DD si menciona un día concreto, o vacío"}"""

_REQUEST_SYSTEM = """Extrae los datos de una solicitud de cambio/mejora que un cliente envía a su agencia web.
Sitios del cliente (referencia): {sites}

Reglas OBLIGATORIAS:
- Usa SOLO lo que diga el cliente. NO inventes páginas, errores, causa ni alcance.
- Conserva URLs, nombres de menús, textos de error, números de pedido y detalles técnicos.
- Si el cliente mandó varios mensajes o descripciones de imagen/audio, intégralos en UNA descripción fiel.
- "description" es NOTA INTERNA para el equipo (puede ser en tercera persona). NUNCA se reenvía tal cual al cliente.
- "description" debe quedar clara para un desarrollador: qué pasa, dónde, y qué pide (2 a 6 frases).
- "site" solo si el cliente lo indica o coincide inequívocamente con la lista; si no, vacío.

Responde SOLO un objeto JSON:
{{"site": "sitio afectado o vacío",
  "description": "descripción fiel y completa del cambio pedido",
  "priority": "alta | normal | baja (alta solo si está caído, no vende, pérdida de dinero o bloquea operación)"}}"""

# Cliente suele mandar la solicitud en varios WhatsApp seguidos. Esperamos esta
# ventana de SILENCIO (se reinicia con cada mensaje nuevo) antes de responder.
_GATHER_SECONDS = 60.0
# Saludos cortos en ráfaga ("Hola Sergio" + "¿Cómo estás?"): ventana corta para unir burbujas.
_GREETING_GATHER_SECONDS = 10.0
# Si ya respondimos hace poco, no volver a saludar por un hola suelto.
_GREETING_COOLDOWN_SECONDS = 120.0
_REQUEST_HINT_RE = re.compile(
    r"(cambi|arreg|mejor|ajust|correg|agreg|quit|error|bug|ca[ií]d|no\s+carga|"
    r"no\s+funciona|roto|falla|solicitud|pantalla|formulario|pasarela|pago|"
    r"whatsapp|m[oó]dulo|imagen|t[ií]tulo|secci[oó]n|men[uú]|bot[oó]n|"
    r"raro|rar[ao]|se\s+ve|mostrar|visual)",
    re.IGNORECASE,
)
_CLARIFICATION_RE = re.compile(
    r"\b(no\s+lo\s+tenemos|no\s+tenemos|as[ií]\s+(es|funciona)|solo\s+recogen|"
    r"para\s+aclarar|el\s+flujo\s+es|actualmente\s+no|no\s+manejamos|"
    r"no\s+hay\s+integraci[oó]n|nos\s+pasan\s+la\s+gu[ií]a)\b",
    re.I,
)
_CHANGE_ASK_RE = re.compile(
    r"\b(solicit|implement|integr|agreg|necesitamos\s+que|nos\s+colabore|"
    r"pueden\s+hacer|háganlo|haganlo|por\s+favor\s+(agreg|hag|implement))\b",
    re.I,
)

_EMAIL_FROM_RE = re.compile(
    r"\b(correo|email|mail|me escribieron|vi el mail)\b", re.I,
)
_EMAIL_CAMPAIGN_RE = re.compile(
    r"\b(landing|200\.?000|200\s*mil|armar una landing|dominio|mejorar mi web|"
    r"p[aá]gina web|nit)\b",
    re.I,
)
_EMAIL_INTENT_RE = re.compile(
    r"\b(agendar|llamada|cita|reuni[oó]n|hablar)\b", re.I,
)


def _is_email_landing_lead(text: str) -> bool:
    """Lead caliente desde campaña correo AMB (prefill WA con NIT / landing $200k)."""
    t = (text or "").strip()
    if not t:
        return False
    from_mail = bool(_EMAIL_FROM_RE.search(t))
    campaign = bool(_EMAIL_CAMPAIGN_RE.search(t))
    intent = bool(_EMAIL_INTENT_RE.search(t))
    prefill = "soy " in t.lower() and "nit" in t.lower()
    return (from_mail or prefill) and (campaign or intent)


_COMMERCIAL_NEED_RE = re.compile(
    r"\b(web|landing|tienda|app|marketing|publicidad|hosting|dominio|"
    r"200\.?000|200\s*mil|correo|promoci[oó]n|quiero\s+una|necesito\s+una|"
    r"p[aá]gina|nit|agendar|llamada|cita|reuni[oó]n)\b",
    re.I,
)

_FAKE_BOOKING_RE = re.compile(
    r"\b(tu\s+reuni[oó]n\s+con\s+unlockers|invitaci[oó]n\s+de\s+calendario|"
    r"reenv[ií]o\s+la\s+invitaci[oó]n)\b",
    re.I,
)


def _is_commercial_lead(text: str, client: dict[str, Any]) -> bool:
    """Prospect with explicit commercial intent (broader than email-landing only)."""
    if not client.get("prospect"):
        return False
    t = (text or "").strip()
    if not t:
        return False
    if _is_email_landing_lead(t):
        return True
    return bool(_COMMERCIAL_NEED_RE.search(t))


def _looks_like_fake_booking_confirmation(text: str) -> bool:
    """LLM hallucination: confirms meeting without scheduler/booking backend."""
    reply = (text or "").strip()
    if not reply:
        return False
    # Legitimate chat scheduler confirmation (store.add_meeting + notify_owner).
    if re.search(r"qued[oó]\s+agendada\s+tu\s+reuni[oó]n", reply, re.I):
        return False
    if _FAKE_BOOKING_RE.search(reply):
        return True
    if re.search(r"qued[oó]\s+agendad", reply, re.I) and "unlockers cloud" in reply.lower():
        return True
    return False


def _lead_first_name(text: str, client: dict[str, Any]) -> str:
    m = re.search(r"(?:soy|me llamo)\s+([^,(]+)", text, re.I)
    if m:
        parts = m.group(1).strip().split()
        if parts:
            return parts[0].title()
    name = str(client.get("name") or "").strip()
    if name and name not in ("Cliente", "Cliente WhatsApp"):
        return name.split()[0].title()
    return ""


def _email_landing_hook(text: str) -> str:
    del text
    return ""


def _warmup_first_reply(text: str, client: dict[str, Any], *, from_email: bool = False) -> str:
    """Primer mensaje a prospecto: empatía y conversación, sin agenda."""
    first = _lead_first_name(text, client)
    if first:
        line1 = f"¡Hola {first}! Te saluda Unlockers Cloud."
    else:
        line1 = "¡Hola! Te saluda Unlockers Cloud."
    if from_email:
        line2 = "Vimos que te interesó lo del correo. ¿Cómo estás?"
    else:
        line2 = "Qué bueno que escribiste. ¿Cómo estás?"
    return (
        f"{line1}\n\n"
        f"{line2}\n\n"
        "Cuéntanos un poquito de tu proyecto antes de agendar — ¿qué tienes en mente?"
    )


def _email_landing_first_reply(text: str, client: dict[str, Any]) -> str:
    return _warmup_first_reply(text, client, from_email=True)


def _arm_prospect_warmup(
    company_alias: str,
    contact: str,
    *,
    topic: str = "Lead comercial",
) -> None:
    store.save_session(
        company_alias,
        contact,
        pending={
            "flow": "calificar",
            "step": "warmup",
            "topic": topic,
        },
    )


# Timers: una sola ventana por (empresa, contacto) desde el primer mensaje del lote.
_gather_timers: dict[str, threading.Timer] = {}
_gather_lock = threading.Lock()
_inflight_lock = threading.Lock()
_inflight_turns: set[str] = set()


# ------------------------------------------------------------------ helpers

def _reply(company_alias: str, company: dict[str, Any], contact: str, body: str) -> None:
    """Send + record wamid so echo detection can recognize our own messages."""
    wamid = send_text(company, contact, body)
    store.record_sent(wamid, company_alias, contact)


def _push_history(
    session: dict[str, Any],
    role: str,
    text: str,
    *,
    kind: str = "text",
) -> list[dict[str, Any]]:
    from cloudy.bot.store import trim_history

    history = list(session.get("history") or [])
    history.append({
        "role": role,
        "content": text[:4000],
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind or "text",
    })
    return trim_history(history)


def _infer_msg_kind(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith("[audio") or "nota de voz" in t[:100]:
        return "audio"
    if t.startswith(("[imagen", "[video", "[documento", "[sticker", "[ubicacion")):
        return "media"
    if t.startswith("recibí tu nota") or t.startswith("recibí tu archivo") or t.startswith("[medio "):
        return "media_fail"
    return "text"


def _is_katana_media_placeholder(text: str) -> bool:
    """Katana log lines like ``[Media audio]`` without a real attachment."""
    import re

    t = (text or "").strip()
    if not t:
        return True
    return bool(re.match(r"^\[Media\s+\w+\]$", t, re.IGNORECASE))


def _is_media_fail_message(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return (
        t.startswith("recibí tu nota de voz")
        or t.startswith("recibí tu archivo")
        or t.startswith("[medio ")
        or t.startswith("[audio] (no pude")
        or t.startswith("[imagen] (imagen recibida")
        or "no la pude escuchar" in t
        or "no pude transcribir" in t
        or "no pude abrirlo" in t
        or "no pude analizarla" in t
    )


def _history_for_llm(
    company_alias: str,
    contact: str,
    session: dict[str, Any],
    limit: int = 100,
) -> list[dict[str, str]]:
    """Last ~7 days of turns (message_log), fallback to session JSON."""
    turns = store.recent_turns(company_alias, contact, days=7, limit=limit)
    if turns:
        return turns
    # Legacy fallback without company log.
    history = list(session.get("history") or [])[-limit:]
    out: list[dict[str, str]] = []
    for turn in history:
        role = turn.get("role") or "user"
        if role not in ("user", "assistant", "system"):
            role = "assistant"
        content = str(turn.get("content") or "").strip()
        if content:
            out.append({"role": role, "content": content})
    return out


def _history_snippet(
    company_alias: str,
    contact: str,
    session: dict[str, Any],
    owner_name: str,
    limit: int = 40,
) -> str:
    """Human-readable recent dialogue for intent classification."""
    turns = _history_for_llm(company_alias, contact, session, limit=limit)
    lines: list[str] = []
    for turn in turns:
        role = turn.get("role")
        who = "Cliente" if role == "user" else owner_name
        content = str(turn.get("content") or "").strip()
        if content:
            lines.append(f"{who}: {content[:400]}")
    return "\n".join(lines)


def _is_yes(text: str) -> bool:
    return bool(re.match(r"^\s*(s[ií]|sí,|si,|claro|dale|listo|ok|correcto|confirmo|perfecto|de acuerdo)\b", text.strip().lower()))


def _is_no(text: str) -> bool:
    return bool(re.match(r"^\s*(no|nel|mejor no|cancela|cancelar|olvida)\b", text.strip().lower()))


def _history_has_assistant(
    session: dict[str, Any],
    company_alias: str = "",
    contact: str = "",
) -> bool:
    """True if we already replied in this conversation (anti greeting-loop)."""
    if any(
        (turn.get("role") == "assistant") and str(turn.get("content") or "").strip()
        for turn in (session.get("history") or [])
    ):
        return True
    if company_alias and contact:
        return any(
            str(turn.get("role") or "") == "assistant"
            for turn in store.recent_turns(company_alias, contact, days=7, limit=20)
        )
    return False


_BARE_GREETING_RE = re.compile(
    r"^\s*("
    r"h+ola+|holi|hey|"
    r"buenos?\s+d[ií]as?|buenas?\s+(tardes?|noches?)|buenas?|"
    r"buen\s+d[ií]a|qu[eé]\s+tal|c[oó]mo\s+est[aá]s?|"
    r"gracias(\s+mil)?|mil\s+gracias|ok\s*gracias|"
    r"saludos?"
    r")[\s!.¡?¿]*$",
    re.IGNORECASE,
)

# Closing thanks that often include the owner's first name ("listo sergio gracias").
_THANKS_CORE_RE = re.compile(
    r"^\s*("
    r"(listo|ok|vale|perfecto|de\s+acuerdo|claro)?\s*"
    r"(gracias(\s+mil)?|mil\s+gracias|gracias\s+mil)"
    r"|"
    r"gracias(\s+mil)?|mil\s+gracias"
    r")[\s!.¡?¿]*$",
    re.IGNORECASE,
)


def _is_bare_greeting(text: str) -> bool:
    """Only true for short hello/thanks with no other content."""
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) > 60:
        return False
    return bool(_BARE_GREETING_RE.match(cleaned))


def _strip_owner_name_mentions(text: str, owner_name: str) -> str:
    """Remove owner first-name mentions so we can classify thanks/greetings."""
    cleaned = (text or "").strip()
    owner = (owner_name or "").strip()
    if owner:
        cleaned = re.sub(rf"\b{re.escape(owner)}\b", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" ,.-")


def _is_closing_thanks(text: str, owner_name: str = "Sergio") -> bool:
    """
    True for short closings like 'gracias', 'listo sergio gracias'.
    The client is thanking the owner — never treat that name as the client's.
    """
    cleaned = _strip_owner_name_mentions(text, owner_name)
    if not cleaned or len(cleaned) > 80:
        return False
    return bool(_THANKS_CORE_RE.match(cleaned))


# ------------------------------------------------------- owner bot commands

def _handle_owner_command(company_alias: str, company: dict[str, Any], text: str) -> str | None:
    """
    Control commands from the owner's own WhatsApp:
      #bot off <numero>  - pause the bot for that conversation
      #bot on <numero>   - resume it
      #bot status        - list paused conversations
    Returns the reply text, or None if the message is not a command.
    """
    lowered = text.strip().lower()
    if not lowered.startswith("#bot"):
        return None
    parts = lowered.split()
    action = parts[1] if len(parts) > 1 else "status"
    number = normalize_number(parts[2]) if len(parts) > 2 else ""

    if action == "off" and number:
        store.pause_session(company_alias, number, hours=24 * 365, reason="manual (#bot off)")
        return f"Bot pausado para {number}. Reanuda con: #bot on {number}"
    if action == "on" and number:
        store.resume_session(company_alias, number)
        return f"Bot reactivado para {number}."
    if action == "status":
        paused = store.list_paused(company_alias)
        if not paused:
            return "Bot activo en todas las conversaciones."
        lines = ["Conversaciones pausadas:"]
        for row in paused:
            lines.append(f"- {row['contact']} ({row.get('paused_reason') or 'sin motivo'})")
        return "\n".join(lines)
    return "Comandos: #bot off <numero> | #bot on <numero> | #bot status"


# --------------------------------------------------------- scheduling flow

_CHAT_AGENDA_RE = re.compile(
    r"\b("
    r"por\s+chat|en\s+(el\s+)?chat|aqu[ií]\s+mismo|desde\s+(el\s+)?chat|"
    r"por\s+aqu[ií]|por\s+wa|por\s+whatsapp|armamos\s+aqu[ií]|lo\s+hago\s+aqu[ií]|"
    r"prefiero\s+(chat|aqu[ií]|whatsapp)|sin\s+enlace|no\s+enlace"
    r")\b",
    re.IGNORECASE,
)
_LINK_AGENDA_RE = re.compile(
    r"\b("
    r"enlace|link|por\s+(el\s+)?link|desde\s+(el\s+)?enlace|en\s+la\s+p[aá]gina|"
    r"agendar\.|1lockers\.net/agendar|prefiero\s+(el\s+)?enlace|yo\s+agendo"
    r")\b",
    re.IGNORECASE,
)


def _booking_offered_at(pending: dict[str, Any]) -> str:
    return str(pending.get("offered_at") or pending.get("date_pref") or "").strip()


def _recent_web_booking_ack(
    company_alias: str,
    contact: str,
    client: dict[str, Any],
    pending: dict[str, Any] | None = None,
) -> str | None:
    """Si ya agendaron por /agendar, confirmar y cerrar el flujo (no volver a ofrecer)."""
    since = ""
    if pending:
        since = _booking_offered_at(pending)
    if not since:
        try:
            from datetime import timedelta

            since = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        except Exception:
            since = ""
    try:
        from cloudy.bot.katana_chat import fetch_latest_booking

        booking = fetch_latest_booking(contact, since or None)
    except Exception:
        booking = None
    if not booking:
        return None

    agenda_nudge.cancel(company_alias, contact)
    store.save_session(company_alias, contact, pending={})

    name = str(booking.get("contact_name") or client.get("name") or "").strip()
    first = (name.split() or ["hola"])[0]
    when = str(booking.get("starts_at_local") or "").strip() or "el horario que elegiste"
    return (
        f"¡Listo {first}! Ya vi tu cita confirmada para el {when}. "
        "Si necesitas moverla, dime y la ajustamos."
    )


def _offer_scheduling_channels(
    company_alias: str, company: dict[str, Any], contact: str,
    client: dict[str, Any], topic: str, date_pref: str,
) -> str:
    """Human first step: enlace self-serve O continuar en el mismo chat."""
    booking_url = str(company.get("appointment_booking_url") or "").strip()
    offered_at = datetime.now(timezone.utc).isoformat()
    store.save_session(
        company_alias, contact,
        pending={
            "flow": "agendar",
            "step": "choose_channel",
            "topic": topic,
            "date_pref": date_pref or "",
            "offered_at": offered_at,
        },
    )
    notify_owner(
        company,
        f"[AGENDA] {client.get('name')} ({contact}) quiere reunión; "
        f"se ofreció enlace y/o chat. Tema: {topic or '(sin tema)'}",
    )
    if booking_url:
        agenda_nudge.schedule(
            company_alias, company, contact,
            client_name=str(client.get("name") or ""),
            topic=topic,
        )
        return (
            f"Listo. ¿Te sirve una llamada corta para verlo bien?\n{booking_url}\n\n"
            'O escribe "por chat" y te paso horarios por aquí.'
        )
    return (
        "Dale, agendamos por aquí. Dime un día que te sirva "
        "(hábil, en la mañana) y te paso opciones."
    )


def _start_scheduling_chat_slots(
    company_alias: str, company: dict[str, Any],
    contact: str, client: dict[str, Any], topic: str, date_pref: str,
) -> str:
    """Offer concrete slots in-chat (SQLite agenda)."""
    preferred = None
    if date_pref:
        try:
            preferred = datetime.fromisoformat(date_pref).replace(tzinfo=scheduler.company_tz(company))
        except ValueError:
            preferred = None
    slots = scheduler.free_slots(company_alias, company, limit=3, preferred_day=preferred)
    if not slots:
        notify_owner(company, f"[AGENDA] {client['name']} pidió reunión por chat y no hay slots libres en 10 días.")
        booking_url = str(company.get("appointment_booking_url") or "").strip()
        if booking_url:
            store.save_session(company_alias, contact, pending={})
            return (
                "Por chat no me quedan huecos claros en estos días. "
                f"Agenda aquí el que te sirva: {booking_url}"
            )
        return (
            "Por ahora no tengo horarios disponibles en los próximos días. "
            f"Ya le avisé a {company.get('owner_name') or 'nuestro equipo'} para coordinar contigo."
        )
    store.save_session(
        company_alias, contact,
        pending={
            "flow": "agendar",
            "step": "pick_slot",
            "topic": topic,
            "slots": [s.isoformat() for s in slots],
        },
    )
    lines = ["Listo, por chat. Tengo estos horarios:"]
    for index, slot in enumerate(slots, start=1):
        lines.append(f"{index}. {scheduler.format_slot(slot)}")
    lines.append("Respóndeme con el número (1, 2 o 3) o dime otro día.")
    return "\n".join(lines)


def _start_scheduling(
    company_alias: str, company: dict[str, Any], session: dict[str, Any],
    contact: str, client: dict[str, Any], topic: str, date_pref: str,
) -> str:
    booking_url = str(company.get("appointment_booking_url") or "").strip()
    # With a public booking URL: always offer enlace OR chat (intuitive choice).
    if booking_url:
        return _offer_scheduling_channels(
            company_alias, company, contact, client, topic, date_pref,
        )
    return _start_scheduling_chat_slots(
        company_alias, company, contact, client, topic, date_pref,
    )


def _continue_scheduling(
    company_alias: str, company: dict[str, Any], session: dict[str, Any],
    contact: str, client: dict[str, Any], text: str,
) -> str:
    pending = session.get("pending") or {}
    step = str(pending.get("step") or "")
    topic = str(pending.get("topic") or "Reunión de seguimiento")
    date_pref = str(pending.get("date_pref") or "")
    booking_url = str(company.get("appointment_booking_url") or "").strip()

    if _is_no(text):
        agenda_nudge.cancel(company_alias, contact)
        store.save_session(company_alias, contact, pending={})
        return "Sin problema, no agendamos por ahora. ¿Te ayudo con algo más?"

    ack = _recent_web_booking_ack(company_alias, contact, client, pending)
    if ack is not None:
        return ack

    if step == "chat_slot_proposed":
        proposed_iso = str(pending.get("proposed_slot") or "").strip()
        if proposed_iso and _is_no(text):
            agenda_nudge.cancel(company_alias, contact)
            store.save_session(company_alias, contact, pending={})
            return "Sin problema, no agendamos por ahora. ¿Te ayudo con algo más?"
        if proposed_iso and meeting_awareness.user_confirms_scheduling(text):
            try:
                slot = datetime.fromisoformat(proposed_iso)
            except ValueError:
                slot = None
            if slot is not None:
                tz = scheduler.company_tz(company)
                slot_local = slot.astimezone(tz)
                meeting_id = store.add_meeting(
                    company_alias, contact, client.get("alias", ""), client.get("name", ""),
                    slot_local.isoformat(), scheduler.slot_minutes(company), topic,
                )
                agenda_nudge.cancel(company_alias, contact)
                store.save_session(company_alias, contact, pending={})
                notify_owner(
                    company,
                    f"[REUNIÓN] #{meeting_id} confirmada por chat\n"
                    f"Cliente: {client.get('name')} ({contact})\n"
                    f"Cuándo: {scheduler.format_slot(slot_local)}\n"
                    f"Tema: {topic}",
                )
                return (
                    f"Listo, quedó agendada tu reunión para el {scheduler.format_slot(slot_local)}. "
                    "Te escribo por aquí si hace falta."
                )
        return "¿Te sirve ese horario? Respóndeme sí o dime otro que te quede mejor."

    if step == "awaiting_link_booking":
        if booking_url:
            return (
                f"Cuando quieras, entra al enlace y eliges horario:\n{booking_url}\n\n"
                "En cuanto confirmes ahí te llega el WhatsApp con la cita."
            )
        return "Avísame cuando hayas podido agendar o si prefieres que lo armemos por chat."

    if step == "choose_channel":
        if _CHAT_AGENDA_RE.search(text) or (
            not _LINK_AGENDA_RE.search(text)
            and re.search(r"\b(chat|aqu[ií]|whatsapp)\b", text, re.IGNORECASE)
        ):
            return _start_scheduling_chat_slots(
                company_alias, company, contact, client, topic, date_pref,
            )
        if _LINK_AGENDA_RE.search(text) or (booking_url and booking_url in text):
            store.save_session(
                company_alias, contact,
                pending={
                    "flow": "agendar",
                    "step": "awaiting_link_booking",
                    "topic": topic,
                    "date_pref": date_pref or "",
                    "offered_at": str(pending.get("offered_at") or datetime.now(timezone.utc).isoformat()),
                },
            )
            return (
                f"Perfecto. Entra aquí y elige el horario que te sirva:\n{booking_url}\n\n"
                "Cuando confirmes ahí, te llega la confirmación por WhatsApp "
                "y la invitación al correo (calendario). ¡Quedamos pendientes!"
            )
        # Day preference → go to chat slots
        try:
            parsed = classify_json(_INTENT_SYSTEM, text)
            maybe_date = str(parsed.get("date_pref") or "")
        except LLMError:
            maybe_date = ""
        if maybe_date:
            return _start_scheduling_chat_slots(
                company_alias, company, contact, client, topic, maybe_date,
            )
        if booking_url:
            return (
                f"Como prefieras: enlace {booking_url} "
                "o escribe \"por chat\" y lo armamos aquí."
            )
        return "Dime un día que te sirva y te paso opciones, o escribe \"por chat\"."

    slots = [datetime.fromisoformat(s) for s in pending.get("slots") or []]
    choice = re.match(r"^\s*([1-9])\b", text.strip())

    if choice and 0 < int(choice.group(1)) <= len(slots):
        slot = slots[int(choice.group(1)) - 1]
        meeting_id = store.add_meeting(
            company_alias, contact, client.get("alias", ""), client.get("name", ""),
            slot.isoformat(), scheduler.slot_minutes(company), topic,
        )
        agenda_nudge.cancel(company_alias, contact)
        store.save_session(company_alias, contact, pending={})
        notify_owner(
            company,
            f"[REUNIÓN] #{meeting_id} confirmada\n"
            f"Cliente: {client.get('name')} ({contact})\n"
            f"Cuándo: {scheduler.format_slot(slot)}\n"
            f"Tema: {topic}",
        )
        return (
            f"Listo, quedó agendada tu reunión para el {scheduler.format_slot(slot)}. "
            f"Tema: {topic}. Te escribo por aquí si hace falta. ¡Gracias!"
        )

    # Neither a number nor a no: maybe they proposed another day.
    try:
        parsed = classify_json(_INTENT_SYSTEM, text)
        new_date = str(parsed.get("date_pref") or "")
    except LLMError:
        new_date = ""
    if new_date:
        return _start_scheduling_chat_slots(
            company_alias, company, contact, client, topic, new_date,
        )
    return (
        "No te entendí. Respóndeme con el número de la opción (1, 2 o 3), "
        "propón otro día, o dime 'no' para cancelar."
    )

# ----------------------------------------------------- change-request flow

def _looks_like_request_burst(text: str) -> bool:
    """Heuristic: client is dictating a change request (possibly across msgs)."""
    t = (text or "").strip()
    if not t:
        return False
    # Never treat media/system tags as a "solicitud" burst (was matching [Audio]… → 60s ACK).
    low = t.lower()
    if low.startswith(("[medio ", "[audio", "[imagen", "[video", "[sticker", "[documento", "[ubicacion")):
        return False
    if _REQUEST_HINT_RE.search(t):
        return True
    # Long pasted briefs / screenshots often arrive as the real request body.
    return len(t) >= 180


def _looks_like_clarification_only(text: str) -> bool:
    """Cliente aclara flujo/limitación sin pedir cambio — no es ticket."""
    t = (text or "").strip()
    if not t:
        return False
    if _CHANGE_ASK_RE.search(t):
        return False
    return bool(_CLARIFICATION_RE.search(t))


def _client_first_name(client: dict[str, Any]) -> str:
    name = str(client.get("name") or "").strip()
    if not name or name in ("Cliente", "Cliente WhatsApp"):
        return ""
    return name.split()[0]


def _looks_like_machine_reply(text: str) -> bool:
    """True if outbound text sounds like internal notes or call-center bot."""
    t = (text or "").strip()
    if not t:
        return True
    return any(p.search(t) for p in _MACHINE_REPLY_PATTERNS)


def _fallback_warm_reply(
    client: dict[str, Any],
    company: dict[str, Any],
    *,
    context: str = "general",
) -> str:
    """Safe human fallback when LLM or templates slip into machine tone."""
    first = _client_first_name(client)
    name_bit = f" {first}" if first else ""
    if context == "request":
        return (
            f"Dale{name_bit}, ya lo tengo. Lo revisamos y te escribo por aquí.\n\n"
            f"Si me faltó algo, me lo dices sin pena."
        )
    return f"Claro{name_bit}, dame un momentico y te confirmo por aquí."


def _polish_client_reply(
    text: str,
    client: dict[str, Any],
    company: dict[str, Any],
    *,
    context: str = "general",
    meeting_state: dict[str, Any] | None = None,
) -> str:
    """Last gate before WhatsApp: block machine/internal tone to clients and leads."""
    reply = (text or "").strip()
    if context != "booking_confirmed" and _looks_like_fake_booking_confirmation(reply):
        logger.warning("reply bloqueado (cita fantasma sin booking): %s", reply[:160])
        booking_url = str(company.get("appointment_booking_url") or "").strip()
        if booking_url:
            return (
                "Para dejar la reunión bien registrada y que te llegue el recordatorio, "
                f"elige el horario aquí:\n{booking_url}\n\n"
                "O escríbeme \"por chat\" y te paso opciones."
            )
        return (
            "Dame un momentico — te paso opciones de horario por aquí en un segundo."
        )
    if _looks_like_machine_reply(reply):
        logger.warning(
            "reply bloqueado (tono máquina/nota interna): %s",
            reply[:160],
        )
        return _fallback_warm_reply(client, company, context=context)
    # Soft rewrite if a single slip survived (e.g. "El cliente" → "Tú").
    reply = re.sub(r"\b[Ee]l cliente\b", "Tú", reply)
    reply = re.sub(r"\b[Ll]a cliente\b", "Tú", reply)
    if _looks_like_machine_reply(reply):
        return _fallback_warm_reply(client, company, context=context)
    return meeting_awareness.strip_redundant_call_offer(reply, meeting_state)


def _request_client_ack(
    client: dict[str, Any],
    owner_name: str,
    site: str = "",
) -> str:
    """Confirmación humana — como Sergio en WA, no como ticket del sistema."""
    del owner_name  # equipo Unlockers; no delegar en tercera persona al cliente.
    first = _client_first_name(client)
    greet = f"Dale {first}" if first else "Dale"
    site_bit = f" en {site}" if site else ""
    return (
        f"{greet}, ya lo tengo{site_bit}. Lo revisamos y te escribo por aquí.\n\n"
        f"Si me faltó algo, me lo dices sin pena."
    )


def _recent_human_in_history(session: dict[str, Any], *, seconds: float = 120.0) -> bool:
    """True si Sergio escribió hace poco — evita pisar al humano tras gather+LLM."""
    now = datetime.now(timezone.utc)
    for turn in reversed(session.get("history") or []):
        if turn.get("role") != "assistant":
            continue
        if str(turn.get("kind") or "") != "human":
            continue
        try:
            ts = datetime.fromisoformat(str(turn.get("ts") or ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return True
        if (now - ts).total_seconds() <= seconds:
            return True
        break
    return False


def _gather_key(company_alias: str, contact: str) -> str:
    return f"{company_alias}:{contact}"


def _inflight_key(company_alias: str, contact: str) -> str:
    return _gather_key(company_alias, contact)


def _begin_inflight(key: str) -> bool:
    with _inflight_lock:
        if key in _inflight_turns:
            return False
        _inflight_turns.add(key)
        return True


def _end_inflight(key: str) -> None:
    with _inflight_lock:
        _inflight_turns.discard(key)


def _assistant_replied_within(
    company_alias: str,
    contact: str,
    *,
    seconds: float,
) -> bool:
    """True if the bot already replied in the last N seconds."""
    now = datetime.now(timezone.utc)
    for turn in reversed(store.recent_turns(company_alias, contact, days=1, limit=12)):
        if turn.get("role") != "assistant":
            continue
        ts_raw = str(turn.get("created_at") or "").strip()
        if not ts_raw:
            return True
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return True
        return (now - ts).total_seconds() < seconds
    return False


def _is_opening_greeting(text: str, company: dict[str, Any]) -> bool:
    """Saludo corto, con o sin nombre del dueño ('Hola Sergio' cuenta como saludo)."""
    if _is_bare_greeting(text):
        return True
    owner = str(company.get("owner_name") or "Sergio")
    stripped = _strip_owner_name_mentions(text, owner)
    return _is_bare_greeting(stripped)


def _gather_delay_for(text: str, company: dict[str, Any]) -> float | None:
    """
    Seconds to wait for silence before replying.
    None = answer immediately (closing thanks).
    """
    if _is_closing_thanks(text, str(company.get("owner_name") or "Sergio")):
        return None
    if _is_opening_greeting(text, company):
        return _GREETING_GATHER_SECONDS
    return _GATHER_SECONDS


def _cancel_gather_timer(company_alias: str, contact: str) -> None:
    key = _gather_key(company_alias, contact)
    with _gather_lock:
        timer = _gather_timers.pop(key, None)
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass


def _schedule_gather_flush(
    company_alias: str,
    company: dict[str, Any],
    contact: str,
    profile_name: str,
    delay: float,
) -> None:
    """Schedule flush after `delay` seconds of silence (sliding: cancel + re-arm)."""
    key = _gather_key(company_alias, contact)
    with _gather_lock:
        old = _gather_timers.pop(key, None)
        if old is not None:
            try:
                old.cancel()
            except Exception:
                pass

        def _run() -> None:
            with _gather_lock:
                _gather_timers.pop(key, None)
            try:
                _flush_gather(company_alias, company, contact, profile_name)
            except Exception:
                logger.exception(
                    "gather flush falló empresa=%s contact=%s", company_alias, contact
                )

        timer = threading.Timer(max(1.0, delay), _run)
        timer.daemon = True
        _gather_timers[key] = timer
        timer.start()
    logger.info(
        "gather armado (silencio) empresa=%s contact=%s en %.0fs",
        company_alias, contact, delay,
    )


def _append_gather_buffer(
    company_alias: str,
    company: dict[str, Any],
    contact: str,
    session: dict[str, Any],
    text: str,
    wamid: str,
    *,
    start_new: bool,
    gather_seconds: float = _GATHER_SECONDS,
) -> None:
    """
    Append inbound text to the gather buffer.

    Sliding silence window: each new message resets gather_until + timer.
    Wait in silence — never announce the 60s window (sounds robotic).
    """
    pending = dict(session.get("pending") or {})
    buffer = list(pending.get("buffer") or [])
    gather_until = datetime.now(timezone.utc).timestamp() + gather_seconds
    if start_new:
        buffer = [text.strip()]
    else:
        buffer.append(text.strip())
    pending = {
        "flow": "gather",
        "buffer": buffer[-40:],
        "gather_until": gather_until,
        "gather_seconds": gather_seconds,
        "ack_sent": True,  # never send a visible ACK
    }

    history = _push_history(session, "user", text, kind=_infer_msg_kind(text))
    store.log_turn(
        company_alias, contact, "user", text,
        kind=_infer_msg_kind(text), wamid=wamid,
    )
    store.save_session(
        company_alias, contact, pending=pending, history=history, last_inbound=wamid,
    )
    logger.info(
        "gather buffer empresa=%s contact=%s msgs=%s silent=%.0fs",
        company_alias, contact, len(pending.get("buffer") or []), gather_seconds,
    )


def _defer_to_gather(
    company_alias: str,
    company: dict[str, Any],
    contact: str,
    session: dict[str, Any],
    text: str,
    wamid: str,
    profile_name: str,
    *,
    gather_seconds: float,
    start_new: bool,
) -> None:
    """Buffer a message while another turn is in flight or during greeting cooldown."""
    _append_gather_buffer(
        company_alias, company, contact, session, text, wamid,
        start_new=start_new, gather_seconds=gather_seconds,
    )
    _schedule_gather_flush(
        company_alias, company, contact, profile_name, gather_seconds,
    )


def _flush_gather(
    company_alias: str,
    company: dict[str, Any],
    contact: str,
    profile_name: str,
) -> None:
    """End the 60s window: join buffered messages and route once."""
    contact = normalize_number(contact)
    session = store.get_session(company_alias, contact)
    if store.is_paused(session) or _recent_human_in_history(session):
        store.save_session(company_alias, contact, pending={})
        logger.info(
            "gather abortado (humano/pausa) empresa=%s contact=%s",
            company_alias, contact,
        )
        return
    pending = session.get("pending") or {}
    if pending.get("flow") != "gather":
        return
    buffer = [str(x).strip() for x in (pending.get("buffer") or []) if str(x).strip()]
    store.save_session(company_alias, contact, pending={})
    if not buffer:
        return
    combined = "\n".join(buffer)
    logger.info(
        "gather flush empresa=%s contact=%s msgs=%s chars=%s",
        company_alias, contact, len(buffer), len(combined),
    )
    # Synthetic wamid: already marked pieces as processed when they arrived.
    _process_routed_turn(
        company_alias, company, contact, combined, profile_name,
        wamid=f"gather-{contact}-{int(datetime.now(timezone.utc).timestamp())}",
        already_in_history=True,
    )


def _process_routed_turn(
    company_alias: str,
    company: dict[str, Any],
    contact: str,
    text: str,
    profile_name: str,
    wamid: str,
    *,
    already_in_history: bool = False,
) -> None:
    """Classify + reply for one (possibly combined) user turn."""
    contact = normalize_number(contact)
    inflight = _inflight_key(company_alias, contact)
    if not _begin_inflight(inflight):
        session = store.get_session(company_alias, contact)
        delay = _gather_delay_for(text, company) or _GREETING_GATHER_SECONDS
        _defer_to_gather(
            company_alias, company, contact, session, text, wamid, profile_name,
            gather_seconds=delay, start_new=False,
        )
        logger.info(
            "turn en vuelo — mensaje rebuffered empresa=%s contact=%s",
            company_alias, contact,
        )
        return

    try:
        session = store.get_session(company_alias, contact)
        if (
            _is_opening_greeting(text, company)
            and _assistant_replied_within(
                company_alias, contact, seconds=_GREETING_COOLDOWN_SECONDS,
            )
        ):
            kind = _infer_msg_kind(text)
            if not already_in_history:
                history = _push_history(session, "user", text, kind=kind)
                store.log_turn(company_alias, contact, "user", text, kind=kind, wamid=wamid)
                store.save_session(company_alias, contact, history=history, last_inbound=wamid)
            logger.info(
                "saludo omitido (cooldown) empresa=%s contact=%s",
                company_alias, contact,
            )
            return

        _process_routed_turn_body(
            company_alias, company, contact, text, profile_name, wamid,
            already_in_history=already_in_history,
        )
    finally:
        _end_inflight(inflight)


def _process_routed_turn_body(
    company_alias: str,
    company: dict[str, Any],
    contact: str,
    text: str,
    profile_name: str,
    wamid: str,
    *,
    already_in_history: bool = False,
) -> None:
    """Classify + reply for one (possibly combined) user turn."""
    session = store.get_session(company_alias, contact)
    client = resolve_client(company, contact)
    if client is None:
        client = {
            "alias": f"prospect_{contact}",
            "name": (profile_name or "").strip() or "Cliente",
            "sites": [],
            "prospect": True,
        }
    client = _enrich_client(contact, client, company_alias=company_alias)
    _sync_contact_country(company_alias, contact, session, text)

    meeting_state = meeting_awareness.meeting_state(
        company_alias, company, contact, session=session, client=client,
    )

    reply_context = "general"
    pending = session.get("pending") or {}
    llm_engine = ""
    llm_model = ""
    if pending.get("flow") == "agendar":
        reply = _continue_scheduling(company_alias, company, session, contact, client, text)
    elif pending.get("flow") == "solicitud":
        reply = _continue_request(company_alias, company, session, contact, client, text)
        reply_context = "request"
    elif _is_email_landing_lead(text) and not _history_has_assistant(session, company_alias, contact):
        reply = _email_landing_first_reply(text, client)
        _arm_prospect_warmup(company_alias, contact, topic="Landing correo $200.000")
    else:
        try:
            owner = str(company.get("owner_name") or "Asesor")
            snippet = _history_snippet(company_alias, contact, session, owner, limit=40)
            classify_input = text
            if snippet:
                classify_input = (
                    f"Historial reciente (ultimos mensajes):\n{snippet}\n\n"
                    f"Mensaje actual del cliente:\n{text}"
                )
            parsed = classify_json(_INTENT_SYSTEM, classify_input)
            intent = str(parsed.get("intent") or "consulta")
            # Hard overrides: never treat continuity/status as a cold greeting.
            if intent == "saludo" and (
                _history_has_assistant(session, company_alias, contact)
                or not _is_bare_greeting(text)
            ):
                intent = "consulta"
        except LLMError:
            parsed = {}
            intent = "consulta"
            if _looks_like_request_burst(text):
                intent = "solicitud_cambio"

        if intent == "solicitud_cambio" and _looks_like_clarification_only(text):
            intent = "consulta"

        if intent == "agendar":
            if meeting_state:
                when = str(meeting_state.get("when_label") or "el horario acordado")
                if meeting_state.get("status") == "recent_past":
                    reply = (
                        f"Ya hablamos en la reunión ({when}). "
                        "Cuéntame qué te gustaría revisar ahora y seguimos por aquí."
                    )
                else:
                    reply = (
                        f"Ya tenemos reunión coordinada para {when}. "
                        "Si necesitas moverla, dime y la ajustamos."
                    )
            ack = None if meeting_state else _recent_web_booking_ack(
                company_alias, contact, client, session.get("pending"),
            )
            if ack is not None:
                reply = ack
            elif meeting_state:
                pass
            elif not _history_has_assistant(session, company_alias, contact) and client.get("prospect"):
                reply = _warmup_first_reply(
                    text, client, from_email=_is_email_landing_lead(text),
                )
                _arm_prospect_warmup(
                    company_alias, contact,
                    topic="Landing correo $200.000" if _is_email_landing_lead(text) else "Lead comercial",
                )
            else:
                reply = _start_scheduling(
                    company_alias, company, session, contact, client,
                    str(parsed.get("topic") or ""), str(parsed.get("date_pref") or ""),
                )
        elif intent == "solicitud_cambio":
            if client.get("prospect") and not (client.get("sites") or []):
                reply = _warmup_first_reply(
                    text, client,
                    from_email=_is_email_landing_lead(text),
                )
                _arm_prospect_warmup(
                    company_alias, contact,
                    topic="Lead comercial (soporte → calificar)",
                )
            else:
                reply = _start_request(
                    company_alias, company, contact, client, text, session=session,
                )
                reply_context = "request"
        elif intent == "humano":
            store.pause_session(
                company_alias, contact,
                hours=float(company.get("resume_after_hours") or 6), reason="cliente pidió humano",
            )
            notify_owner(
                company,
                f"[HUMANO] {client.get('name')} ({contact}) pide atención humana: {text[:200]}",
            )
            reply = (
                f"Claro, le aviso a {company.get('owner_name') or 'nuestro equipo'} "
                "para que te atienda personalmente."
            )
        elif _is_closing_thanks(text, str(company.get("owner_name") or "Sergio")):
            # "listo sergio gracias" — never invent a sales topic or call them Sergio.
            reply = "Con gusto. Aquí estoy si algo más."
        else:
            # consulta + saludo (primer hola): siempre LLM con historial e identity_prompt.
            reply, llm_engine, llm_model = _answer_query(
                company_alias, company, session, contact, client, text,
                meeting_state=meeting_state,
            )

    session = store.get_session(company_alias, contact)
    kind = _infer_msg_kind(text)
    if store.is_paused(session) or _recent_human_in_history(session):
        logger.info(
            "reply omitido (humano/pausa) empresa=%s contact=%s",
            company_alias, contact,
        )
        if not already_in_history:
            history = _push_history(session, "user", text, kind=kind)
            store.log_turn(company_alias, contact, "user", text, kind=kind, wamid=wamid)
            store.save_session(company_alias, contact, history=history, last_inbound=wamid)
        return
    outbound = _polish_client_reply(
        reply or "", client, company, context=reply_context,
        meeting_state=meeting_state,
    )
    outbound = _finalize_reply_country(company_alias, contact, session, outbound, text)
    owner_name = str(company.get("owner_name") or "Sergio")
    snippet = _history_snippet(company_alias, contact, session, owner_name, limit=20)
    proposed_slot = meeting_awareness.capture_slot_proposal(outbound, company, snippet)
    if proposed_slot:
        pending = dict(session.get("pending") or {})
        pending.update({
            "flow": "agendar",
            "step": "chat_slot_proposed",
            "proposed_slot": proposed_slot,
            "topic": str(pending.get("topic") or "Reunión comercial (coordinada por chat)"),
        })
        store.save_session(company_alias, contact, pending=pending)
    if already_in_history:
        history = _push_history(session, "assistant", outbound, kind="text")
        store.log_turn(
            company_alias, contact, "assistant", outbound, kind="text",
            llm_engine=llm_engine, llm_model=llm_model,
        )
    else:
        history = _push_history(session, "user", text, kind=kind)
        history = _push_history({**session, "history": history}, "assistant", outbound, kind="text")
        store.log_turn(company_alias, contact, "user", text, kind=kind, wamid=wamid)
        store.log_turn(
            company_alias, contact, "assistant", outbound, kind="text",
            llm_engine=llm_engine, llm_model=llm_model,
        )
    store.save_session(company_alias, contact, history=history, last_inbound=wamid)
    try:
        _reply(company_alias, company, contact, outbound)
    except WhatsAppError as exc:
        notify_owner(company, f"[ERROR] Error enviando respuesta a {contact}: {exc}")


def _start_request(
    company_alias: str, company: dict[str, Any], contact: str,
    client: dict[str, Any], text: str, session: dict[str, Any] | None = None,
) -> str:
    """
    Register the change request immediately with a human ack — no robotic
    \"Te confirmo… ¿Está correcto?\" checklist (that felt archaic and looped).
    """
    sites = ", ".join(client.get("sites") or []) or "(sin sitios registrados)"
    session = session or store.get_session(company_alias, contact)
    owner = str(company.get("owner_name") or "Asesor")
    hist = _history_snippet(company_alias, contact, session, owner, limit=40)
    user_block = (text or "").strip()
    classify_user = (
        f"Historial reciente (solo contexto; no inventes a partir de él si no aporta):\n{hist or '(vacío)'}\n\n"
        f"Texto completo de la solicitud del cliente (varios mensajes unidos):\n{user_block}"
    )
    try:
        data = classify_json(
            _REQUEST_SYSTEM.format(sites=sites),
            classify_user,
            escalate=False,
            num_predict=450,
        )
    except LLMError:
        data = {"site": "", "description": user_block, "priority": "normal"}
    site = str(data.get("site") or "").strip()
    if not site and len(client.get("sites") or []) == 1:
        site = str((client.get("sites") or [""])[0] or "")
    description = str(data.get("description") or user_block).strip()
    if len(description) < 40 and len(user_block) > len(description) + 20:
        description = user_block[:2500]
    priority = str(data.get("priority") or "normal").lower()
    if priority not in ("alta", "normal", "baja"):
        priority = "normal"

    request_id = store.add_request(
        company_alias, contact, client.get("alias", ""), client.get("name", ""),
        site, description[:4000], priority,
    )
    store.save_session(company_alias, contact, pending={})
    notify_owner(
        company,
        f"[SOLICITUD] #{request_id} ({priority})\n"
        f"Cliente: {client.get('name')} ({contact})\n"
        f"Sitio: {site or 'por definir'}\n"
        f"Detalle: {description}",
    )

    owner_name = str(company.get("owner_name") or "Sergio").strip() or "Sergio"
    return _request_client_ack(client, owner_name, site)


def _continue_request(
    company_alias: str, company: dict[str, Any], session: dict[str, Any],
    contact: str, client: dict[str, Any], text: str,
) -> str:
    """Legacy confirm-step sessions: fold into a fresh auto-register."""
    pending = session.get("pending") or {}
    data = dict(pending.get("data") or {})
    store.save_session(company_alias, contact, pending={})
    if _is_no(text):
        return "Ok, lo dejo quieto. Cuéntame de nuevo qué necesitas cuando quieras."
    prev = str(data.get("raw") or data.get("description") or "").strip()
    combined = f"{prev}\n{text.strip()}".strip() if not _is_yes(text) else prev
    if not combined:
        combined = text.strip()
    return _start_request(
        company_alias, company, contact, client, combined, session=session,
    )


# ------------------------------------------------------------ RAG queries

def _answer_query(
    company_alias: str, company: dict[str, Any], session: dict[str, Any],
    contact: str, client: dict[str, Any], text: str,
    *,
    meeting_state: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    from cloudy.bot.company_paths import client_cloudy_aliases
    from cloudy.bot.rag import (
        load_alias_kb,
        load_contact_persona,
        load_style_examples,
        retrieve_for_contact,
    )

    try:
        passages = retrieve_for_contact(company_alias, company, contact, client, text, k=4)
    except Exception:
        passages = []
    context = "\n---\n".join(passages) if passages else "(sin contexto adicional)"
    style = load_style_examples(company, company_alias)

    is_prospect = bool(client.get("prospect"))
    owner_name = str(company.get("owner_name") or "Sergio").strip() or "Sergio"
    display_brand = str(company.get("display_name") or company_alias).strip()
    system = str(company.get("identity_prompt") or "Eres un asistente profesional.")
    system += _CLIENT_VOICE_GUARDRAIL

    from cloudy.bot.contact_country import prompt_block, resolve_country

    user_msgs = store.count_user_messages(company_alias, contact)
    resolved_country = resolve_country(contact, session, text, user_message_count=user_msgs)
    system += prompt_block(resolved_country)

    # Role-specific prompt overlay (sales / client / project).
    try:
        from cloudy.bot.prompts import load_prompt

        role = "sales" if is_prospect else "client"
        system += f"\n\n{load_prompt(role, company_alias)}"
    except (FileNotFoundError, ValueError):
        pass

    if is_prospect:
        system += (
            "\nModo ventas (prospecto): mensajes MUY cortos (2-3 líneas), tono humano. "
            "Primero empatía (¿cómo estás?) y que cuenten su proyecto; NO enlaces de agenda "
            "ni precios hasta que respondan con contexto. Después orienta y ofrece cita con calidez."
            "\nSi el mensaje es solo un gracias/cierre, responde corto (Con gusto) "
            "SIN preguntar por sitio web nuevo, hosting ni planes."
        )
        if meeting_state:
            system += meeting_awareness.prompt_block(meeting_state, owner_name)
        else:
            system += (
                "\nSi no sabes algo, dilo y ofrece agendar una reunión solo cuando aún "
                "NO hay cita coordinada en el historial."
            )
    display_name = str(client.get("name") or "Cliente").strip()
    if display_name.lower() == owner_name.lower() or owner_name.lower() in display_name.lower():
        display_name = "Cliente"
    sites = ", ".join(client.get("sites") or []) or "(sin sitios registrados)"
    system += (
        f"\n\nCliente actual (etiqueta interna, NO es su nombre de pila para saludar): {display_name} "
        f"(sitios: {sites})."
        f"\nIDENTIDAD: Tú hablas en nombre de {display_brand}; {owner_name} es el dueño/asesor humano. "
        f"NUNCA llames al cliente \"{owner_name}\" ni digas \"De nada, {owner_name}\". "
        f"Si el cliente escribe \"Hola {owner_name}\" o \"gracias {owner_name}\", te está hablando a ti/al equipo; "
        "responde sin ponerle ese nombre. Si el contacto es empresa o línea compartida, no uses nombre propio."
        "\nResponde en máximo 4 frases, útil y concreto. Si no sabes algo, dilo."
        f"{' NO ofrezcas otra llamada ni cita: ya hay reunión coordinada.' if meeting_state else ' Si aplica y no hay cita, ofrece agendar una reunión.'}"
        " Puedes citar precios que estén en el contexto KB (convertidos a la moneda del contacto si no es Colombia); no inventes otros."
        f"\n\nContexto de la empresa (solo tenant {company_alias}):\n{context}"
    )

    persona = load_contact_persona(company_alias, contact)
    if persona:
        system += (
            f"\n\nPERSONA / VOZ CON ESTE CONTACTO (tenant {company_alias}, no mezclar con otros clientes):\n"
            f"{persona}"
        )

    alias_kb_blocks: list[str] = []
    for alias in client_cloudy_aliases(client):
        kb_text = load_alias_kb(company_alias, alias)
        if kb_text:
            alias_kb_blocks.append(f"### {alias}\n{kb_text}")
    if alias_kb_blocks:
        system += (
            "\n\nKB conversacional del proyecto (solo este tenant):\n"
            + "\n\n".join(alias_kb_blocks)
        )

    client_notes = str(client.get("notes") or "").strip()
    if not is_prospect:
        system += (
            "\n\nMODO CLIENTE EXISTENTE: este número ya está registrado como cliente/contacto "
            "de un proyecto Cloudy. NO ofrezcas paquetes web, hosting, tienda online, landing "
            "$200.000 ni agendar llamada comercial salvo que el cliente lo pida. "
            "Da continuidad de soporte con empatía."
        )
    if client_notes and not is_prospect:
        system += (
            f"\n\nFicha interna del cliente (número reconocido): {client_notes}"
            "\nEste número YA es un cliente conocido: no respondas como si no supieras quién es "
            "ni qué tiene contratado con nosotros; da continuidad. No cites la ficha textualmente "
            "ni menciones que tienes una ficha o base de datos."
        )
    if style:
        system += f"\n\n{style}"

    system += (
        "\n\nUsa el historial de los últimos días (hasta ~1 semana) para mantener continuidad: "
        "qué preguntó el cliente, audios/fotos ya descritos, qué ofreció el equipo o tú, "
        "y no repitas preguntas ya respondidas."
        "\nNUNCA repitas un saludo de presentación ni el menú de servicios si ya hablaste en el historial."
        "\nSi el cliente habla de retraso, almuerzo, noche, disculpas o reagendar, responde a ESO "
        f"con empatía y flexibilidad (como {owner_name}), sin volver a presentarte."
        "\nSi no pudiste escuchar un audio o abrir una imagen, pide reenvío o texto; "
        "NO inventes que quiere armar un sitio web u otro servicio sin evidencia en el historial."
        "\nPuedes citar o retomar su frase en una línea corta y luego el siguiente paso."
    )
    if _history_has_assistant(session, company_alias, contact) or any(
        t.get("role") == "assistant"
        for t in store.recent_turns(company_alias, contact, days=7, limit=30)
    ):
        system += (
            f"\nYa hubo mensajes tuyos o de {owner_name} en este chat: continúa el hilo, no te "
            f"presentes otra vez como asistente de {display_brand}."
        )
    messages = [{"role": "system", "content": system}]
    messages.extend(_history_for_llm(company_alias, contact, session, limit=100))
    messages.append({"role": "user", "content": text})
    profile_ctx = (
        f"empresa={company_alias}; nombre={client.get('name')}; tel={contact}; "
        f"prospecto={'si' if is_prospect else 'no'}"
    )
    try:
        reply = chat(
            messages,
            channel="whatsapp",
            profile_context=profile_ctx,
            allow_cloud_fallback=True,
        )
        meta = pop_chat_meta()
        return reply, meta.get("engine", ""), meta.get("model", "")
    except (LLMError, RuntimeError):
        try:
            from cloudy.bot.katana_chat import cloud_chat

            fallback = cloud_chat(
                messages,
                channel="whatsapp",
                profile_context=profile_ctx,
                user_message=text,
            )
            return fallback, "openai-katana", "platform"
        except Exception:
            booking_url = str(company.get("appointment_booking_url") or "").strip()
            fallback_agenda = f" También puedes agendar aquí: {booking_url}." if booking_url else ""
            return (
                "En este momento tengo un problema técnico para consultar. "
                f"Escríbenos o espera un momento.{fallback_agenda} "
                "WhatsApp comercial: 316 624 8968."
            ), "", ""


# ---------------------------------------------------------- entry points

def handle_inbound(
    company_alias: str, company: dict[str, Any], contact: str,
    text: str, profile_name: str, wamid: str,
) -> None:
    """Process one inbound text message and reply (called from the webhook)."""
    if not store.mark_processed(wamid):
        return  # Meta redelivery of an event we already handled

    contact = normalize_number(contact)
    text = (text or "").strip()
    if not text:
        return

    if _is_katana_media_placeholder(text):
        logger.info(
            "inbound placeholder Katana omitido contact=%s wamid=%s text=%s",
            contact, wamid, text[:40],
        )
        return

    from cloudy.bot.config import is_listen_only

    if is_listen_only(company):
        from cloudy.bot.observe import capture_inbound

        capture_inbound(company_alias, company, contact, text, profile_name, wamid)
        return

    # Owner control commands work even though the owner may also be a client.
    if is_owner(company, contact):
        command_reply = _handle_owner_command(company_alias, company, text)
        if command_reply is not None:
            _reply(company_alias, company, contact, command_reply)
            return

    session = store.get_session(company_alias, contact)
    client = resolve_client(company, contact)

    if client is None:
        client = {
            "alias": f"prospect_{contact}",
            "name": (profile_name or "").strip() or "Cliente",
            "sites": [],
            "prospect": True,
        }
    client = _enrich_client(contact, client, company_alias=company_alias)
    country_resolved = _sync_contact_country(company_alias, contact, session, text)

    prior_user_msgs = store.count_user_messages(company_alias, contact)
    is_first_contact = prior_user_msgs == 0
    attribution = _detect_inbound_attribution(text)

    # Primer contacto o prospecto → CRM Katana (lead comercial + seguimiento).
    if is_first_contact or client.get("prospect"):
        try:
            from cloudy.bot.katana_chat import upsert_contact

            hist_snip = _history_snippet(
                company_alias, contact, session,
                str(company.get("owner_name") or "Asesor"), limit=12,
            )
            stage = "calificar"
            if _is_commercial_lead(text, client):
                stage = "calificar"
            country_note = ""
            if country_resolved.is_known:
                country_note = f"[país detectado: {country_resolved.label_es} ({country_resolved.code})] "
            crm_result = upsert_contact({
                "name": client["name"],
                "phone": contact,
                "channel": "whatsapp",
                "summary": (country_note + hist_snip + "\n" + text)[:1500],
                "stage": stage,
                "is_first_contact": is_first_contact,
                "last_inbound_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "detected_country": country_resolved.code if country_resolved.is_known else "",
                "interested_services": attribution.get("interested_services"),
                "utm_source": attribution.get("utm_source", ""),
                "utm_medium": attribution.get("utm_medium", ""),
                "utm_campaign": attribution.get("utm_campaign", ""),
                "utm_content": attribution.get("utm_content", ""),
            })
            if crm_result and crm_result.get("success"):
                lead_id = crm_result.get("commercial_lead_id")
                if lead_id:
                    store.mark_crm_synced(company_alias, contact, int(lead_id))
            else:
                store.mark_crm_pending(company_alias, contact)
        except Exception:
            store.mark_crm_pending(company_alias, contact)
            logger.exception("CRM upsert prospect failed contact=%s", contact)

    # Human already took over this conversation -> stay silent but KEEP context.
    if store.is_paused(session):
        kind = _infer_msg_kind(text)
        history = _push_history(session, "user", text, kind=kind)
        store.log_turn(company_alias, contact, "user", text, kind=kind, wamid=wamid)
        store.save_session(
            company_alias, contact,
            history=history, last_inbound=wamid,
        )
        return

    store.save_session(
        company_alias, contact,
        client_alias=str(client.get("alias") or ""),
        client_name=str(client.get("name") or ""),
        last_inbound=wamid,
    )
    session = store.get_session(company_alias, contact)

    # Immediate keyword handoff, before any LLM / gather.
    if any(word in text.lower() for word in _HANDOFF_WORDS):
        _cancel_gather_timer(company_alias, contact)
        store.save_session(company_alias, contact, pending={})
        store.pause_session(
            company_alias, contact,
            hours=float(company.get("resume_after_hours") or 6), reason="cliente pidió humano",
        )
        notify_owner(company, f"[HUMANO] {client.get('name')} ({contact}) pide atención humana: {text[:200]}")
        reply = (
            f"Claro, le aviso a {company.get('owner_name') or 'nuestro equipo'} para que te atienda "
            "personalmente. Mientras tanto quedo atento por aquí."
        )
        kind = _infer_msg_kind(text)
        history = _push_history(session, "user", text, kind=kind)
        history = _push_history({**session, "history": history}, "assistant", reply, kind="text")
        store.log_turn(company_alias, contact, "user", text, kind=kind, wamid=wamid)
        store.log_turn(company_alias, contact, "assistant", reply, kind="text")
        store.save_session(company_alias, contact, history=history)
        _reply(company_alias, company, contact, reply)
        return

    pending = session.get("pending") or {}

    # Mid-flow confirmations (sí/no, slots) must stay immediate.
    if pending.get("flow") in ("agendar", "solicitud"):
        _cancel_gather_timer(company_alias, contact)
        _process_routed_turn(
            company_alias, company, contact, text, profile_name, wamid,
            already_in_history=False,
        )
        return

    # Voice/image failed to process: ask for resend — never invent a sales pitch.
    if _is_media_fail_message(text):
        _cancel_gather_timer(company_alias, contact)
        store.save_session(company_alias, contact, pending={})
        reply = (
            "Uy, no me cargó la foto o el audio. "
            "¿Me lo reenvías o me cuentas en una línea qué hay?"
        )
        kind = "media_fail"
        history = _push_history(session, "user", text, kind=kind)
        history = _push_history({**session, "history": history}, "assistant", reply, kind="text")
        store.log_turn(company_alias, contact, "user", text, kind=kind, wamid=wamid)
        store.log_turn(company_alias, contact, "assistant", reply, kind="text")
        store.save_session(company_alias, contact, history=history, last_inbound=wamid)
        try:
            _reply(company_alias, company, contact, reply)
        except WhatsAppError as exc:
            notify_owner(company, f"[ERROR] media-fail reply {contact}: {exc}")
        notify_owner(company, f"[MEDIA] No pude procesar medio de {client.get('name')} ({contact})")
        return

    # Active gather window: keep buffering; each msg resets the silence timer.
    if pending.get("flow") == "gather":
        until = float(pending.get("gather_until") or 0)
        gather_seconds = float(pending.get("gather_seconds") or _GATHER_SECONDS)
        now_ts = datetime.now(timezone.utc).timestamp()
        if until and now_ts < until:
            _append_gather_buffer(
                company_alias, company, contact, session, text, wamid,
                start_new=False, gather_seconds=gather_seconds,
            )
            _schedule_gather_flush(
                company_alias, company, contact, profile_name, gather_seconds,
            )
            return
        # Window expired but timer missed: flush including this message.
        buffer = list(pending.get("buffer") or [])
        buffer.append(text)
        store.save_session(company_alias, contact, pending={
            "flow": "gather",
            "buffer": buffer,
            "gather_until": until,
            "ack_sent": True,
        })
        _cancel_gather_timer(company_alias, contact)
        _flush_gather(company_alias, company, contact, profile_name)
        return

    # New turn: wait for silence so the client can finish typing (human feel).
    gather_delay = _gather_delay_for(text, company)
    if gather_delay is None:
        _process_routed_turn(
            company_alias, company, contact, text, profile_name, wamid,
            already_in_history=False,
        )
        return

    _defer_to_gather(
        company_alias, company, contact, session, text, wamid, profile_name,
        gather_seconds=gather_delay, start_new=True,
    )


def handle_outbound_echo(
    company_alias: str,
    company: dict[str, Any],
    contact: str,
    wamid: str,
    text: str = "",
) -> None:
    """
    Outbound message echo from Meta (coexistence: phone app / WhatsApp Web).

    - If the bot sent it (or recent self-heal window): ignore / register.
    - Otherwise a human (Sergio) is answering: store THEIR text in history
      so the next bot turn sees how they speak and what the client asked,
      then pause the conversation.
    """
    from cloudy.bot.config import is_listen_only

    contact = normalize_number(contact)
    if is_listen_only(company):
        from cloudy.bot.observe import capture_outbound_echo

        capture_outbound_echo(company_alias, company, contact, wamid, text)
        return

    if not wamid or store.was_sent_by_bot(wamid):
        return

    if store.recently_sent(company_alias, contact, seconds=_ECHO_GRACE_SECONDS):
        store.record_sent(wamid, company_alias, contact)
        logger.info(
            "echo self-heal: wamid=%s contact=%s empresa=%s (dentro de %ss desde el ultimo envio del bot, no se pausa)",
            wamid, contact, company_alias, _ECHO_GRACE_SECONDS,
        )
        return

    session = store.get_session(company_alias, contact)
    body = (text or "").strip()
    if body:
        # Persist owner style into the 7-day window (assistant role).
        history = _push_history(session, "assistant", body, kind="human")
        store.log_turn(company_alias, contact, "assistant", body, kind="human", wamid=wamid)
        store.save_session(company_alias, contact, history=history)
        session = store.get_session(company_alias, contact)

    # Takeover keyword: Sergio types a phrase into the client chat to silence
    # the bot for a full day. This overrides the default short pause and
    # re-arms it even if the conversation was already paused.
    takeover_words = [
        _strip_accents(w)
        for w in (company.get("takeover_keywords") or _TAKEOVER_WORDS)
    ]
    if body:
        normalized = _strip_accents(body)
        if any(word and word in normalized for word in takeover_words):
            pause_hours = float(
                company.get("takeover_pause_hours") or _TAKEOVER_PAUSE_HOURS
            )
            _cancel_gather_timer(company_alias, contact)
            store.pause_session(
                company_alias, contact,
                hours=pause_hours,
                reason=f"toma de control de Sergio (palabra clave, {pause_hours:.0f}h)",
            )
            logger.warning(
                "toma de control por palabra clave: contact=%s empresa=%s -> bot en silencio %.0fh",
                contact, company_alias, pause_hours,
            )
            return

    if store.is_paused(session):
        return

    _cancel_gather_timer(company_alias, contact)
    logger.warning(
        "handoff humano detectado: wamid=%s contact=%s empresa=%s -> pausando conversacion",
        wamid, contact, company_alias,
    )
    store.pause_session(
        company_alias, contact,
        hours=float(company.get("resume_after_hours") or 6),
        reason="intervención humana (celular/WhatsApp Web)",
    )



