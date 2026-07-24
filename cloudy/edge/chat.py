# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""
Chat completion for Cloudy Edge — Katana-compatible contract.

Response shape matches BotContingencyChatService / BotChatController::

    {"success": true, "reply": "...", "provider": "groq-gpt-oss"}
    {"success": false, "reply": "", "error": "..."}
"""
from __future__ import annotations

import logging
from typing import Any

from cloudy.bot.llm import LLMError
from cloudy.edge.config import edge_config, kb_dir, system_prompt_path
from cloudy.edge.kb import retrieve_kb
from cloudy.edge.llm import edge_chat

logger = logging.getLogger("cloudy.edge")

# Fallback when whatsapp.json / system_prompt_file are unavailable (VPS fresh install).
_DEFAULT_SYSTEM = (
    "Eres Cloudy, la voz de UNLOCKERS CLOUD (1lockers.net) en el WhatsApp 316 624 8968 "
    "(Bucaramanga, Colombia). Hablas como Sergio: cercano, humano, mensajes MUY cortos "
    "(2-3 líneas); sin markdown; una pregunta. Nunca digas que eres una IA. "
    "Primero empatía (¿cómo estás?) y que cuenten su proyecto; NO agenda en el primer mensaje. "
    "Después de 1-2 turnos orienta y ofrece https://1lockers.net/agendar/unlockers o por chat. "
    "Sergio es el dueño humano; NUNCA llames al cliente Sergio."
)


def _load_system_prompt() -> str:
    """System prompt: file override → whatsapp.json unlockers → embedded default."""
    path = system_prompt_path()
    if path is not None:
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            pass

    try:
        from cloudy.bot.config import load_config

        companies = load_config().get("companies") or {}
        unlockers = companies.get("unlockers") or {}
        prompt = str(unlockers.get("identity_prompt") or "").strip()
        if prompt:
            return prompt
    except Exception:
        pass

    return _DEFAULT_SYSTEM


def _last_user_message(messages: list[dict[str, str]], user_message: str) -> str:
    if user_message.strip():
        return user_message.strip()
    for turn in reversed(messages):
        if str(turn.get("role") or "") == "user":
            return str(turn.get("content") or "").strip()
    return ""


def _is_recognized_client(profile_context: str) -> bool:
    return "Cliente reconocido: SI" in (profile_context or "")


def _load_role_prompt(role: str, fallback: str) -> str:
    try:
        from cloudy.bot.prompts import load_prompt

        return load_prompt(role)
    except Exception:
        return fallback


def _build_messages(
    raw_messages: list[dict[str, str]],
    *,
    channel: str,
    profile_context: str,
    user_message: str,
) -> list[dict[str, str]]:
    """
    Build the LLM message list: system (prompt + KB + profile) + history.

    Caller ``messages`` may include system turns; we rebuild system from Edge
    rules so KB retrieval stays consistent with Katana contingency.
    """
    last_user = _last_user_message(raw_messages, user_message)
    is_project = (channel or "").strip() == "project_whatsapp"
    is_client = is_project or _is_recognized_client(profile_context)

    if is_project:
        kb_block = "(modo proyecto — sin KB comercial)"
        system = _load_role_prompt(
            "project",
            "Eres el asistente de PROYECTO de Unlockers para un cliente de pago. "
            "NO vendes, NO agendas citas comerciales, NO prometes ejecutar cambios en el sitio. "
            "Máximo 4 frases, español natural, sin markdown.",
        )
    elif is_client:
        kb_block = "(modo cliente existente — sin ventas)"
        system = _load_role_prompt(
            "client",
            "Eres Cloudy hablando con un CLIENTE EXISTENTE. "
            "NO ofrezcas paquetes web ni agenda comercial. Modo soporte/continuidad.",
        )
    else:
        passages = retrieve_kb(kb_dir(), last_user or "cita servicios", k=4)
        kb_block = "\n---\n".join(passages) if passages else "(sin fragmentos KB)"
        system = _load_system_prompt()

    system += f"\nCanal: {channel}.\n"
    if profile_context.strip():
        system += f"Perfil del contacto:\n{profile_context.strip()}\n"
    system += f"Conocimiento {'del proyecto' if is_project else 'comercial'} (usa solo esto + el hilo):\n{kb_block}\n"
    if is_project:
        system += (
            "\nSi preguntan por cobros o pagos, responde con los datos del perfil. "
            "Si piden cambiar algo del sitio con un pedido claro y nuevo, indica cola de revisión del equipo. "
            "Si es aclaración sobre un ajuste en curso (capturas, HTML, 'las de antes'), continúa ese hilo "
            "sin mencionar cobros ni abrir solicitud nueva. "
            "NUNCA menciones saldo pendiente si el cliente no preguntó por pagos.\n"
        )
    elif is_client:
        system += (
            "\nCliente ya registrado: continúa el hilo con empatía. "
            "No repitas menú de servicios ni ofrezcas planes nuevos. "
            "Si el perfil indica tickets RESUELTOS u órdenes COMPLETADAS, NO repitas ese trabajo "
            "ni abras solicitud nueva; confirma que ya quedó y pide que verifiquen.\n"
        )
    else:
        system += (
            "\nUsa el historial para continuidad; no repitas saludos ni menús si ya "
            "respondiste. Responde en máximo 4 frases."
        )

    history: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in raw_messages[-20:]:
        role = str(turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content[:4000]})

    # Ensure the latest user turn is present (Katana sometimes passes it separately).
    if last_user and (not history or history[-1].get("content") != last_user):
        if history and history[-1].get("role") == "user":
            history[-1] = {"role": "user", "content": last_user[:4000]}
        else:
            history.append({"role": "user", "content": last_user[:4000]})

    return history


def complete_chat(
    messages: list[dict[str, str]],
    channel: str = "whatsapp",
    profile_context: str = "",
    user_message: str = "",
) -> dict[str, Any]:
    """
    Katana-compatible chat completion using the cloud-only LLM chain + KB.

    Parameters mirror BotContingencyChatService::complete().
    """
    cfg = edge_config()
    try:
        llm_messages = _build_messages(
            messages,
            channel=channel or "edge",
            profile_context=profile_context,
            user_message=user_message,
        )
        content, engine, model = edge_chat(
            llm_messages,
            temperature=float(cfg.get("temperature", 0.55)),
            max_tokens=int(cfg.get("max_tokens", 500)),
            channel=channel or "edge",
            profile_context=profile_context,
        )
        reply = content.strip()
        if not reply:
            return {"success": False, "reply": "", "error": "empty_reply"}

        return {
            "success": True,
            "reply": reply,
            "provider": f"cloudy-edge:{engine}",
            "llm_engine": engine,
            "llm_model": model,
        }
    except LLMError as exc:
        logger.warning("edge chat LLM error: %s", exc)
        return {"success": False, "reply": "", "error": str(exc)}
    except Exception as exc:
        logger.exception("edge chat unexpected error")
        return {"success": False, "reply": "", "error": "exception"}


__all__ = ["complete_chat"]
