# Cloudy bot mirror (Cursor Cloud)

Auto-generated from Mac Cloudy repo. **No secrets** (no llm.json, cursor.json, whatsapp.json, tokens).

## Contents

| Path | Purpose |
|------|---------|
| `prompts/` | System prompts (sales, client, project, handoff) |
| `kb/` | Commercial KB markdown (unlockers) |
| `cloudy/bot/` | LLM chain, engine, Katana sync, RAG (read-only audit) |
| `cloudy/cursor/` | Cursor SDK client |
| `cloudy/security/` | M2M HMAC + Katana HTTP helpers |
| `cloudy/edge/` | Edge contingency LLM |
| `config/llm.json.example` | Engine chain template (no API keys) |
| `deploy/macos/sync-bundled-node.sh` | Node bundle sync for launchd |

Synced by `scripts/maintenance/push_cursor_cloud_mirror.sh` (also via `bot sync-all`).
