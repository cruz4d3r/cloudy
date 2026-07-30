"""Single source of truth for Cloudy paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
FILEZILLA_XML = CONFIG_DIR / "FileZilla.xml"
PROJECTS_JSON = CONFIG_DIR / "projects.json"
BACKUPS_DIR = CONFIG_DIR / "backups"
DATABASES_JSON = CONFIG_DIR / "databases.json"
CPANEL_JSON = CONFIG_DIR / "cpanel.json"

CLIENTS_DIR = ROOT / "clients"
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
MARKET_CONTACTS_DIR = DATA_DIR / "market-contacts"
SCRIPTS_DIR = ROOT / "scripts"

FTP_TIMEOUT = 15
MAX_LIST_CHILDREN = 80

SKIP_NAMES = frozenset({
    "error_log",
    ".DS_Store",
    "Thumbs.db",
    ".git",
    "node_modules",
    "vendor",
    ".env",
})

# Directorios omitidos en pull (adjuntos, uploads, cachés, backups).
SKIP_DIR_NAMES = frozenset({
    "uploads",
    "upload",
    "cache",
    "caches",
    "storage",
    "tmp",
    "temp",
    "attachments",
    "adjuntos",
    "media",
    "backup",
    "backups",
    "logs",
    "log",
    "private",
    "sessions",
    "session",
    "mail",
    "maildir",
    "imap",
    "cpanel",
    "softaculous",
    "awstats",
    "stats",
    "cgi-bin",
    "trash",
    ".trash",
    ".recycle",
    "recycle",
    ".next",
    ".nuxt",
    ".cache",
    ".npm",
    ".yarn",
    "bower_components",
})

# Extensiones de archivos binarios/medios que no se descargan en pull.
SKIP_FILE_EXTENSIONS = frozenset({
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".7z",
    ".rar",
    ".sql",
    ".sqlite",
    ".sqlite3",
    ".bak",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".flac",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
    ".pdf",
    ".psd",
    ".ai",
    ".eps",
})

# Tamaño máximo por archivo en pull (solo código; evita dumps/imágenes sueltas).
MAX_PULL_FILE_BYTES = 2 * 1024 * 1024

SKIP_PREFIXES = (".trash", ".cpanel", ".spamassassin")

SITE_FOLDER_NAMES = frozenset({"crm", "catalogo", "Catalogos", "api", "frontend", "dist"})
FILE_EXTENSIONS = (".html", ".php", ".txt", ".pdf", ".png", ".ini")

# Bundled Node.js tarballs (Cursor SDK requires >= 22.13 for local agents).
_BUNDLED_NODE_CANDIDATES = (
    "node-v22.16.0-darwin-arm64",
    "node-v20.20.2-darwin-arm64",
)


def bundled_node_binary() -> str:
    """Return path to bundled Node binary, or 'node' from PATH."""
    tools = ROOT / ".tools"
    for dirname in _BUNDLED_NODE_CANDIDATES:
        candidate = tools / dirname / "bin" / "node"
        if candidate.is_file():
            return str(candidate)
    return "node"


def client_meta_dir(alias: str) -> Path:
    return CLIENTS_DIR / alias / "meta"


def client_deliveries_dir(alias: str) -> Path:
    if alias == "impark":
        return (
            CLIENTS_DIR
            / "1lockers"
            / "subprojects"
            / "impark"
            / "meta"
            / "deliveries"
        )
    return client_meta_dir(alias) / "deliveries"


def client_sites_dir(alias: str) -> Path:
    return CLIENTS_DIR / alias / "sites"


def client_inventory_file(alias: str) -> Path:
    return client_meta_dir(alias) / "remote-inventory.json"
