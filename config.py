"""
config.py — loads all settings from .env
Supports: SQLite (local) + Supabase Postgres (online)
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Flask ──────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
DEBUG      = os.getenv("DEBUG", "False").lower() == "true"

# ── AI Mode ───────────────────────────────────────────────
AGE_AI_MODE = os.getenv("AGE_AI_MODE", "opencv")

# ── Database ───────────────────────────────────────────────
# Online  → set DATABASE_URL in environment/Render dashboard
# Local   → leave DATABASE_URL empty → SQLite used automatically
_db_url = os.getenv("DATABASE_URL", "").strip()

if _db_url:
    # Render sometimes gives postgres:// → fix to postgresql://
    if _db_url.startswith("postgres://"):
        _db_url = _db_url.replace("postgres://", "postgresql://", 1)
    # Add sslmode for Supabase pooler (required)
    if "supabase.com" in _db_url and "sslmode" not in _db_url:
        _db_url += "?sslmode=require"
    SQLALCHEMY_DATABASE_URI = _db_url
    DB_IS_POSTGRES = True
else:
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "database.db")
    DB_IS_POSTGRES = False

SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_pre_ping":  True,     # detect dead connections
    "pool_recycle":   300,      # recycle every 5 min (Supabase drops idle)
    "connect_args":   {"connect_timeout": 10} if DB_IS_POSTGRES else {},
}

# ── File upload ────────────────────────────────────────────
UPLOAD_FOLDER      = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MODEL_PATH         = os.path.join(BASE_DIR, "models", "best_utkface_model.pth")

# ── Supabase ───────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── Cloudinary ─────────────────────────────────────────────
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
CLOUDINARY_API_KEY    = os.getenv("CLOUDINARY_API_KEY",    "").strip()
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "").strip()

USE_CLOUDINARY = bool(
    CLOUDINARY_CLOUD_NAME
    and CLOUDINARY_API_KEY
    and CLOUDINARY_API_SECRET
    and " " not in CLOUDINARY_CLOUD_NAME
)
