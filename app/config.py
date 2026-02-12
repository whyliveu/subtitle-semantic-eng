import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUBTITLE_DIR = DATA_DIR / "subtitles"
INDEX_DIR = DATA_DIR / "index"
FAVORITES_PATH = DATA_DIR / "favorites.json"

CHROMA_COLLECTION = "movie_quotes"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

LINE_CONTEXT_SIZE = 2
SCENE_MIN_SECONDS = 20
SCENE_MAX_SECONDS = 60
