import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SUBTITLE_DIR
from app.subtitle_parser import parse_subtitle_file

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    if not SUBTITLE_DIR.exists():
        logging.info("Subtitle folder not found: %s", SUBTITLE_DIR)
        return

    files = list(SUBTITLE_DIR.rglob("*.srt")) + list(SUBTITLE_DIR.rglob("*.vtt"))
    if not files:
        logging.info("No subtitle files found in %s", SUBTITLE_DIR)
        return

    logging.info("Found %s subtitle files.", len(files))
    total_entries = 0
    for path in files:
        entries = parse_subtitle_file(path)
        total_entries += len(entries)
        logging.info("%s: %s entries", path.name, len(entries))

    logging.info("Total entries: %s", total_entries)


if __name__ == "__main__":
    main()
