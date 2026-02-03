import argparse
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chromadb
from openai import OpenAI

from app.config import (
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    INDEX_DIR,
    LINE_CONTEXT_SIZE,
    SCENE_MAX_SECONDS,
    SCENE_MIN_SECONDS,
    SUBTITLE_DIR,
)
from app.subtitle_parser import SubtitleEntry, parse_subtitle_file

logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class Record:
    record_id: str
    document: str
    metadata: Dict[str, Optional[str]]


def extract_movie_info(path: Path) -> Tuple[str, Optional[str]]:
    stem = path.stem.replace(".", " ").replace("_", " ")
    match = re.search(r"\((\d{4})\)", stem)
    year = None
    if match:
        year = match.group(1)
        title = stem.replace(match.group(0), "").strip()
    else:
        title = stem.strip()
    return title, year


def build_line_windows(entries: List[SubtitleEntry], source: Path) -> List[Record]:
    records: List[Record] = []
    source_hash = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:12]
    movie_title, year = extract_movie_info(source)
    source_dir = str(source.parent)

    for idx, entry in enumerate(entries):
        before = " ".join(
            e.text for e in entries[max(0, idx - LINE_CONTEXT_SIZE) : idx]
        ).strip()
        after = " ".join(
            e.text
            for e in entries[idx + 1 : idx + 1 + LINE_CONTEXT_SIZE]
        ).strip()
        search_text = " ".join([before, entry.text, after]).strip()
        record_id = f"{source_hash}_line_{idx}"
        records.append(
            Record(
                record_id=record_id,
                document=search_text,
                metadata={
                    "type": "line_window",
                    "source_file": str(source),
                    "source_dir": source_dir,
                    "movie_title": movie_title,
                    "year": year,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "line_text": entry.text,
                    "context_before": before,
                    "context_after": after,
                },
            )
        )
    return records


def build_scene_chunks(entries: List[SubtitleEntry], source: Path) -> List[Record]:
    records: List[Record] = []
    source_hash = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:12]
    movie_title, year = extract_movie_info(source)
    source_dir = str(source.parent)

    chunk_entries: List[SubtitleEntry] = []
    chunk_start = None

    def flush_chunk(chunk_idx: int) -> None:
        if not chunk_entries:
            return
        start_time = chunk_entries[0].start_time
        end_time = chunk_entries[-1].end_time
        chunk_text = " ".join(entry.text for entry in chunk_entries)
        record_id = f"{source_hash}_scene_{chunk_idx}"
        records.append(
            Record(
                record_id=record_id,
                document=chunk_text,
                metadata={
                    "type": "scene_chunk",
                    "source_file": str(source),
                    "source_dir": source_dir,
                    "movie_title": movie_title,
                    "year": year,
                    "start_time": start_time,
                    "end_time": end_time,
                    "line_text": chunk_text,
                    "context_before": "",
                    "context_after": "",
                },
            )
        )

    chunk_idx = 0
    for entry in entries:
        if not chunk_entries:
            chunk_entries.append(entry)
            chunk_start = entry.start_seconds
            continue
        chunk_entries.append(entry)
        duration = entry.end_seconds - (chunk_start or entry.start_seconds)
        if duration >= SCENE_MAX_SECONDS:
            flush_chunk(chunk_idx)
            chunk_idx += 1
            chunk_entries = []
            chunk_start = None

    if chunk_entries:
        duration = chunk_entries[-1].end_seconds - (chunk_start or 0)
        if duration >= SCENE_MIN_SECONDS or not records:
            flush_chunk(chunk_idx)

    return records


def collect_records(files: Iterable[Path]) -> List[Record]:
    records: List[Record] = []
    for path in files:
        entries = parse_subtitle_file(path)
        if not entries:
            logging.warning("Skipped empty file: %s", path)
            continue
        records.extend(build_line_windows(entries, path))
        records.extend(build_scene_chunks(entries, path))
    return records


def iter_subtitle_files(root: Path) -> List[Path]:
    return sorted(
        [path for path in root.rglob("*") if path.suffix.lower() in {".srt", ".vtt"}]
    )


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def index_records(records: List[Record], batch_size: int) -> None:
    if not records:
        logging.info("No records to index.")
        return
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    chroma = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = chroma.get_or_create_collection(name=CHROMA_COLLECTION)

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        embeddings = embed_texts(client, [record.document for record in batch])
        collection.upsert(
            ids=[record.record_id for record in batch],
            documents=[record.document for record in batch],
            embeddings=embeddings,
            metadatas=[record.metadata for record in batch],
        )
        logging.info("Indexed %s/%s records...", min(i + batch_size, len(records)), len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description="Index subtitle files for semantic search.")
    parser.add_argument("--batch-size", type=int, default=100, help="Embedding batch size.")
    args = parser.parse_args()

    if not SUBTITLE_DIR.exists():
        logging.info("Subtitle folder not found: %s", SUBTITLE_DIR)
        return

    files = iter_subtitle_files(SUBTITLE_DIR)
    if not files:
        logging.info("No subtitle files found in %s", SUBTITLE_DIR)
        return

    logging.info("Found %s subtitle files.", len(files))
    records = collect_records(files)
    logging.info("Prepared %s records.", len(records))
    index_records(records, batch_size=args.batch_size)
    logging.info("Indexing completed.")


if __name__ == "__main__":
    main()
