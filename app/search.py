import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from openai import OpenAI

from app.config import (
    CHROMA_COLLECTION,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    FAVORITES_PATH,
    INDEX_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_filters(args: argparse.Namespace) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if args.movie:
        filters["movie_title"] = args.movie
    if args.year:
        filters["year"] = str(args.year)
    if args.folder:
        filters["source_dir"] = args.folder
    if args.record_type:
        filters["type"] = args.record_type
    return filters


def generate_paraphrases(client: OpenAI, query: str, count: int) -> List[str]:
    prompt = (
        "Generate {count} short English paraphrases of this search query for movie subtitles.\n"
        "Query: {query}\n"
        "Return each paraphrase on a new line, no numbering."
    ).format(count=count, query=query)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response.choices[0].message.content or ""
    paraphrases = [line.strip() for line in content.splitlines() if line.strip()]
    return paraphrases[:count]


def embed_queries(client: OpenAI, queries: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=queries)
    return [item.embedding for item in response.data]


def query_index(
    collection: chromadb.Collection,
    query_embeddings: List[List[float]],
    top_k: int,
    filters: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results_by_id: Dict[str, Dict[str, Any]] = {}

    response = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        where=filters or None,
        include=["documents", "metadatas", "distances"],
    )

    for ids, docs, metas, distances in zip(
        response["ids"],
        response["documents"],
        response["metadatas"],
        response["distances"],
    ):
        for record_id, doc, meta, distance in zip(ids, docs, metas, distances):
            score = 1 - distance if distance is not None else 0
            current = results_by_id.get(record_id)
            if current is None or score > current["score"]:
                results_by_id[record_id] = {
                    "id": record_id,
                    "document": doc,
                    "metadata": meta,
                    "score": score,
                }

    return sorted(results_by_id.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def format_result(rank: int, item: Dict[str, Any]) -> str:
    meta = item["metadata"]
    lines = [
        f"#{rank} | Score: {item['score']:.3f}",
        f"Movie: {meta.get('movie_title', 'Unknown')} ({meta.get('year', 'n/a')})",
        f"Time: {meta.get('start_time')} - {meta.get('end_time')}",
        f"Quote: {meta.get('line_text')}",
    ]
    before = meta.get("context_before") or ""
    after = meta.get("context_after") or ""
    if before or after:
        lines.append("Context:")
        if before:
            lines.append(f"  ... {before}")
        if after:
            lines.append(f"  {after} ...")
    return "\n".join(lines)


def save_favorites(query: str, results: List[Dict[str, Any]]) -> None:
    payload = {
        "query": query,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }
    FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FAVORITES_PATH.exists():
        data = json.loads(FAVORITES_PATH.read_text(encoding="utf-8"))
    else:
        data = []
    data.append(payload)
    FAVORITES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def export_results(path: Path, results: List[Dict[str, Any]]) -> None:
    if path.suffix.lower() == ".csv":
        import csv

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["rank", "score", "movie_title", "year", "start_time", "end_time", "quote"],
            )
            writer.writeheader()
            for idx, item in enumerate(results, start=1):
                meta = item["metadata"]
                writer.writerow(
                    {
                        "rank": idx,
                        "score": f"{item['score']:.3f}",
                        "movie_title": meta.get("movie_title"),
                        "year": meta.get("year"),
                        "start_time": meta.get("start_time"),
                        "end_time": meta.get("end_time"),
                        "quote": meta.get("line_text"),
                    }
                )
    else:
        lines = ["# Search results", ""]
        for idx, item in enumerate(results, start=1):
            lines.append(format_result(idx, item))
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed subtitles.")
    parser.add_argument("query", help="Search query (English or Russian).")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return.")
    parser.add_argument("--movie", help="Filter by movie title.")
    parser.add_argument("--year", help="Filter by year (YYYY).")
    parser.add_argument("--folder", help="Filter by folder path (director or collection).")
    parser.add_argument(
        "--record-type",
        choices=["line_window", "scene_chunk"],
        help="Filter by record type.",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="Generate English paraphrases of the query.",
    )
    parser.add_argument(
        "--paraphrase-count",
        type=int,
        default=5,
        help="Number of paraphrases to generate.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export results to .md or .csv file.",
    )
    parser.add_argument(
        "--save-favorites",
        action="store_true",
        help="Save results to data/favorites.json.",
    )
    args = parser.parse_args()

    if not INDEX_DIR.exists():
        logging.info("Index not found. Run: python -m app.index")
        return

    chroma = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = chroma.get_or_create_collection(name=CHROMA_COLLECTION)

    client = OpenAI()
    queries = [args.query]
    if args.paraphrase:
        paraphrases = generate_paraphrases(client, args.query, args.paraphrase_count)
        if paraphrases:
            logging.info("Using %s paraphrases.", len(paraphrases))
            queries.extend(paraphrases)

    embeddings = embed_queries(client, queries)
    filters = build_filters(args)
    results = query_index(collection, embeddings, args.top_k, filters)

    if not results:
        logging.info("No results found.")
        return

    for idx, item in enumerate(results, start=1):
        print(format_result(idx, item))
        print("-" * 60)

    if args.export:
        export_results(args.export, results)
        logging.info("Exported results to %s", args.export)

    if args.save_favorites:
        save_favorites(args.query, results)
        logging.info("Saved results to %s", FAVORITES_PATH)


if __name__ == "__main__":
    main()
