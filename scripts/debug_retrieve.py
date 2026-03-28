#!/usr/bin/env python3
"""
Local CLI for retrieval debugging.
本地检索调试脚本
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieve.bm25_retriever import BM25Retriever
from app.retrieve.chroma_retriever import ChromaRetriever
from app.retrieve.hybrid_retriever import HybridRetriever
from app.schemas.models import RetrievedDocument


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Debug local retrieval results from the terminal.",
    )
    parser.add_argument("query", help="Query text to search")
    parser.add_argument(
        "--mode",
        choices=["bm25", "vector", "hybrid"],
        default="hybrid",
        help="Retriever mode to use",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Optional year filter",
    )
    parser.add_argument(
        "--item-type",
        help="Optional item_type filter, e.g. financial_statements",
    )
    parser.add_argument(
        "--text-len",
        type=int,
        default=240,
        help="Preview length for result text",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw results as JSON",
    )
    return parser.parse_args()


def build_filters(args: argparse.Namespace) -> Dict[str, Any]:
    """Build optional metadata filters."""
    filters: Dict[str, Any] = {}

    if args.year is not None:
        filters["year"] = args.year

    if args.item_type:
        filters["item_type"] = args.item_type

    return filters


def load_retriever(mode: str):
    """Load the requested retriever."""
    if mode == "bm25":
        return BM25Retriever()
    if mode == "vector":
        return ChromaRetriever()
    return HybridRetriever()


def retrieve(
    mode: str,
    query: str,
    top_k: int,
    filters: Dict[str, Any],
) -> Tuple[List[RetrievedDocument], Dict[str, Any]]:
    """Run retrieval and normalize outputs."""
    retriever = load_retriever(mode)

    if mode == "hybrid":
        return retriever.retrieve(query=query, k=top_k, filters=filters or None)

    results = retriever.retrieve(query=query, k=top_k, filters=filters or None)
    return results, {"mode": mode, "filters_applied": filters}


def to_json_ready(results: List[RetrievedDocument]) -> List[Dict[str, Any]]:
    """Convert results to serializable dictionaries."""
    return [result.model_dump() for result in results]


def render_result(index: int, result: RetrievedDocument, text_len: int) -> None:
    """Print one retrieval result."""
    metadata = result.metadata
    preview = result.text[:text_len].replace("\n", " / ")

    print(f"[{index}] score={result.score:.4f} method={result.retrieval_method}")
    print(
        "    "
        f"doc_id={result.doc_id} "
        f"year={metadata.get('year', 'N/A')} "
        f"item_type={metadata.get('item_type', 'N/A')} "
        f"chunk_id={metadata.get('chunk_id', 'N/A')}"
    )
    print(f"    section={metadata.get('section_title', 'N/A')}")
    print(f"    text={preview}")


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    filters = build_filters(args)

    try:
        results, debug_info = retrieve(
            mode=args.mode,
            query=args.query,
            top_k=args.top_k,
            filters=filters,
        )
    except Exception as exc:
        print(f"Retrieve failed: {exc}", file=sys.stderr)
        if args.mode in {"vector", "hybrid"}:
            print(
                "Hint: vector/hybrid modes need a usable embedding model. "
                "If the model is not cached locally, fix proxy support first "
                "(for example install `httpx[socks]` / `socksio`) or switch to `--mode bm25`.",
                file=sys.stderr,
            )
        return 1

    if args.json:
        payload = {
            "query": args.query,
            "mode": args.mode,
            "filters": filters,
            "debug": debug_info,
            "results": to_json_ready(results),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"Filters: {filters or '{}'}")

    if debug_info:
        print("Debug:")
        print(json.dumps(debug_info, ensure_ascii=False, indent=2))

    print(f"Results: {len(results)}")
    print("-" * 80)

    for index, result in enumerate(results, start=1):
        render_result(index=index, result=result, text_len=args.text_len)
        print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
