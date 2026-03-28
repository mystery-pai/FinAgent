#!/usr/bin/env python3
"""
Local CLI for answer debugging.
本地问答调试脚本
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.generate.answer_generator import AnswerGenerator, PromptTemplate, QueryTranslator
from app.retrieve.bm25_retriever import BM25Retriever
from app.retrieve.chroma_retriever import ChromaRetriever
from app.retrieve.hybrid_retriever import HybridRetriever, QueryParser
from app.schemas.models import Citation, RetrievedDocument


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Debug local answer generation from the terminal.",
    )
    parser.add_argument("question", help="Question to answer")
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
        help="Number of retrieved documents to use",
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
        default=180,
        help="Preview length for retrieved context",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Disable query translation before retrieval",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable the configured LLM and use the local fallback answer",
    )
    parser.add_argument(
        "--show-retrieved",
        action="store_true",
        help="Print retrieved documents before the answer",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw answer payload as JSON",
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

    parser = QueryParser()
    parsed_query = parser.parse(query)
    results = retriever.retrieve(query=query, k=top_k, filters=filters or None)
    debug_info = {
        "mode": mode,
        "original_query": query,
        "normalized_query": parsed_query["normalized_query"],
        "query_type": parsed_query["query_type"],
        "extracted_years": parsed_query["years"],
        "extracted_item_types": parsed_query["item_types"],
        "query_hints": parsed_query["query_hints"],
        "filters_applied": filters,
        "final_count": len(results),
    }
    return results, debug_info


def render_retrieved_docs(results: List[RetrievedDocument], text_len: int) -> None:
    """Print retrieved documents."""
    print("Retrieved:")
    print("-" * 80)

    for index, result in enumerate(results, start=1):
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
        print("-" * 80)


def citations_to_dicts(citations: List[Citation]) -> List[Dict[str, Any]]:
    """Convert citations to dictionaries."""
    return [citation.model_dump() for citation in citations]


def create_generator(use_llm: bool, use_translate: bool) -> AnswerGenerator:
    """Create answer generator without forcing remote client initialization."""
    if use_llm or use_translate:
        return AnswerGenerator()

    translator = QueryTranslator.__new__(QueryTranslator)
    translator.client = None

    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.client = None
    generator.query_translator = translator
    generator.templates = PromptTemplate()
    return generator


def prepare_client_env() -> None:
    """
    Avoid forcing SOCKS proxy initialization when HTTP proxies are already set.
    """
    if os.environ.get("http_proxy") or os.environ.get("https_proxy"):
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    filters = build_filters(args)
    use_llm = not args.no_llm
    use_translate = not args.no_translate

    prepare_client_env()

    try:
        generator = create_generator(
            use_llm=use_llm,
            use_translate=use_translate,
        )
    except Exception as exc:
        print(f"Answer generator init failed: {exc}", file=sys.stderr)
        print(
            "Hint: default mode uses the configured remote client. "
            "For pure local debugging, run with `--no-translate --no-llm`.",
            file=sys.stderr,
        )
        return 1

    retrieval_query = args.question
    if use_translate:
        retrieval_query = generator.translate_query(args.question)

    try:
        retrieved_docs, debug_info = retrieve(
            mode=args.mode,
            query=retrieval_query,
            top_k=args.top_k,
            filters=filters,
        )
    except Exception as exc:
        print(f"Retrieve failed: {exc}", file=sys.stderr)
        if args.mode in {"vector", "hybrid"}:
            print(
                "Hint: vector/hybrid modes need a usable embedding model. "
                "If the model is not cached locally, fix proxy support first "
                "(for example install `httpx[socks]` / `socksio`).",
                file=sys.stderr,
            )
        return 1

    query_type = debug_info.get("query_type", "factual")
    answer, citations = generator.generate(
        question=args.question,
        retrieved_docs=retrieved_docs,
        query_type=query_type,
        debug_info=debug_info,
    )

    if args.json:
        payload = {
            "question": args.question,
            "retrieval_query": retrieval_query,
            "mode": args.mode,
            "filters": filters,
            "use_llm": use_llm,
            "translated": use_translate,
            "debug": debug_info,
            "citations": citations_to_dicts(citations),
            "results": [doc.model_dump() for doc in retrieved_docs],
            "answer": answer,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Question: {args.question}")
    print(f"Retrieval Query: {retrieval_query}")
    print(f"Mode: {args.mode}")
    print(f"Filters: {filters or '{}'}")
    print(f"Use LLM: {use_llm}")
    print(f"Translate: {use_translate}")
    print("Debug:")
    print(json.dumps(debug_info, ensure_ascii=False, indent=2))

    if args.show_retrieved:
        render_retrieved_docs(results=retrieved_docs, text_len=args.text_len)

    print("Answer:")
    print(answer)

    if citations:
        print("\nCitations:")
        for index, citation in enumerate(citations, start=1):
            data = citation.model_dump()
            print(
                f"[{index}] year={data.get('year')} "
                f"section={data.get('section_title')} "
                f"chunk_id={data.get('chunk_id')} "
                f"score={data.get('relevance_score', 0):.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
