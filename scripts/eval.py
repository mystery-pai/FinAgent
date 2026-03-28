#!/usr/bin/env python3
"""
Evaluate retrieval and generation quality with RAGAS.
评估检索与生成质量
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.eval.ragas_evaluator import RagasEvaluator


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run RAG evaluation for Fin-Agent.")
    parser.add_argument(
        "--dataset",
        default="tests/eval_questions.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--mode",
        choices=["bm25", "vector", "hybrid"],
        default="hybrid",
        help="Retriever mode to evaluate",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval depth",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["hit_rate", "mrr", "faithfulness"],
        default=["hit_rate", "mrr", "faithfulness"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save JSON results",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    evaluator = RagasEvaluator(
        mode=args.mode,
        top_k=args.top_k,
        metrics=args.metrics,
    )

    try:
        result = evaluator.evaluate_dataset(args.dataset)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Summary:")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("\nSamples:")
    for sample in result["samples"]:
        print("-" * 80)
        print(f"Question: {sample['question']}")
        print(f"Hit Rate: {sample.get('hit_rate', 0.0):.4f}")
        print(f"MRR: {sample.get('mrr', 0.0):.4f}")
        if sample.get("faithfulness") is not None:
            print(f"Faithfulness: {sample['faithfulness']:.4f}")
        print(f"First Relevant Rank: {sample.get('first_relevant_rank')}")
        print(f"Retrieval Query: {sample['retrieval_query']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
