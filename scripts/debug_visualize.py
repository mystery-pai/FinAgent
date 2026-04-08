#!/usr/bin/env python3
"""
Local CLI for visualization debugging.
本地可视化调试脚本
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.base import AgentConfig
from app.agents.visualization_agent import VisualizationAgent
from app.ui.workflows import build_visualization_filters, write_chart_html


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Debug local visualization generation from the terminal.",
    )
    parser.add_argument("question", help="Question to visualize")
    parser.add_argument(
        "--chart-type",
        choices=["auto", "line", "bar", "grouped_bar", "pie"],
        default="auto",
        help="Chart type to request",
    )
    parser.add_argument(
        "--max-results",
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
        "--output-html",
        type=Path,
        default=Path("test_output/debug_visualization.html"),
        help="Path to save generated chart HTML",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw visualization payload as JSON",
    )
    return parser.parse_args()


def create_visualization_agent() -> VisualizationAgent:
    """Create visualization agent for CLI debugging."""
    return VisualizationAgent(
        config=AgentConfig(
            name="visualization",
            description="Visualization agent for local debugging",
        )
    )


def run_visualization(
    question: str,
    chart_type: str,
    max_results: int,
    year: int | None,
) -> Dict[str, Any]:
    """Run visualization request and return raw payload."""
    from app.ui.workflows import execute_visualization_request

    agent = create_visualization_agent()
    return execute_visualization_request(
        agent=agent,
        question=question,
        session_id="debug-visualization",
        chart_type=chart_type,
        max_results=max_results,
        year_filter=year,
    )


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    try:
        result = run_visualization(
            question=args.question,
            chart_type=args.chart_type,
            max_results=args.max_results,
            year=args.year,
        )
    except Exception as exc:
        print(f"Visualization failed: {exc}", file=sys.stderr)
        print(
            "Hint: ensure indices are built and DEEPSEEK_API_KEY is configured.",
            file=sys.stderr,
        )
        return 1

    output_path = write_chart_html(result["chart_html"], args.output_html)

    if args.json:
        payload = {
            "question": args.question,
            "filters": build_visualization_filters(args.year),
            "output_html": str(output_path),
            **result,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Question: {args.question}")
    print(f"Chart Type: {result['chart_type']}")
    print(f"Filters: {build_visualization_filters(args.year) or '{}'}")
    print(f"Saved HTML: {output_path}")
    print("Analysis:")
    print(result["answer"])
    print("Debug:")
    print(json.dumps(result["debug_info"], ensure_ascii=False, indent=2))
    print(f"Citations: {len(result['citations'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
