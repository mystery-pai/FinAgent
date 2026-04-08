"""
Shared workflows for UI and CLI visualization flows.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional


def build_visualization_filters(year_filter: Optional[int]) -> Optional[Dict[str, Any]]:
    """Build optional retrieval filters for visualization."""
    if year_filter is None:
        return None
    return {"year": year_filter}


def execute_visualization_request(
    agent: Any,
    question: str,
    session_id: str,
    chart_type: str = "auto",
    max_results: int = 10,
    year_filter: Optional[int] = None,
    engine: str = "plotly",
) -> Dict[str, Any]:
    """Execute visualization synchronously for UI-style callers."""
    filters = build_visualization_filters(year_filter)
    result = asyncio.run(
        agent.execute(
            question=question,
            chart_type=chart_type,
            session_id=session_id,
            max_results=max_results,
            engine=engine,
            filters=filters,
        )
    )

    if not result.success:
        raise ValueError(result.error or "Visualization failed")

    payload = result.data
    metadata = payload.get("metadata", result.metadata or {})

    return {
        "mode": "visualization",
        "question": question,
        "answer": payload["analysis"],
        "citations": payload.get("citations", []),
        "chart_html": payload["chart_html"],
        "chart_json": payload["chart_json"],
        "chart_data": payload["chart_data"],
        "chart_type": payload["chart_type"],
        "debug_info": {
            "session_id": session_id,
            "engine": engine,
            "requested_chart_type": chart_type,
            "year_filter": year_filter,
            **metadata,
        },
    }


def write_chart_html(chart_html: str, output_path: Path) -> Path:
    """Persist chart HTML to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(chart_html, encoding="utf-8")
    return output_path
