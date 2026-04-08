from pathlib import Path

from app.agents.base import AgentResult


class StubVisualizationAgent:
    async def execute(
        self,
        question,
        chart_type="auto",
        session_id=None,
        max_results=10,
        engine="plotly",
        filters=None,
        **kwargs,
    ):
        return AgentResult(
            agent_name="visualization",
            success=True,
            data={
                "session_id": session_id,
                "chart_html": "<div>chart</div>",
                "chart_json": {"data": [{"type": "bar"}]},
                "chart_data": {"title": "Apple Revenue"},
                "analysis": "2024 年苹果营收高于 2023 年。",
                "citations": [{"year": 2024, "section_title": "Item 8"}],
                "chart_type": "bar",
                "metadata": {"retrieval_debug": {"final_count": 2}},
            },
            metadata={"retrieval_debug": {"final_count": 2}},
        )


class FailingVisualizationAgent:
    async def execute(self, **kwargs):
        return AgentResult(
            agent_name="visualization",
            success=False,
            data={},
            error="visualization failed",
        )


def test_execute_visualization_request_builds_visualization_turn():
    from app.ui.workflows import execute_visualization_request

    turn = execute_visualization_request(
        agent=StubVisualizationAgent(),
        question="显示苹果营收趋势",
        session_id="session-1",
        chart_type="auto",
        max_results=5,
        year_filter=2024,
    )

    assert turn["mode"] == "visualization"
    assert turn["question"] == "显示苹果营收趋势"
    assert turn["answer"] == "2024 年苹果营收高于 2023 年。"
    assert turn["chart_html"] == "<div>chart</div>"
    assert turn["chart_type"] == "bar"
    assert turn["debug_info"]["year_filter"] == 2024
    assert turn["debug_info"]["retrieval_debug"]["final_count"] == 2


def test_execute_visualization_request_raises_for_failed_agent():
    from app.ui.workflows import execute_visualization_request

    try:
        execute_visualization_request(
            agent=FailingVisualizationAgent(),
            question="显示苹果营收趋势",
            session_id="session-1",
        )
    except ValueError as exc:
        assert str(exc) == "visualization failed"
    else:
        raise AssertionError("Expected ValueError for failed visualization result")


def test_write_chart_html_persists_output(tmp_path: Path):
    from app.ui.workflows import write_chart_html

    output_path = tmp_path / "chart.html"
    written_path = write_chart_html("<div>chart</div>", output_path)

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8") == "<div>chart</div>"
