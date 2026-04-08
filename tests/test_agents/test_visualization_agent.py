import asyncio

from app.agents.base import AgentConfig
from app.schemas.models import ChartDataSchema, ChartSeries, RetrievedDocument


class StubHybridRetriever:
    def retrieve(self, query, k, filters=None):
        return [
            RetrievedDocument(
                doc_id="2024_item8_0",
                text="Revenue was 391.0 billion USD in 2024.",
                score=0.92,
                metadata={
                    "year": 2024,
                    "section_title": "Item 8. Financial Statements",
                    "doc_id": "2024_item8",
                    "chunk_id": 0,
                },
                retrieval_method="hybrid",
            )
        ], {"final_count": 1, "query_type": "factual"}


class EmptyHybridRetriever:
    def retrieve(self, query, k, filters=None):
        return [], {"final_count": 0, "query_type": "factual"}


class StubDataExtractor:
    def extract(self, question, documents, max_context_length=4000):
        return ChartDataSchema(
            title="Apple Revenue",
            x_label="Year",
            y_label="Revenue",
            x_values=[2023, 2024],
            series=[
                ChartSeries(
                    name="Revenue",
                    values=[383.3, 391.0],
                    unit="Billion USD",
                )
            ],
            chart_type_hint="bar",
            data_source="10-K 2024 Item 8",
        )


class StubFigure:
    def to_html(self, full_html=False, include_plotlyjs="cdn"):
        return "<div>chart</div>"

    def to_plotly_json(self):
        return {"data": [{"type": "bar"}], "layout": {"title": {"text": "Apple Revenue"}}}


class StubChartGenerator:
    def resolve_chart_type(self, data, chart_type="auto"):
        return "bar"

    def generate(self, data, chart_type="auto"):
        return StubFigure()


def test_visualization_agent_returns_chart_payload():
    from app.agents.visualization_agent import VisualizationAgent

    agent = VisualizationAgent(
        config=AgentConfig(name="visualization", description="Visualization agent"),
        retriever=StubHybridRetriever(),
        data_extractor=StubDataExtractor(),
        chart_generator=StubChartGenerator(),
    )

    result = asyncio.run(
        agent.execute(
            question="显示苹果营收趋势",
            chart_type="auto",
            session_id="session-1",
            max_results=5,
        )
    )

    assert result.success is True
    assert result.data["chart_html"] == "<div>chart</div>"
    assert result.data["chart_json"]["data"][0]["type"] == "bar"
    assert result.data["chart_type"] == "bar"
    assert result.data["chart_data"]["title"] == "Apple Revenue"
    assert len(result.data["citations"]) == 1
    assert "2024" in result.data["analysis"]
    assert result.metadata["retrieval_debug"]["final_count"] == 1


def test_visualization_agent_fails_when_no_documents_found():
    from app.agents.visualization_agent import VisualizationAgent

    agent = VisualizationAgent(
        config=AgentConfig(name="visualization", description="Visualization agent"),
        retriever=EmptyHybridRetriever(),
        data_extractor=StubDataExtractor(),
        chart_generator=StubChartGenerator(),
    )

    result = asyncio.run(agent.execute(question="显示苹果营收趋势"))

    assert result.success is False
    assert result.error == "No relevant documents found for visualization"
