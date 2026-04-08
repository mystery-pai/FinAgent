from fastapi.testclient import TestClient

import app.api.main as api_main
from app.agents.base import AgentResult
from app.schemas.models import Citation


class StubBM25Retriever:
    bm25_index = object()


class StubChromaRetriever:
    def get_collection_stats(self):
        return {"count": 1}


class StubAnswerGenerator:
    pass


class StubConversationManager:
    def __init__(self):
        self.turns = []

    def ensure_session(self, session_id):
        return session_id or "session-test"

    def append_turn(self, session_id, question, answer):
        self.turns.append((session_id, question, answer))
        return []


class StubHybridRetriever:
    def __init__(self, bm25_retriever=None, chroma_retriever=None):
        self.bm25_retriever = bm25_retriever
        self.chroma_retriever = chroma_retriever


class StubRouterAgent:
    def __init__(self, config, llm_client=None):
        self.config = config


class StubDataExtractor:
    pass


class StubChartGenerator:
    pass


class StubVisualizationAgent:
    def __init__(self, config, retriever, data_extractor, chart_generator):
        self.retriever = retriever

    async def execute(
        self,
        question,
        chart_type="auto",
        session_id=None,
        max_results=10,
        engine="plotly",
        **kwargs,
    ):
        return AgentResult(
            agent_name="visualization",
            success=True,
            data={
                "session_id": session_id or "session-test",
                "chart_html": "<div>chart</div>",
                "chart_json": {"data": [{"type": "line"}]},
                "chart_data": {"title": "Apple Revenue"},
                "analysis": "2024 年苹果营收高于 2023 年。",
                "citations": [
                    Citation(
                        year=2024,
                        section_title="Item 8. Financial Statements",
                        chunk_id="2024_item8_0",
                        relevance_score=0.9,
                    ).model_dump()
                ],
                "chart_type": "line",
                "metadata": {"retrieval_debug": {"final_count": 1}},
            },
            metadata={"retrieval_debug": {"final_count": 1}},
        )


class StubAgentOrchestrator:
    def __init__(self, router, visualization_agent=None, qa_agent=None, report_agent=None):
        self.visualization_agent = visualization_agent

    async def execute_visualization(
        self,
        question,
        chart_type="auto",
        session_id=None,
        max_results=10,
        engine="plotly",
        **kwargs,
    ):
        return await self.visualization_agent.execute(
            question=question,
            chart_type=chart_type,
            session_id=session_id,
            max_results=max_results,
            engine=engine,
            **kwargs,
        )


def test_visualize_endpoint_returns_visualization_response(monkeypatch):
    api_main.components.clear()

    monkeypatch.setattr(api_main, "BM25Retriever", StubBM25Retriever)
    monkeypatch.setattr(api_main, "ChromaRetriever", StubChromaRetriever)
    monkeypatch.setattr(api_main, "AnswerGenerator", StubAnswerGenerator)
    monkeypatch.setattr(api_main, "ConversationManager", StubConversationManager)
    monkeypatch.setattr(api_main, "HybridRetriever", StubHybridRetriever)
    monkeypatch.setattr(api_main, "RouterAgent", StubRouterAgent, raising=False)
    monkeypatch.setattr(api_main, "VisualizationAgent", StubVisualizationAgent, raising=False)
    monkeypatch.setattr(api_main, "AgentOrchestrator", StubAgentOrchestrator, raising=False)
    monkeypatch.setattr(api_main, "DataExtractor", StubDataExtractor, raising=False)
    monkeypatch.setattr(api_main, "ChartGenerator", StubChartGenerator, raising=False)

    with TestClient(api_main.app) as client:
        response = client.post(
            "/visualize",
            json={
                "question": "显示苹果营收趋势",
                "chart_type": "auto",
                "engine": "plotly",
                "max_results": 5,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chart_html"] == "<div>chart</div>"
    assert payload["chart_type"] == "line"
    assert payload["metadata"]["retrieval_debug"]["final_count"] == 1
