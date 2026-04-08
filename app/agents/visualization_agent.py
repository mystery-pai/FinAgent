"""
Visualization agent for end-to-end chart generation.
"""

from typing import Any, Dict, List, Optional
import logging

from app.agents.base import BaseAgent, AgentConfig
from app.core.config import settings
from app.retrieve.hybrid_retriever import HybridRetriever
from app.schemas.models import ChartDataSchema, Citation, RetrievedDocument
from app.tools.chart_generator import ChartGenerator
from app.tools.data_extractor import DataExtractor

logger = logging.getLogger(__name__)


class VisualizationAgent(BaseAgent):
    """Build charts from retrieved financial documents."""

    SUPPORTED_CHART_TYPES = {"auto", "line", "bar", "grouped_bar", "pie"}
    SUPPORTED_ENGINES = {"plotly"}

    def __init__(
        self,
        config: AgentConfig,
        retriever: Optional[HybridRetriever] = None,
        data_extractor: Optional[DataExtractor] = None,
        chart_generator: Optional[ChartGenerator] = None,
    ):
        super().__init__(config)
        self.retriever = retriever or HybridRetriever()
        self.data_extractor = data_extractor or DataExtractor()
        self.chart_generator = chart_generator or ChartGenerator()

    async def execute(
        self,
        question: str,
        chart_type: str = "auto",
        session_id: Optional[str] = None,
        max_results: int = 10,
        engine: str = "plotly",
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Execute the visualization pipeline."""
        try:
            self.validate_input(
                question=question,
                chart_type=chart_type,
                engine=engine,
                max_results=max_results,
            )

            retrieved_docs, debug_info = self.retriever.retrieve(
                query=question,
                k=max_results,
                filters=filters,
            )

            metadata = {
                "session_id": session_id,
                "retrieval_debug": debug_info,
                "document_count": len(retrieved_docs),
            }

            if not retrieved_docs:
                return self._create_result(
                    success=False,
                    data={},
                    error="No relevant documents found for visualization",
                    metadata=metadata,
                )

            chart_data = self.data_extractor.extract(
                question=question,
                documents=retrieved_docs,
                max_context_length=settings.max_chart_data_points * 4,
            )
            actual_chart_type = self._resolve_chart_type(chart_data, chart_type)
            figure = self.chart_generator.generate(chart_data, chart_type=actual_chart_type)
            citations = self._extract_citations(retrieved_docs)
            analysis = self._build_analysis(chart_data, actual_chart_type)

            payload = {
                "session_id": session_id,
                "chart_html": figure.to_html(full_html=False, include_plotlyjs="cdn"),
                "chart_json": figure.to_plotly_json(),
                "chart_data": chart_data.model_dump(),
                "analysis": analysis,
                "citations": [citation.model_dump() for citation in citations],
                "chart_type": actual_chart_type,
                "metadata": metadata,
            }

            return self._create_result(
                success=True,
                data=payload,
                metadata=metadata,
            )

        except Exception as exc:
            logger.error("Visualization pipeline failed: %s", exc, exc_info=True)
            return self._create_result(
                success=False,
                data={},
                error=str(exc),
            )

    def validate_input(
        self,
        question: str,
        chart_type: str = "auto",
        engine: str = "plotly",
        max_results: int = 10,
        **kwargs,
    ) -> bool:
        """Validate visualization input."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if chart_type not in self.SUPPORTED_CHART_TYPES:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        if engine not in self.SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported visualization engine: {engine}")

        if max_results <= 0:
            raise ValueError("max_results must be greater than 0")

        return True

    def _resolve_chart_type(self, chart_data: ChartDataSchema, requested_chart_type: str) -> str:
        """Resolve the final chart type for the current request."""
        if hasattr(self.chart_generator, "resolve_chart_type"):
            return self.chart_generator.resolve_chart_type(chart_data, requested_chart_type)

        if requested_chart_type != "auto":
            return requested_chart_type

        return self.chart_generator._detect_chart_type(chart_data)

    def _extract_citations(self, documents: List[RetrievedDocument]) -> List[Citation]:
        """Convert retrieved documents into citations."""
        citations = []

        for doc in documents:
            metadata = doc.metadata
            citations.append(
                Citation(
                    year=metadata.get("year", 0),
                    section_title=metadata.get("section_title", ""),
                    chunk_id=f"{metadata.get('doc_id', '')}_{metadata.get('chunk_id', 0)}",
                    relevance_score=doc.score,
                )
            )

        return citations

    def _build_analysis(self, chart_data: ChartDataSchema, chart_type: str) -> str:
        """Build a deterministic analysis summary for the generated chart."""
        summary_parts = [f"已生成 {chart_type} 图表，展示 {chart_data.title}。"]

        if not chart_data.series or not chart_data.x_values:
            return "".join(summary_parts)

        start_label = chart_data.x_values[0]
        end_label = chart_data.x_values[-1]

        for series in chart_data.series:
            latest_value = series.values[-1]
            unit_suffix = f" {series.unit}" if series.unit else ""
            series_summary = (
                f"{series.name} 在 {end_label} 为 {self._format_value(latest_value)}"
                f"{unit_suffix}"
            )

            if len(series.values) >= 2:
                first_value = series.values[0]
                delta = latest_value - first_value
                direction = "增长" if delta >= 0 else "下降"
                series_summary += (
                    f"，较 {start_label} {direction} "
                    f"{self._format_value(abs(delta))}{unit_suffix}"
                )
                if first_value not in (0, 0.0):
                    percentage = abs(delta) / abs(first_value) * 100
                    series_summary += f"（{percentage:.2f}%）"

            summary_parts.append(series_summary + "。")

        if chart_data.data_source:
            summary_parts.append(f"数据来源：{chart_data.data_source}。")

        return "".join(summary_parts)

    def _format_value(self, value: Any) -> str:
        """Format numeric values for readable analysis text."""
        if isinstance(value, int):
            return f"{value:,}"

        if isinstance(value, float):
            return f"{value:,.2f}".rstrip("0").rstrip(".")

        return str(value)
