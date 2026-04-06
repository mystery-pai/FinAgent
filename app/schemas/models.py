"""
Data models for the financial RAG system.
数据模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DocumentChunk(BaseModel):
    """Represents a chunk of financial document"""

    doc_id: str = Field(description="Unique document identifier: {year}_{section_id}")
    symbol: str = Field(description="Stock symbol, e.g., AAPL")
    year: int = Field(description="Fiscal year")
    form_type: str = Field(description="Form type, e.g., 10-K")
    section_id: str = Field(description="Section identifier")
    section_title: str = Field(description="Section title")
    item_type: str = Field(
        description="Normalized item type: business, risk_factors, md&a, etc."
    )
    text: str = Field(description="Chunk text content")
    chunk_id: int = Field(description="Chunk index within section")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    """Document retrieved from search"""

    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str  # 'bm25', 'vector', or 'hybrid'


class QueryRequest(BaseModel):
    """User query request"""

    question: str = Field(description="User question in Chinese or English")
    max_results: int = Field(default=5, description="Maximum number of results")
    include_citations: bool = Field(default=True, description="Include source citations")
    session_id: Optional[str] = Field(default=None, description="Conversation session identifier")


class ConversationTurn(BaseModel):
    """Single turn in a conversation session"""

    question: str = Field(description="User question")
    answer: str = Field(description="Assistant answer")


class Citation(BaseModel):
    """Source citation for answer"""

    year: int
    section_title: str
    chunk_id: Optional[str] = None
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Query response with answer and citations"""

    session_id: str = Field(description="Conversation session identifier")
    answer: str = Field(description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Recent conversation turns",
    )
    retrieval_debug: Optional[Dict[str, Any]] = Field(
        default=None, description="Debug information for retrieval process"
    )


class ReportRequest(BaseModel):
    """Request for generating a summary report"""

    topic: str = Field(description="Report topic")
    year_start: int = Field(description="Start year (inclusive)")
    year_end: int = Field(description="End year (inclusive)")
    focus_areas: List[str] = Field(
        default_factory=list,
        description="Specific areas to focus on, e.g., ['risk_factors', 'md&a']",
    )


class ReportResponse(BaseModel):
    """Generated report response"""

    topic: str
    year_range: str
    summary: str
    key_findings: List[str]
    citations: List[Citation]


# ==================== Visualization Models ====================


class ChartSeries(BaseModel):
    """Single data series for chart"""
    name: str = Field(description="Series name (e.g., 'Revenue', 'Net Income')")
    values: List[float] = Field(description="Data values")
    unit: Optional[str] = Field(None, description="Unit (e.g., 'Million USD', 'Billion USD')")


class ChartDataSchema(BaseModel):
    """
    Structured chart data extracted from documents

    This schema is used for LLM Function Calling to ensure structured output
    """
    title: str = Field(description="Chart title")
    x_label: str = Field(description="X-axis label (e.g., 'Year', 'Quarter')")
    y_label: str = Field(description="Y-axis label (e.g., 'Revenue', 'Amount')")
    x_values: List[Any] = Field(description="X-axis values (e.g., [2023, 2024, 2025])")
    series: List[ChartSeries] = Field(description="Data series")
    chart_type_hint: Optional[str] = Field(
        None,
        description="Suggested chart type based on data (line, bar, pie, scatter)"
    )
    data_source: Optional[str] = Field(
        None,
        description="Data source citation (e.g., '10-K 2024, Item 8')"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Apple Revenue Trend 2023-2025",
                "x_label": "Year",
                "y_label": "Revenue",
                "x_values": [2023, 2024, 2025],
                "series": [
                    {
                        "name": "Revenue",
                        "values": [383.285, 391.035, 401.610],
                        "unit": "Billion USD"
                    }
                ],
                "chart_type_hint": "line",
                "data_source": "10-K Reports, Item 8"
            }
        }


class VisualizationRequest(BaseModel):
    """Request for visualization endpoint"""
    question: str = Field(description="User question for visualization")
    chart_type: str = Field(
        default="auto",
        description="Chart type: auto, line, bar, pie, grouped_bar"
    )
    engine: str = Field(
        default="plotly",
        description="Visualization engine: plotly or echarts"
    )
    session_id: Optional[str] = Field(
        None,
        description="Conversation session ID"
    )
    max_results: int = Field(
        default=10,
        ge=1, le=50,
        description="Max documents to retrieve"
    )


class VisualizationResponse(BaseModel):
    """Response from visualization endpoint"""
    session_id: str
    chart_html: str = Field(description="Chart HTML for embedding")
    chart_json: Dict[str, Any] = Field(description="Chart JSON for programmatic use")
    chart_data: Dict[str, Any] = Field(description="Raw extracted data")
    analysis: str = Field(description="Text analysis of the data")
    citations: List[Citation] = Field(description="Source citations")
    chart_type: str = Field(description="Actual chart type used")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (retrieval debug, etc.)"
    )


class IndexBuildRequest(BaseModel):
    """Request to build search indices"""
    force_rebuild: bool = Field(
        default=False,
        description="Force rebuild even if indices exist"
    )


class IndexStatus(BaseModel):
    """Status of search indices"""
    indexed: bool = Field(description="Whether documents are indexed")
    document_count: int = Field(description="Number of indexed documents")
    bm25_indexed: bool = Field(description="BM25 index status")
    chroma_indexed: bool = Field(description="ChromaDB index status")
