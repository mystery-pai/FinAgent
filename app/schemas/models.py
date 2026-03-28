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


class Citation(BaseModel):
    """Source citation for answer"""

    year: int
    section_title: str
    chunk_id: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Query response with answer and citations"""

    answer: str = Field(description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
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
