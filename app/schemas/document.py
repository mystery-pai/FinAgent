"""
Document and chunk schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class Document(BaseModel):
    """Standardized 10-K document"""

    doc_id: str = Field(description="{year}_{section_id}")
    symbol: str
    year: int
    form_type: str
    section_id: int
    section_title: str
    item_type: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Text chunk for retrieval"""

    chunk_id: str = Field(description="{year}_{section_id}_{chunk_order}")
    doc_id: str
    year: int
    section_title: str
    item_type: str
    chunk_order: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Retrieval result with score"""

    chunk_id: str
    text: str
    score: float
    year: int
    section_title: str
    item_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Query request from user"""

    query: str
    top_k: Optional[int] = 5
    year_filter: Optional[int] = None
    item_type_filter: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response to user"""

    answer: str
    citations: list[Dict[str, Any]]
    retrieval_debug: Dict[str, Any]
