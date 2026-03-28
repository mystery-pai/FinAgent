"""
FastAPI application for financial Q&A system.
"""
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.schemas.models import QueryRequest, QueryResponse, ReportRequest, ReportResponse
from app.ingest.preprocessor import FinancialDataProcessor
from app.retrieve.bm25_retriever import BM25Retriever
from app.retrieve.chroma_retriever import ChromaRetriever
from app.retrieve.hybrid_retriever import HybridRetriever
from app.generate.answer_generator import AnswerGenerator
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global components
components: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting up financial Q&A system...")
    try:
        # Initialize components
        components["bm25_retriever"] = BM25Retriever()
        components["chroma_retriever"] = ChromaRetriever()
        components["answer_generator"] = AnswerGenerator()
        components["hybrid_retriever"] = HybridRetriever(
            bm25_retriever=components["bm25_retriever"],
            chroma_retriever=components["chroma_retriever"],
        )

        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Financial Q&A API",
    description="RAG-based financial document Q&A system",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexStatus(BaseModel):
    """Index status response"""
    indexed: bool
    document_count: int
    bm25_indexed: bool
    chroma_indexed: bool


class IndexBuildRequest(BaseModel):
    """Request to build index"""
    force_rebuild: bool = False


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Q&A API",
        "version": "1.0.0",
        "endpoints": {
            "index": "/index",
            "status": "/index/status",
            "query": "/query",
            "health": "/health",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/index/status", response_model=IndexStatus)
async def get_index_status():
    """Get index status"""
    try:
        chroma_stats = components["chroma_retriever"].get_collection_stats()

        return IndexStatus(
            indexed=True,
            document_count=chroma_stats.get("count", 0),
            bm25_indexed=components["bm25_retriever"].bm25_index is not None,
            chroma_indexed=chroma_stats.get("count", 0) > 0,
        )
    except Exception as e:
        logger.error(f"Error getting index status: {e}")
        return IndexStatus(
            indexed=False,
            document_count=0,
            bm25_indexed=False,
            chroma_indexed=False,
        )


@app.post("/index/build")
async def build_index(request: IndexBuildRequest):
    """
    Build search indices from source data.

    This will:
    1. Load and preprocess the AAPL 10-K JSON data
    2. Build BM25 index for keyword search
    3. Build ChromaDB index for vector search
    """
    try:
        logger.info("Starting index build...")

        # Load and preprocess data
        processor = FinancialDataProcessor()
        chunks = processor.process_file()

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found. Please ensure aapl_10k.json is in data/raw/"
            )

        # Build BM25 index
        logger.info("Building BM25 index...")
        components["bm25_retriever"].build_index(chunks)

        # Build ChromaDB index
        logger.info("Building ChromaDB index...")
        components["chroma_retriever"].delete_collection()
        components["chroma_retriever"].build_index(chunks)

        logger.info(f"Index build complete: {len(chunks)} chunks indexed")

        return {
            "message": "Index built successfully",
            "chunks_indexed": len(chunks),
            "bm25_indexed": True,
            "chroma_indexed": True,
        }
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data file not found: {e}"
        )
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error building index: {e}"
        )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the financial document database.

    Process:
    1. Translate Chinese query to English (if needed)
    2. Retrieve relevant documents using hybrid search
    3. Generate answer using LLM with retrieved context
    4. Return answer with citations
    """
    try:
        # Translate query for better retrieval
        english_query = components["answer_generator"].translate_query(request.question)

        # Retrieve documents
        retrieved_docs, debug_info = components["hybrid_retriever"].retrieve(
            query=english_query,
            k=request.max_results,
        )

        if not retrieved_docs:
            return QueryResponse(
                answer="抱歉，未找到相关信息。请尝试重新表述您的问题。",
                citations=[],
                retrieval_debug=debug_info,
            )

        # Generate answer
        query_type = debug_info.get("query_type", "factual")
        answer, citations = components["answer_generator"].generate(
            question=request.question,
            retrieved_docs=retrieved_docs,
            query_type=query_type,
            debug_info=debug_info,
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieval_debug=debug_info if request.include_citations else None,
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {e}"
        )


@app.post("/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate a summary report on a specific topic.

    This endpoint creates a comprehensive report by:
    1. Retrieving relevant documents across specified years
    2. Synthesizing information into a structured report
    3. Providing key findings with citations
    """
    try:
        # Build query for retrieval
        filters = {"year": list(range(request.year_start, request.year_end + 1))}

        # Retrieve documents
        retrieved_docs, _ = components["hybrid_retriever"].retrieve(
            query=request.topic,
            k=20,
            filters=filters,
        )

        if not retrieved_docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for the specified topic and years"
            )

        # Generate report-style answer
        answer, citations = components["answer_generator"].generate(
            question=f"Generate a comprehensive report on: {request.topic}",
            retrieved_docs=retrieved_docs,
            query_type="summary",
        )

        # Extract key findings (simplified)
        key_findings = [doc.text[:200] + "..." for doc in retrieved_docs[:3]]

        return ReportResponse(
            topic=request.topic,
            year_range=f"{request.year_start}-{request.year_end}",
            summary=answer,
            key_findings=key_findings,
            citations=citations,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating report: {e}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
