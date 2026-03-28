"""
Hybrid retriever combining BM25 and vector search.
混合检索器 - 结合BM25和向量搜索
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import re

from app.schemas.models import RetrievedDocument
from app.retrieve.bm25_retriever import BM25Retriever
from app.retrieve.chroma_retriever import ChromaRetriever
from app.core.config import settings

logger = logging.getLogger(__name__)


class QueryParser:
    """Parse user queries to extract constraints and type"""

    def __init__(self):
        # Year patterns
        self.year_pattern = re.compile(r"\b(20\d{2})\b")
        self.recent_years_pattern = re.compile(r"(?:近|过去|最近|last|past)\s*(\d+)\s*(?:年|years?)")

        # Item type patterns
        self.item_patterns = {
            "risk_factors": [
                r"风险", r"risk", r"不确定", r"uncertain",
            ],
            "md&a": [
                r"管理层", r"讨论", r"分析", r"经营",
                r"management", r"discussion", r"md&a",
            ],
            "financial_statements": [
                r"财务", r"报表", r"收入", r"利润", r"资产", r"负债",
                r"financial", r"statement", r"revenue", r"income",
            ],
            "business": [
                r"业务", r"产品", r"服务", r"竞争",
                r"business", r"product", r"service", r"competition",
            ],
        }

        # Question type patterns
        self.comparison_patterns = [
            r"对比|比较", r"变化|趋势", r"差异",
            r"compare|comparison", r"change|trend", r"difference|vs|versus",
        ]
        self.summary_patterns = [
            r"总结|概括|概述", r"整体|全面",
            r"summarize|summary", r"overview|overall",
        ]

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse query to extract constraints.

        Args:
            query: User query

        Returns:
            Dictionary with parsed information
        """
        result = {
            "years": [],
            "year_range": None,
            "item_types": [],
            "query_type": "factual",  # factual, comparison, summary
            "original_query": query,
        }

        # Extract years
        years = self.year_pattern.findall(query)
        result["years"] = [int(y) for y in years]

        # Extract recent years pattern
        recent_match = self.recent_years_pattern.search(query)
        if recent_match:
            count = int(recent_match.group(1))
            # Assume we want the most recent count years
            result["recent_years_count"] = count

        # Detect item types
        query_lower = query.lower()
        for item_type, patterns in self.item_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    if item_type not in result["item_types"]:
                        result["item_types"].append(item_type)
                    break

        # Detect question type
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                result["query_type"] = "comparative"
                break

        if result["query_type"] == "factual":
            for pattern in self.summary_patterns:
                if re.search(pattern, query_lower):
                    result["query_type"] = "summary"
                    break

        return result


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search with RRF fusion.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever = None,
        chroma_retriever: ChromaRetriever = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_retriever: BM25 retriever instance
            chroma_retriever: Chroma retriever instance
        """
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.chroma_retriever = chroma_retriever or ChromaRetriever()
        self.query_parser = QueryParser()

        logger.info("Hybrid retriever initialized")

    def retrieve(
        self,
        query: str,
        k: int = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RetrievedDocument], Dict[str, Any]]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Tuple of (retrieved documents, debug info)
        """
        k = k or settings.hybrid_k

        # Parse query
        parsed_query = self.query_parser.parse(query)

        # Build filters from parsed query
        search_filters = filters or {}

        # Add item type filters
        if parsed_query["item_types"] and "item_type" not in search_filters:
            search_filters["item_type"] = parsed_query["item_types"][0]

        # Add year filters
        if parsed_query["years"] and "year" not in search_filters:
            search_filters["year"] = parsed_query["years"][0]

        # Retrieve from both sources
        logger.info(f"Retrieving with filters: {search_filters}")

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            k=settings.bm25_k,
            filters=search_filters,
        )

        chroma_results = self.chroma_retriever.retrieve(
            query=query,
            k=settings.vector_k,
            filters=search_filters,
        )

        # Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(
            bm25_results=bm25_results,
            chroma_results=chroma_results,
            k=k,
        )

        # Apply metadata-based boosting
        boosted_results = self._apply_metadata_boosting(
            results=fused_results,
            parsed_query=parsed_query,
        )

        # Prepare debug info
        debug_info = {
            "query_type": parsed_query["query_type"],
            "extracted_years": parsed_query["years"],
            "extracted_item_types": parsed_query["item_types"],
            "bm25_count": len(bm25_results),
            "chroma_count": len(chroma_results),
            "final_count": len(boosted_results),
            "filters_applied": search_filters,
        }

        logger.info(f"Hybrid retrieval complete: {debug_info}")

        return boosted_results[:k], debug_info

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[RetrievedDocument],
        chroma_results: List[RetrievedDocument],
        k: int = 60,  # RRF constant
    ) -> List[RetrievedDocument]:
        """
        Combine results using Reciprocal Rank Fusion.

        Args:
            bm25_results: Results from BM25
            chroma_results: Results from vector search
            k: RRF constant (higher = more weight to top ranks)

        Returns:
            Fused and sorted results
        """
        # Score accumulation by doc_id
        scores = {}

        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.doc_id
            # RRF score: k / (k + rank)
            rrf_score = k / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Process Chroma results
        for rank, doc in enumerate(chroma_results):
            doc_id = doc.doc_id
            rrf_score = k / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Sort by score
        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build result list
        fused = []
        seen_docs = {}

        for doc_id in sorted_doc_ids:
            # Prefer results from vector search for final text
            doc = next((d for d in chroma_results if d.doc_id == doc_id), None)
            if not doc:
                doc = next((d for d in bm25_results if d.doc_id == doc_id), None)

            if doc and doc_id not in seen_docs:
                # Update score with RRF score
                updated_doc = RetrievedDocument(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=scores[doc_id],
                    metadata=doc.metadata,
                    retrieval_method="hybrid",
                )
                fused.append(updated_doc)
                seen_docs[doc_id] = True

        return fused

    def _apply_metadata_boosting(
        self,
        results: List[RetrievedDocument],
        parsed_query: Dict[str, Any],
    ) -> List[RetrievedDocument]:
        """
        Boost results based on metadata matching.

        Args:
            results: Retrieved documents
            parsed_query: Parsed query information

        Returns:
            Boosted and re-sorted results
        """
        for doc in results:
            boost = 1.0

            metadata = doc.metadata

            # Boost for exact year match
            if parsed_query["years"]:
                if metadata.get("year") in parsed_query["years"]:
                    boost *= 1.2

            # Boost for item type match
            if parsed_query["item_types"]:
                if metadata.get("item_type") in parsed_query["item_types"]:
                    boost *= 1.3

            # Boost for tables in financial queries
            if parsed_query["query_type"] == "factual":
                if metadata.get("is_table"):
                    boost *= 1.1

            # Apply boost
            doc.score *= boost

        # Re-sort by boosted score
        results.sort(key=lambda x: x.score, reverse=True)

        return results
