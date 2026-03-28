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
        self.alias_patterns = [
            (re.compile(r"\bappl\b", re.IGNORECASE), "apple"),
            (re.compile(r"\baapl\b", re.IGNORECASE), "apple"),
            (re.compile(r"\bapple\s+inc\.?\b", re.IGNORECASE), "apple"),
        ]

        # Year patterns
        self.year_pattern = re.compile(r"\b(20\d{2})\b")
        self.recent_years_pattern = re.compile(r"(?:近|过去|最近|last|past)\s*(\d+)\s*(?:年|years?)")
        self.cash_flow_pattern = re.compile(
            r"现金流|cash\s*flow|operating\s*cash|free\s*cash\s*flow|statement\s*of\s*cash",
            re.IGNORECASE,
        )

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
                r"财务", r"报表", r"收入", r"利润", r"资产", r"负债", r"现金流",
                r"financial", r"statement", r"revenue", r"income",
                r"cash\s*flow", r"balance\s*sheet", r"liquidity",
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

    def normalize(self, query: str) -> str:
        """
        Normalize common aliases and typos before retrieval.

        Args:
            query: Raw user query

        Returns:
            Normalized query
        """
        normalized = query.strip()

        for pattern, replacement in self.alias_patterns:
            normalized = pattern.sub(replacement, normalized)

        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

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
            "normalized_query": self.normalize(query),
            "query_hints": [],
        }

        normalized_query = result["normalized_query"]

        # Extract years
        years = self.year_pattern.findall(normalized_query)
        result["years"] = [int(y) for y in years]

        # Extract recent years pattern
        recent_match = self.recent_years_pattern.search(normalized_query)
        if recent_match:
            count = int(recent_match.group(1))
            # Assume we want the most recent count years
            result["recent_years_count"] = count

        # Detect item types
        query_lower = normalized_query.lower()
        for item_type, patterns in self.item_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    if item_type not in result["item_types"]:
                        result["item_types"].append(item_type)
                    break

        if self.cash_flow_pattern.search(normalized_query):
            result["query_hints"].append("cash_flow")
            if "financial_statements" not in result["item_types"]:
                result["item_types"].append("financial_statements")

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
        retrieval_query = parsed_query["normalized_query"]

        # Build filters from parsed query
        search_filters = dict(filters or {})

        # Add year filters
        if parsed_query["years"] and "year" not in search_filters:
            search_filters["year"] = parsed_query["years"][0]

        # Retrieve from both sources
        logger.info(f"Retrieving with filters: {search_filters}")

        bm25_results = self.bm25_retriever.retrieve(
            query=retrieval_query,
            k=settings.bm25_k,
            filters=search_filters,
        )

        chroma_results = self.chroma_retriever.retrieve(
            query=retrieval_query,
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

        expanded_results = self._expand_adjacent_chunks(
            results=boosted_results,
            parsed_query=parsed_query,
        )

        # Prepare debug info
        debug_info = {
            "original_query": query,
            "normalized_query": retrieval_query,
            "query_type": parsed_query["query_type"],
            "extracted_years": parsed_query["years"],
            "extracted_item_types": parsed_query["item_types"],
            "query_hints": parsed_query["query_hints"],
            "bm25_count": len(bm25_results),
            "chroma_count": len(chroma_results),
            "final_count": len(expanded_results),
            "filters_applied": search_filters,
        }

        logger.info(f"Hybrid retrieval complete: {debug_info}")

        return expanded_results[:k], debug_info

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
        # Score accumulation by chunk key
        scores = {}
        docs_by_key = {}

        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            chunk_key = self._get_chunk_key(doc)
            # RRF score: k / (k + rank)
            rrf_score = k / (k + rank + 1)
            scores[chunk_key] = scores.get(chunk_key, 0) + rrf_score
            docs_by_key[chunk_key] = self._select_better_doc(
                existing_doc=docs_by_key.get(chunk_key),
                candidate_doc=doc,
            )

        # Process Chroma results
        for rank, doc in enumerate(chroma_results):
            chunk_key = self._get_chunk_key(doc)
            rrf_score = k / (k + rank + 1)
            scores[chunk_key] = scores.get(chunk_key, 0) + rrf_score
            docs_by_key[chunk_key] = self._select_better_doc(
                existing_doc=docs_by_key.get(chunk_key),
                candidate_doc=doc,
            )

        # Sort by score
        sorted_chunk_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build result list
        fused = []

        for chunk_key in sorted_chunk_keys:
            doc = docs_by_key.get(chunk_key)
            if not doc:
                continue

            updated_doc = RetrievedDocument(
                doc_id=doc.doc_id,
                text=doc.text,
                score=scores[chunk_key],
                metadata=doc.metadata,
                retrieval_method="hybrid",
            )
            fused.append(updated_doc)

        return fused

    def _get_chunk_key(self, doc: RetrievedDocument) -> str:
        """
        Build a stable key for chunk-level fusion.

        Args:
            doc: Retrieved document

        Returns:
            Chunk-level identifier
        """
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id is None:
            return doc.doc_id
        return f"{doc.doc_id}_{chunk_id}"

    def _select_better_doc(
        self,
        existing_doc: Optional[RetrievedDocument],
        candidate_doc: RetrievedDocument,
    ) -> RetrievedDocument:
        """
        Choose the better source document for a fused chunk.

        Args:
            existing_doc: Previously stored document
            candidate_doc: New candidate document

        Returns:
            Preferred document
        """
        if existing_doc is None:
            return candidate_doc

        if candidate_doc.score > existing_doc.score:
            return candidate_doc

        return existing_doc

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
            section_title = metadata.get("section_title", "").lower()
            item_type = metadata.get("item_type")

            # Boost for exact year match
            if parsed_query["years"]:
                if metadata.get("year") in parsed_query["years"]:
                    boost *= 1.2

            # Boost for item type match
            if parsed_query["item_types"]:
                if item_type in parsed_query["item_types"]:
                    boost *= 1.35

            # Boost for tables in financial queries
            if parsed_query["query_type"] == "factual":
                if metadata.get("is_table"):
                    boost *= 1.1

            if "cash_flow" in parsed_query["query_hints"]:
                if "cash flow" in section_title:
                    boost *= 2.0
                elif item_type == "financial_statements":
                    boost *= 1.4

            if "reserved" in section_title:
                boost *= 0.2

            if "selected financial data / reserved" in section_title:
                boost *= 0.1

            # Apply boost
            doc.score *= boost

        # Re-sort by boosted score
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _expand_adjacent_chunks(
        self,
        results: List[RetrievedDocument],
        parsed_query: Dict[str, Any],
    ) -> List[RetrievedDocument]:
        """
        Expand neighboring chunks for fragmented financial tables.

        Args:
            results: Ranked retrieval results
            parsed_query: Parsed query information

        Returns:
            Results with relevant neighbors injected
        """
        if "cash_flow" not in parsed_query["query_hints"]:
            return results

        expanded_results = []
        seen_chunk_keys = set()
        expanded_doc_ids = set()

        for doc in results:
            chunk_key = self._get_chunk_key(doc)
            if chunk_key not in seen_chunk_keys:
                expanded_results.append(doc)
                seen_chunk_keys.add(chunk_key)

            section_title = doc.metadata.get("section_title", "").lower()
            if "cash flow statement" not in section_title:
                continue

            if doc.doc_id in expanded_doc_ids:
                continue
            expanded_doc_ids.add(doc.doc_id)

            for neighbor in self._get_adjacent_chunks(doc):
                neighbor_key = self._get_chunk_key(neighbor)
                if neighbor_key in seen_chunk_keys:
                    continue
                expanded_results.append(neighbor)
                seen_chunk_keys.add(neighbor_key)

        return expanded_results

    def _get_adjacent_chunks(self, anchor_doc: RetrievedDocument) -> List[RetrievedDocument]:
        """
        Fetch adjacent chunks from the same document for context expansion.

        Args:
            anchor_doc: Anchor chunk

        Returns:
            Neighboring chunks ordered by proximity
        """
        anchor_chunk_id = anchor_doc.metadata.get("chunk_id")
        if anchor_chunk_id is None:
            return []

        if not self.bm25_retriever.corpus:
            return []

        candidate_chunks = []
        for chunk_data in self.bm25_retriever.corpus:
            metadata = chunk_data.get("metadata", {})
            if metadata.get("doc_id") != anchor_doc.doc_id:
                continue

            chunk_id = metadata.get("chunk_id")
            if chunk_id is None or chunk_id == anchor_chunk_id:
                continue

            candidate_chunks.append((chunk_id, chunk_data))

        candidate_chunks.sort(key=lambda item: abs(item[0] - anchor_chunk_id))

        neighbors = []
        for chunk_id, chunk_data in candidate_chunks:
            distance = abs(chunk_id - anchor_chunk_id)
            decay = max(0.7, 1.0 - distance * 0.08)
            neighbors.append(
                RetrievedDocument(
                    doc_id=chunk_data["doc_id"],
                    text=chunk_data["text"],
                    score=anchor_doc.score * decay,
                    metadata=chunk_data["metadata"],
                    retrieval_method="hybrid",
                )
            )

        return neighbors
