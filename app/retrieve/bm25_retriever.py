"""
BM25 keyword-based retriever for financial documents.
BM25关键词检索器
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re

try:
    import bm25s
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False
    logging.warning("bm25s not available, using fallback implementation")

from app.schemas.models import DocumentChunk, RetrievedDocument
from app.core.config import settings

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 retriever using bm25s library.
    Local keyword-based search for precise term matching.
    """

    def __init__(self, index_path: str = None):
        """
        Initialize BM25 retriever.

        Args:
            index_path: Path to load/save index
        """
        self.index_path = index_path or "./data/bm25_index"
        self.corpus = []  # List of chunk dictionaries
        self.bm25_index = None
        self.doc_ids = []  # Mapping from index to doc_id

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for English financial text.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Split on whitespace and punctuation
        tokens = re.findall(r"\b[\w-]+\b", text)

        return tokens

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """
        Build BM25 index from document chunks.

        Args:
            chunks: List of document chunks
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks")

        self.corpus = []
        self.doc_ids = []
        tokenized_corpus = []

        for chunk in chunks:
            # Store chunk data
            self.corpus.append({
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.model_dump(),
            })
            self.doc_ids.append(chunk.doc_id)

            # Tokenize text
            tokens = self._tokenize(chunk.text)
            tokenized_corpus.append(tokens)

        # Build BM25 index
        if BM25S_AVAILABLE:
            self.bm25_index = bm25s.BM25()
            self.bm25_index.index(tokenized_corpus)
        else:
            # Fallback: simple implementation
            self.bm25_index = self._build_simple_bm25(tokenized_corpus)

        # Save index
        self._save_index()

        logger.info("BM25 index built and saved")

    def _build_simple_bm25(self, tokenized_corpus: List[List[str]]) -> Dict[str, Any]:
        """
        Simple BM25 implementation as fallback.

        Args:
            tokenized_corpus: List of tokenized documents

        Returns:
            Dictionary with BM25 data
        """
        # Calculate document frequencies
        df = {}
        for doc_tokens in tokenized_corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1

        # Calculate average document length
        doc_lengths = [len(tokens) for tokens in tokenized_corpus]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        return {
            "tokenized_corpus": tokenized_corpus,
            "df": df,
            "avg_doc_length": avg_doc_length,
            "k1": k1,
            "b": b,
            "num_docs": len(tokenized_corpus),
        }

    def retrieve(
        self,
        query: str,
        k: int = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using BM25.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional metadata filters (e.g., year, item_type)

        Returns:
            List of retrieved documents with scores
        """
        k = k or settings.bm25_k

        if self.bm25_index is None:
            logger.warning("BM25 index not built, loading from disk")
            self._load_index()

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get scores
        if BM25S_AVAILABLE and isinstance(self.bm25_index, bm25s.BM25):
            # Use bm25s library
            results, scores = self.bm25_index.retrieve(query_tokens, k=k * 2)  # Get more for filtering
        else:
            # Use simple implementation
            scores = self._score_simple(query_tokens)
            # Get top indices
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            results = [sorted_indices[:k * 2]]
            scores = [scores[i] for i in sorted_indices[:k * 2]]

        # Convert to RetrievedDocument objects
        retrieved = []
        seen_doc_ids = set()

        for result_set, score_list in zip(results, [scores] if not isinstance(scores, list) else [scores]):
            for idx, score in zip(result_set, score_list):
                if idx >= len(self.corpus):
                    continue

                doc_id = self.doc_ids[idx]
                chunk_data = self.corpus[idx]
                metadata = chunk_data["metadata"]

                # Apply filters if provided
                if filters:
                    if "year" in filters and metadata.get("year") != filters["year"]:
                        continue
                    if "item_type" in filters and metadata.get("item_type") != filters["item_type"]:
                        continue

                # Deduplicate by doc_id
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)

                retrieved.append(
                    RetrievedDocument(
                        doc_id=doc_id,
                        text=chunk_data["text"],
                        score=float(score),
                        metadata=metadata,
                        retrieval_method="bm25",
                    )
                )

                if len(retrieved) >= k:
                    break

        logger.info(f"BM25 retrieved {len(retrieved)} documents for query: {query[:50]}...")
        return retrieved

    def _score_simple(self, query_tokens: List[str]) -> List[float]:
        """
        Score documents using simple BM25 formula.

        Args:
            query_tokens: Tokenized query

        Returns:
            List of scores for each document
        """
        if not isinstance(self.bm25_index, dict):
            return [0.0] * len(self.corpus)

        tokenized_corpus = self.bm25_index["tokenized_corpus"]
        df = self.bm25_index["df"]
        avg_doc_length = self.bm25_index["avg_doc_length"]
        k1 = self.bm25_index["k1"]
        b = self.bm25_index["b"]
        num_docs = self.bm25_index["num_docs"]

        scores = []

        for doc_tokens in tokenized_corpus:
            doc_length = len(doc_tokens)
            score = 0.0

            for token in query_tokens:
                if token not in df:
                    continue

                # Term frequency in document
                tf = doc_tokens.count(token)

                # Document frequency
                df_token = df[token]

                # IDF component
                idf = num_docs / df_token

                # BM25 score for this term
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                score += idf * (numerator / denominator)

            scores.append(score)

        return scores

    def _save_index(self) -> None:
        """Save index to disk"""
        index_path = Path(self.index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        # Save corpus and doc_ids
        with open(index_path / "corpus.pkl", "wb") as f:
            pickle.dump({"corpus": self.corpus, "doc_ids": self.doc_ids}, f)

        # Save BM25 index
        if BM25S_AVAILABLE and isinstance(self.bm25_index, bm25s.BM25):
            self.bm25_index.save(str(index_path / "bm25"))
        else:
            with open(index_path / "bm25_index.pkl", "wb") as f:
                pickle.dump(self.bm25_index, f)

        logger.info(f"BM25 index saved to {index_path}")

    def _load_index(self) -> None:
        """Load index from disk"""
        index_path = Path(self.index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {index_path}")

        # Load corpus and doc_ids
        with open(index_path / "corpus.pkl", "rb") as f:
            data = pickle.load(f)
            self.corpus = data["corpus"]
            self.doc_ids = data["doc_ids"]

        # Load BM25 index
        if BM25S_AVAILABLE:
            try:
                self.bm25_index = bm25s.BM25.load(str(index_path / "bm25"), load_corpus=False)
            except:
                # Fallback to simple implementation
                with open(index_path / "bm25_index.pkl", "rb") as f:
                    self.bm25_index = pickle.load(f)
        else:
            with open(index_path / "bm25_index.pkl", "rb") as f:
                self.bm25_index = pickle.load(f)

        logger.info(f"BM25 index loaded from {index_path}")
