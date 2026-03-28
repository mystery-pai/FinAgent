"""
ChromaDB vector retriever for semantic search.
ChromaDB向量检索器
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from chromadb import Collection, Client, PersistentClient
from chromadb.config import Settings as ChromaSettings
import chromadb.utils.embedding_functions as embedding_functions

from app.schemas.models import DocumentChunk, RetrievedDocument
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChromaRetriever:
    """
    Vector retriever using ChromaDB and sentence-transformers.
    Provides semantic search capabilities.
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: str = None,
    ):
        """
        Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: Name of the embedding model
        """
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.embedding_model_name = embedding_model or settings.embedding_model

        # Prefer local model cache first to avoid network/proxy issues in local debugging.
        self.embedding_function = self._create_embedding_function()

        # Initialize ChromaDB client
        self.client = PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

        logger.info(f"ChromaDB retriever initialized with collection: {self.collection_name}")

    def _create_embedding_function(self):
        """
        Create embedding function with local-cache-first strategy.

        Returns:
            SentenceTransformer embedding function
        """
        try:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name,
                device=settings.embedding_device,
                local_files_only=True,
            )
            logger.info(f"Loaded embedding model from local cache: {self.embedding_model_name}")
            return embedding_function
        except Exception as exc:
            logger.warning(f"Local embedding cache unavailable for {self.embedding_model_name}: {exc}")

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name,
            device=settings.embedding_device,
        )

    def _get_or_create_collection(self) -> Collection:
        """
        Get existing collection or create new one.

        Returns:
            ChromaDB collection
        """
        # Try to get existing collection
        try:
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Financial reports 10-K"},
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """
        Build vector index from document chunks.

        Args:
            chunks: List of document chunks
        """
        logger.info(f"Building ChromaDB index from {len(chunks)} chunks")

        # Prepare data for batch insertion
        ids = []
        texts = []
        metadatas = []

        for chunk in chunks:
            # Create unique ID
            chunk_id = f"{chunk.doc_id}_{chunk.chunk_id}"
            ids.append(chunk_id)

            # Document text
            texts.append(chunk.text)

            # Metadata
            metadata = {
                "doc_id": chunk.doc_id,
                "symbol": chunk.symbol,
                "year": chunk.year,
                "form_type": chunk.form_type,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "item_type": chunk.item_type,
                "chunk_id": chunk.chunk_id,
            }
            # Add additional metadata
            metadata.update(chunk.metadata)
            metadatas.append(metadata)

        # Batch insert
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
            )

            logger.info(f"Added batch {i // batch_size + 1}/{(len(ids) + batch_size - 1) // batch_size}")

        logger.info(f"ChromaDB index built with {self.collection.count()} documents")

    def retrieve(
        self,
        query: str,
        k: int = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using vector similarity search.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional metadata filters (e.g., year, item_type)

        Returns:
            List of retrieved documents with scores
        """
        k = k or settings.vector_k

        # Build where clause for filters
        where_clause = None
        if filters:
            where_clauses = []
            for key, value in filters.items():
                where_clauses.append({key: {"$eq": value}})
            if len(where_clauses) == 1:
                where_clause = where_clauses[0]
            elif len(where_clauses) > 1:
                where_clause = {"$and": where_clauses}

        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_clause,
        )

        # Convert to RetrievedDocument objects
        retrieved = []

        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                text = results["documents"][0][i]
                distance = results["distances"][0][i]

                # Convert distance to similarity score (Chroma uses L2 distance)
                # Higher is better, so we convert distance to similarity
                score = 1.0 / (1.0 + distance)

                retrieved.append(
                    RetrievedDocument(
                        doc_id=metadata.get("doc_id", doc_id),
                        text=text,
                        score=score,
                        metadata=metadata,
                        retrieval_method="vector",
                    )
                )

        logger.info(f"Vector retrieved {len(retrieved)} documents for query: {query[:50]}...")
        return retrieved

    def delete_collection(self) -> None:
        """Delete the current collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            # Recreate collection
            self.collection = self._get_or_create_collection()
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata,
        }
