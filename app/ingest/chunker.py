"""
Chunk documents for retrieval
Uses section-aware chunking strategy
"""
import json
from pathlib import Path
from typing import List
from loguru import logger

from app.schemas.document import Document, Chunk


class SectionAwareChunker:
    """
    Chunk documents with section-aware strategy

    Strategy:
    1. Keep section as primary unit (preserve context)
    2. Only chunk long sections (>1000 tokens)
    3. Preserve table boundaries in Item 8
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_tokens: int = 1000
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            max_tokens: Max section length before chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 1 token ≈ 4 chars)

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _split_text(self, text: str, chunk_order: int) -> List[str]:
        """
        Split text into chunks with overlap

        Args:
            text: Input text
            chunk_order: Starting chunk order

        Returns:
            List of text chunks
        """
        chunks = []
        # Rough character limit (4 chars per token)
        char_limit = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4

        start = 0
        order = chunk_order

        while start < len(text):
            end = start + char_limit

            # Get chunk
            if end >= len(text):
                chunk = text[start:]
            else:
                # Try to break at sentence boundary
                chunk = text[start:end]

                # Look for sentence endings
                for sep in [". ", "! ", "? ", "\n"]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > char_limit // 2:  # At least half chunk
                        chunk = chunk[:last_sep + len(sep)]
                        end = start + len(chunk)
                        break

            chunks.append(chunk)
            start = end - overlap_chars
            order += 1

        return chunks

    def _is_financial_table(self, text: str) -> bool:
        """
        Check if text looks like a financial table

        Args:
            text: Input text

        Returns:
            True if looks like a table
        """
        # Heuristics for table detection
        indicators = [
            "$" in text and "," in text,  # Numbers with commas
            text.count("|") > 5,  # Pipe-separated
            len(text.split("\n")) > 5 and all(
                "|" in line or "$" in line for line in text.split("\n")[:5]
            )
        ]
        return any(indicators)

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Chunk a single document

        Args:
            doc: Document to chunk

        Returns:
            List of chunks
        """
        # Check if chunking is needed
        text = doc.text.strip()
        estimated_tokens = self._estimate_tokens(text)

        # Short section: keep as single chunk
        if estimated_tokens <= self.max_tokens:
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_0",
                doc_id=doc.doc_id,
                year=doc.year,
                section_title=doc.section_title,
                item_type=doc.item_type,
                chunk_order=0,
                text=text,
                metadata={
                    **doc.metadata,
                    "is_single_chunk": True
                }
            )
            return [chunk]

        # Long section: need to chunk
        # But preserve table boundaries for Item 8
        if doc.item_type == "financial_statements":
            # Try to split by table boundaries
            chunks = []
            chunk_order = 0

            # Split by double newlines (table separators)
            sections = text.split("\n\n")
            current_chunk = ""

            for section in sections:
                # If adding this section exceeds limit and current_chunk is not empty
                if (
                    current_chunk and
                    self._estimate_tokens(current_chunk + "\n\n" + section) > self.chunk_size
                ):
                    # Save current chunk
                    chunk = Chunk(
                        chunk_id=f"{doc.doc_id}_{chunk_order}",
                        doc_id=doc.doc_id,
                        year=doc.year,
                        section_title=doc.section_title,
                        item_type=doc.item_type,
                        chunk_order=chunk_order,
                        text=current_chunk.strip(),
                        metadata={**doc.metadata}
                    )
                    chunks.append(chunk)
                    chunk_order += 1
                    current_chunk = section
                else:
                    current_chunk += "\n\n" + section if current_chunk else section

            # Don't forget the last chunk
            if current_chunk:
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_{chunk_order}",
                    doc_id=doc.doc_id,
                    year=doc.year,
                    section_title=doc.section_title,
                    item_type=doc.item_type,
                    chunk_order=chunk_order,
                    text=current_chunk.strip(),
                    metadata={**doc.metadata}
                )
                chunks.append(chunk)

            return chunks

        # General case: split with overlap
        text_chunks = self._split_text(text, 0)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_{i}",
                doc_id=doc.doc_id,
                year=doc.year,
                section_title=doc.section_title,
                item_type=doc.item_type,
                chunk_order=i,
                text=chunk_text,
                metadata={**doc.metadata}
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk multiple documents

        Args:
            documents: List of documents

        Returns:
            List of all chunks
        """
        logger.info(f"Chunking {len(documents)} documents")

        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks")
        return all_chunks

    def save_chunks(self, chunks: List[Chunk], output_path: str):
        """
        Save chunks to JSON

        Args:
            chunks: List of chunks
            output_path: Output file path
        """
        logger.info(f"Saving chunks to {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [chunk.model_dump() for chunk in chunks]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(chunks)} chunks")


def main():
    """Test the chunker"""
    from app.ingest.parser import TenKParser

    # Parse documents
    parser = TenKParser("data/raw/aapl_10k.json")
    documents = parser.run()

    # Chunk documents
    chunker = SectionAwareChunker(
        chunk_size=512,
        chunk_overlap=50,
        max_tokens=1000
    )
    chunks = chunker.chunk_documents(documents)
    chunker.save_chunks(chunks, "data/processed/chunks.json")

    # Print stats
    logger.info(f"Total chunks: {len(chunks)}")

    # Count multi-chunk documents
    multi_chunk = 0
    for doc in documents:
        doc_chunks = [c for c in chunks if c.doc_id == doc.doc_id]
        if len(doc_chunks) > 1:
            multi_chunk += 1

    logger.info(f"Documents with multiple chunks: {multi_chunk}")


if __name__ == "__main__":
    main()
