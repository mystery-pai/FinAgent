#!/usr/bin/env python3
"""
Build search indexes from 10-K data
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from app.ingest.parser import TenKParser
from app.ingest.chunker import SectionAwareChunker
from app.retrieve.chroma_retriever import ChromaRetriever
from app.retrieve.bm25_retriever import BM25Retriever


def main():
    """Build all indexes"""
    logger.info("🏗️  Building search indexes...")

    # Step 1: Parse documents
    logger.info("\n[1/4] Parsing documents...")
    parser = TenKParser("data/raw/aapl_10k.json")
    documents = parser.run()
    logger.info(f"✅ Parsed {len(documents)} documents")

    # Step 2: Chunk documents
    logger.info("\n[2/4] Chunking documents...")
    chunker = SectionAwareChunker(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = chunker.chunk_documents(documents)
    chunker.save_chunks(chunks, "data/processed/chunks.json")
    logger.info(f"✅ Created {len(chunks)} chunks")

    # Step 3: Build BM25 index
    logger.info("\n[3/4] Building BM25 index...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(chunks)
    bm25_retriever.save("indexes/bm25_index")
    logger.info(f"✅ BM25 index built")

    # Step 4: Build Chroma index
    logger.info("\n[4/4] Building Chroma index...")
    chroma_retriever = ChromaRetriever(
        collection_name="aapl_10k",
        persist_directory="indexes/chroma"
    )
    chroma_retriever.build_index(chunks)
    logger.info(f"✅ Chroma index built")

    logger.info("\n" + "="*60)
    logger.info("✅ All indexes built successfully!")
    logger.info("="*60)
    logger.info(f"\nIndex locations:")
    logger.info(f"  - BM25: indexes/bm25_index")
    logger.info(f"  - Chroma: indexes/chroma")
    logger.info(f"  - Chunks: data/processed/chunks.json")


if __name__ == "__main__":
    main()
