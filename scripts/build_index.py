#!/usr/bin/env python3
"""
Script to build search indices from source data.
运行此脚本来构建搜索索引
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingest.preprocessor import FinancialDataProcessor
from app.retrieve.bm25_retriever import BM25Retriever
from app.retrieve.chroma_retriever import ChromaRetriever
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to build indices"""
    logger.info("=" * 60)
    logger.info("Building search indices for AAPL 10-K reports")
    logger.info("=" * 60)

    # Step 1: Load and preprocess data
    logger.info("\n[Step 1/3] Loading and preprocessing data...")
    processor = FinancialDataProcessor()

    try:
        chunks = processor.process_file()
        logger.info(f"Processed {len(chunks)} chunks from source data")
    except FileNotFoundError:
        logger.error(f"Data file not found: {settings.data_raw_path}")
        logger.error("Please ensure aapl_10k.json is in the data/raw/ directory")
        return 1
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return 1

    # Step 2: Build BM25 index
    logger.info("\n[Step 2/3] Building BM25 index...")
    bm25_retriever = BM25Retriever()

    try:
        bm25_retriever.build_index(chunks)
        logger.info(f"BM25 index built with {len(chunks)} documents")
    except Exception as e:
        logger.error(f"Error building BM25 index: {e}")
        return 1

    # Step 3: Build ChromaDB index
    logger.info("\n[Step 3/3] Building ChromaDB vector index...")
    chroma_retriever = ChromaRetriever()

    try:
        # Clear existing collection if any
        chroma_retriever.delete_collection()
        # Build new index
        chroma_retriever.build_index(chunks)
        stats = chroma_retriever.get_collection_stats()
        logger.info(f"ChromaDB index built with {stats['count']} documents")
    except Exception as e:
        logger.error(f"Error building ChromaDB index: {e}")
        return 1

    logger.info("\n" + "=" * 60)
    logger.info("Index building completed successfully!")
    logger.info("=" * 60)
    logger.info("\nYou can now run the API server:")
    logger.info("  python -m app.api.main")
    logger.info("\nOr the Streamlit UI:")
    logger.info("  streamlit run app/ui/streamlit_app.py")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
