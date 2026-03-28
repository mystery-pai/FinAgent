"""
Data preprocessing and normalization for financial documents.
数据预处理和标准化模块
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import logging

from app.schemas.models import DocumentChunk
from app.core.config import settings

logger = logging.getLogger(__name__)


class ItemTypeMapper:
    """Maps section titles to normalized item types"""

    # Item type mapping patterns
    ITEM_PATTERNS = {
        "reserved": [
            r"item\s*6\.?\s*\[?\s*reserved\s*\]?",
            r"selected\s*financial\s*data\s*/\s*reserved",
        ],
        "business": [
            r"item\s*1\.?\s*business",
            r"item\s*i\.?\s*business",
        ],
        "risk_factors": [
            r"item\s*1a\.?\s*risk\s*factors",
            r"item\s*ia\.?\s*risk\s*factors",
        ],
        "md&a": [
            r"item\s*7\.?\s*management's?\s*discussion",
            r"item\s*vii\.?\s*management's?\s*discussion",
            r"item\s*7\.?\s*md&a",
        ],
        "financial_statements": [
            r"item\s*8\.?\s*financial\s*statements",
            r"item\s*viii\.?\s*financial\s*statements",
        ],
        "legal": [
            r"item\s*3\.?\s*legal",
            r"item\s*iii\.?\s*legal",
        ],
        "executive_compensation": [
            r"item\s*11\.?\s*executive",
            r"item\s*xi\.?\s*executive",
        ],
    }

    @classmethod
    def get_item_type(cls, section_title: str) -> str:
        """
        Map section title to normalized item type.

        Args:
            section_title: Original section title

        Returns:
            Normalized item type (e.g., 'risk_factors', 'md&a')
        """
        if not section_title:
            return "other"

        title_lower = section_title.lower()

        for item_type, patterns in cls.ITEM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, title_lower):
                    return item_type

        if "balance sheet" in title_lower:
            return "financial_statements"
        if "income statement" in title_lower or "statements of operations" in title_lower:
            return "financial_statements"
        if "cash flow" in title_lower:
            return "financial_statements"

        return "other"


class TextCleaner:
    """Text cleaning utilities for financial documents"""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean document text by removing artifacts.

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Normalize line endings first so table structure is preserved.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove control characters but keep newlines and tabs.
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Normalize inline spaces without flattening the whole document.
        text = re.sub(r"[ \t]+", " ", text)

        # Remove page numbers and headers (common patterns).
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
        text = re.sub(r"Page\s+\d+", "", text)

        # Collapse excessive blank lines while preserving row boundaries.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def is_table(text: str) -> bool:
        """
        Detect if text contains tabular data.

        Args:
            text: Text content

        Returns:
            True if text appears to be a table
        """
        # Simple heuristic: lots of numbers and '|' characters
        # or multiple numeric values per line
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) < 2:
            return False

        numeric_lines = 0

        for line in lines:
            # Count numbers in line
            numbers = re.findall(r"[\d,]+\.?\d*", line)
            if len(numbers) >= 3:
                numeric_lines += 1

        return numeric_lines > len(lines) * 0.3


class DocumentChunker:
    """Split documents into chunks for RAG"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens/characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.table_chunk_size = max(self.chunk_size, settings.table_chunk_size)
        self.table_header_lines = max(0, settings.table_header_lines)
        self.table_row_overlap = max(0, settings.table_row_overlap)

    def chunk_section(
        self,
        doc_id: str,
        symbol: str,
        year: int,
        form_type: str,
        section_id: str,
        section_title: str,
        section_text: str,
    ) -> List[DocumentChunk]:
        """
        Split a section into chunks.

        Args:
            doc_id: Document identifier
            symbol: Stock symbol
            year: Fiscal year
            form_type: Form type
            section_id: Section identifier
            section_title: Section title
            section_text: Section text content

        Returns:
            List of document chunks
        """
        # Clean the text
        cleaned_text = TextCleaner.clean_text(section_text)

        # Get item type
        item_type = ItemTypeMapper.get_item_type(section_title)

        # Check if section is a table
        is_table = TextCleaner.is_table(cleaned_text)

        chunks = []

        if is_table:
            # For tables, keep as single chunk if reasonable size
            if len(cleaned_text) <= self.table_chunk_size:
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        symbol=symbol,
                        year=year,
                        form_type=form_type,
                        section_id=section_id,
                        section_title=section_title,
                        item_type=item_type,
                        text=cleaned_text,
                        chunk_id=0,
                        metadata={"is_table": True},
                    )
                )
            else:
                # Split large tables
                table_chunks = self._split_table(
                    doc_id, symbol, year, form_type, section_id, section_title, item_type, cleaned_text
                )
                chunks.extend(table_chunks)
        else:
            # Split text chunks
            text_chunks = self._split_text(
                doc_id, symbol, year, form_type, section_id, section_title, item_type, cleaned_text
            )
            chunks.extend(text_chunks)

        return chunks

    def _split_text(
        self,
        doc_id: str,
        symbol: str,
        year: int,
        form_type: str,
        section_id: str,
        section_title: str,
        item_type: str,
        text: str,
    ) -> List[DocumentChunk]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind(". ", start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 2
                else:
                    # Try paragraph break
                    para_end = text.rfind("\n", start, end)
                    if para_end > start + self.chunk_size // 2:
                        end = para_end + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        symbol=symbol,
                        year=year,
                        form_type=form_type,
                        section_id=section_id,
                        section_title=section_title,
                        item_type=item_type,
                        text=chunk_text,
                        chunk_id=chunk_id,
                        metadata={"is_table": False, "char_count": len(chunk_text)},
                    )
                )
                chunk_id += 1

            start = end - self.chunk_overlap

        return chunks

    def _split_table(
        self,
        doc_id: str,
        symbol: str,
        year: int,
        form_type: str,
        section_id: str,
        section_title: str,
        item_type: str,
        text: str,
    ) -> List[DocumentChunk]:
        """Split table into larger row-based chunks with header reuse."""
        chunks = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not lines:
            return chunks

        header_lines = self._get_table_header_lines(lines)
        body_lines = lines[len(header_lines):]

        if not body_lines:
            return [
                DocumentChunk(
                    doc_id=doc_id,
                    symbol=symbol,
                    year=year,
                    form_type=form_type,
                    section_id=section_id,
                    section_title=section_title,
                    item_type=item_type,
                    text="\n".join(lines),
                    chunk_id=0,
                    metadata={
                        "is_table": True,
                        "row_count": len(lines),
                        "header_lines": len(header_lines),
                        "overlap_lines": 0,
                    },
                )
            ]

        header_size = sum(len(line) + 1 for line in header_lines)
        body_budget = max(self.chunk_size, self.table_chunk_size - header_size)
        chunk_id = 0
        start = 0

        while start < len(body_lines):
            chunk_body = []
            current_size = 0
            end = start

            while end < len(body_lines):
                line = body_lines[end]
                line_size = len(line) + 1
                if chunk_body and current_size + line_size > body_budget:
                    break

                chunk_body.append(line)
                current_size += line_size
                end += 1

            chunk_text = "\n".join(header_lines + chunk_body if header_lines else chunk_body)
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    symbol=symbol,
                    year=year,
                    form_type=form_type,
                    section_id=section_id,
                    section_title=section_title,
                    item_type=item_type,
                    text=chunk_text,
                    chunk_id=chunk_id,
                    metadata={
                        "is_table": True,
                        "row_count": len(chunk_body),
                        "header_lines": len(header_lines),
                        "overlap_lines": self.table_row_overlap if start > 0 else 0,
                    },
                )
            )
            chunk_id += 1

            if end >= len(body_lines):
                break

            next_start = max(start + 1, end - self.table_row_overlap)
            start = next_start

        return chunks

    def _get_table_header_lines(self, lines: List[str]) -> List[str]:
        """Reuse a small leading header block for each table chunk."""
        if self.table_header_lines == 0:
            return []

        header_lines = []
        for line in lines:
            if not line:
                continue
            header_lines.append(line)
            if len(header_lines) >= self.table_header_lines:
                break

        return header_lines


class FinancialDataProcessor:
    """Main processor for financial 10-K data"""

    def __init__(self):
        self.chunker = DocumentChunker()

    def load_raw_data(self, file_path: str = None) -> List[Dict[str, Any]]:
        """
        Load raw JSON data file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of section dictionaries
        """
        file_path = file_path or settings.data_raw_path

        logger.info(f"Loading data from {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle JSON structure where keys are SQL statements
        sections = []
        for key, value in data.items():
            if isinstance(value, list):
                sections.extend(value)
            elif isinstance(value, dict):
                sections.append(value)

        logger.info(f"Loaded {len(sections)} sections")
        return sections

    def process_sections(
        self, sections: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Process raw sections into document chunks.

        Args:
            sections: List of raw section dictionaries

        Returns:
            List of document chunks
        """
        all_chunks = []

        for section in sections:
            try:
                # Extract fields with fallbacks
                symbol = section.get("symbol", "AAPL")
                year = int(section.get("file_fiscal_year", 0))
                form_type = section.get("form_type", "10-K")
                section_id = str(section.get("section_id", ""))
                section_title = section.get("section_title", "")
                section_text = section.get("section_text", "")

                if not section_text or year == 0:
                    continue

                # Create document ID
                doc_id = f"{year}_{section_id}"

                # Chunk the section
                chunks = self.chunker.chunk_section(
                    doc_id=doc_id,
                    symbol=symbol,
                    year=year,
                    form_type=form_type,
                    section_id=section_id,
                    section_title=section_title,
                    section_text=section_text,
                )

                all_chunks.extend(chunks)

            except Exception as e:
                logger.warning(f"Error processing section: {e}")
                continue

        logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks

    def process_file(self, input_path: str = None, output_path: str = None) -> List[DocumentChunk]:
        """
        Process a JSON file and save chunks.

        Args:
            input_path: Input JSON file path
            output_path: Output path for processed chunks

        Returns:
            List of document chunks
        """
        # Load raw data
        sections = self.load_raw_data(input_path)

        # Process into chunks
        chunks = self.process_sections(sections)

        # Save processed chunks
        output_path = output_path or settings.data_processed_path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        chunks_file = output_path / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")

        return chunks
