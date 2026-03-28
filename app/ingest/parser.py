"""
Parse aapl_10k.json and convert to standardized document format
"""
import json
from pathlib import Path
from typing import List
from loguru import logger

from app.schemas.document import Document


# Item type normalization mapping
ITEM_TYPE_MAP = {
    "Item 1": "business",
    "Item 1A": "risk_factors",
    "Item 1B": "unresolved_comments",
    "Item 1C": "cybersecurity",
    "Item 2": "properties",
    "Item 3": "legal",
    "Item 4": "mine_safety",
    "Item 5": "market",
    "Item 6": "reserved",
    "Item 7": "md&a",
    "Item 7A": "market_risk",
    "Item 8": "financial_statements",
    "Item 9": "accounting",
    "Item 9A": "controls",
    "Item 9B": "other_info",
    "Item 9C": "foreign_jurisdictions",
    "Item 10": "corporate_governance",
    "Item 11": "executive_compensation",
    "Item 12": "ownership",
    "Item 13": "relationships",
    "Item 14": "accountant_fees",
    "Item 15": "exhibits",
    "Item 16": "summary",
}


def parse_item_type(section_title: str) -> str:
    """
    Extract and normalize item type from section title

    Args:
        section_title: Raw section title like "Item 1A. Risk Factors"

    Returns:
        Normalized item type like "risk_factors"
    """
    # Try to match Item X, Item XA, Item XX patterns
    for item_pattern, item_type in ITEM_TYPE_MAP.items():
        if item_pattern in section_title:
            return item_type

    # Fallback: special handling for financial statements
    if "Balance Sheet" in section_title:
        return "financial_statements"
    if "Income Statement" in section_title:
        return "financial_statements"
    if "Cash Flow" in section_title:
        return "financial_statements"

    # Default
    return "other"


class TenKParser:
    """Parser for 10-K JSON data"""

    def __init__(self, json_path: str):
        """
        Initialize parser

        Args:
            json_path: Path to aapl_10k.json
        """
        self.json_path = Path(json_path)
        self.documents: List[Document] = []

    def load(self) -> List[dict]:
        """
        Load raw JSON data

        Returns:
            List of raw records
        """
        logger.info(f"Loading data from {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # The JSON has a SQL query as key, get the actual data
        key = list(data.keys())[0]
        records = data[key]

        logger.info(f"Loaded {len(records)} records")
        return records

    def parse(self, records: List[dict]) -> List[Document]:
        """
        Parse raw records into standardized Document objects

        Args:
            records: List of raw records from JSON

        Returns:
            List of Document objects
        """
        logger.info("Parsing records into documents")

        documents = []
        for record in records:
            # Extract basic fields
            year = record.get("file_fiscal_year")
            section_id = record.get("section_id")
            section_title = record.get("section_title", "")

            # Create doc_id: {year}_{section_id}
            doc_id = f"{year}_{section_id}"

            # Parse item type
            item_type = parse_item_type(section_title)

            # Create document
            doc = Document(
                doc_id=doc_id,
                symbol=record.get("symbol", "AAPL"),
                year=year,
                form_type=record.get("form_type", "10-K"),
                section_id=section_id,
                section_title=section_title,
                item_type=item_type,
                text=record.get("section_text", ""),
                metadata={
                    "fiscal_year": year,
                    "form_type": record.get("form_type", "10-K"),
                }
            )

            documents.append(doc)

        logger.info(f"Parsed {len(documents)} documents")
        return documents

    def save_processed(self, documents: List[Document], output_path: str):
        """
        Save processed documents to JSON

        Args:
            documents: List of Document objects
            output_path: Output file path
        """
        logger.info(f"Saving processed documents to {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        data = [doc.model_dump() for doc in documents]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(documents)} documents")

    def run(self, output_path: str = "data/processed/documents.json") -> List[Document]:
        """
        Run the full parsing pipeline

        Args:
            output_path: Output file path

        Returns:
            List of Document objects
        """
        # Load
        records = self.load()

        # Parse
        documents = self.parse(records)
        self.documents = documents

        # Save
        self.save_processed(documents, output_path)

        return documents


def main():
    """Test the parser"""
    parser = TenKParser("data/raw/aapl_10k.json")
    documents = parser.run()

    # Print some stats
    logger.info(f"Total documents: {len(documents)}")

    # Group by year
    years = {}
    for doc in documents:
        year = doc.year
        if year not in years:
            years[year] = []
        years[year].append(doc)

    logger.info(f"Years: {sorted(years.keys())}")
    for year in sorted(years.keys()):
        logger.info(f"  {year}: {len(years[year])} documents")

    # Group by item type
    item_types = {}
    for doc in documents:
        item_type = doc.item_type
        if item_type not in item_types:
            item_types[item_type] = []
        item_types[item_type].append(doc)

    logger.info(f"Item types: {len(item_types)}")
    for item_type, count in sorted(item_types.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {item_type}: {len(count)}")


if __name__ == "__main__":
    main()
