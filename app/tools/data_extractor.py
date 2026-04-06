"""
Data extraction tool using LLM Function Calling
使用 LLM Function Calling 提取结构化数据
"""
import json
from typing import List, Optional
import logging
from openai import OpenAI

from app.schemas.models import RetrievedDocument, ChartDataSchema, ChartSeries
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataExtractor:
    """Extract structured chart data from text using LLM Function Calling"""

    def __init__(self, llm_client: Optional[OpenAI] = None):
        """
        Initialize DataExtractor with LLM client

        Args:
            llm_client: OpenAI-compatible client for DeepSeek API. If None, creates default client
        """
        if llm_client is None:
            # Create OpenAI-compatible client for DeepSeek
            self.client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url
            )
        else:
            self.client = llm_client

        # Define function schema for extraction
        self.function_schema = {
            "name": "extract_chart_data",
            "description": "Extract structured financial data for visualization from documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Chart title that summarizes the data"
                    },
                    "x_label": {
                        "type": "string",
                        "description": "X-axis label (e.g., 'Year', 'Quarter', 'Date')"
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Y-axis label (e.g., 'Revenue', 'Amount', 'Percentage')"
                    },
                    "x_values": {
                        "type": "array",
                        "items": {},
                        "description": "X-axis values (e.g., [2023, 2024, 2025] or ['Q1', 'Q2', 'Q3'])"
                    },
                    "series": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Series name (e.g., 'Revenue', 'Net Income')"
                                },
                                "values": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "Data values matching x_values length"
                                },
                                "unit": {
                                    "type": "string",
                                    "description": "Unit (e.g., 'Million USD', 'Billion USD', '%')"
                                }
                            },
                            "required": ["name", "values"]
                        },
                        "description": "Data series for the chart"
                    },
                    "chart_type_hint": {
                        "type": "string",
                        "enum": ["line", "bar", "pie", "scatter", "grouped_bar"],
                        "description": "Suggested chart type based on data characteristics"
                    },
                    "data_source": {
                        "type": "string",
                        "description": "Data source citation (e.g., '10-K 2024, Item 8')"
                    }
                },
                "required": ["title", "x_label", "y_label", "x_values", "series"]
            }
        }

        logger.info("DataExtractor initialized with DeepSeek API")

    def extract(
        self,
        question: str,
        documents: List[RetrievedDocument],
        max_context_length: int = 4000
    ) -> ChartDataSchema:
        """
        Extract structured chart data from retrieved documents

        Args:
            question: User question requesting visualization
            documents: Retrieved documents containing financial data
            max_context_length: Maximum context length in characters

        Returns:
            ChartDataSchema: Structured chart data

        Raises:
            ValueError: If extraction fails or no valid data found
        """
        if not documents:
            raise ValueError("No documents provided for extraction")

        logger.info(f"Extracting data for question: {question}")
        logger.info(f"Processing {len(documents)} documents")

        # Build context from documents
        context = self._build_context(documents, max_context_length)
        logger.debug(f"Context length: {len(context)} chars")

        # Prepare messages
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": self._get_user_prompt(question, context)}
        ]

        try:
            # Call LLM with function calling
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=[{
                    "type": "function",
                    "function": self.function_schema
                }],
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_chart_data"}
                },
                temperature=0.0  # Use deterministic output for data extraction
            )

            # Extract function call result
            message = response.choices[0].message
            if not message.tool_calls:
                raise ValueError("LLM did not return function call")

            tool_call = message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)

            logger.info(f"Successfully extracted data: {function_args.get('title', 'N/A')}")

            # Convert to ChartDataSchema
            chart_data = ChartDataSchema(
                title=function_args["title"],
                x_label=function_args["x_label"],
                y_label=function_args["y_label"],
                x_values=function_args["x_values"],
                series=[
                    ChartSeries(
                        name=s["name"],
                        values=s["values"],
                        unit=s.get("unit")
                    )
                    for s in function_args["series"]
                ],
                chart_type_hint=function_args.get("chart_type_hint"),
                data_source=function_args.get("data_source")
            )

            # Validate data consistency
            self._validate_chart_data(chart_data)

            return chart_data

        except Exception as e:
            logger.error(f"Failed to extract chart data: {str(e)}")
            raise ValueError(f"Data extraction failed: {str(e)}")

    def _build_context(
        self,
        documents: List[RetrievedDocument],
        max_length: int
    ) -> str:
        """
        Build context from documents, prioritizing financial statements (tables)

        Args:
            documents: Retrieved documents
            max_length: Maximum context length

        Returns:
            Formatted context string
        """
        # Sort documents to prioritize table data
        sorted_docs = sorted(
            documents,
            key=lambda d: (
                # Prioritize documents containing financial statements
                "financial_position" in d.metadata.get("section_title", "").lower() or
                "statements" in d.metadata.get("section_title", "").lower() or
                "consolidated" in d.text.lower(),
                # Then by relevance score
                -d.score
            ),
            reverse=True
        )

        context_parts = []
        current_length = 0

        for doc in sorted_docs:
            formatted_doc = self._format_document(doc)
            doc_length = len(formatted_doc)

            if current_length + doc_length > max_length:
                # Try to add partial content if space allows
                remaining = max_length - current_length
                if remaining > 200:  # Only add if meaningful content can fit
                    context_parts.append(formatted_doc[:remaining] + "...\n")
                break

            context_parts.append(formatted_doc)
            current_length += doc_length

        return "\n".join(context_parts)

    def _format_document(self, doc: RetrievedDocument) -> str:
        """
        Format single document for context

        Args:
            doc: Retrieved document

        Returns:
            Formatted document string
        """
        metadata = doc.metadata
        year = metadata.get("year", "N/A")
        section = metadata.get("section_title", "N/A")
        score = doc.score

        header = f"=== Document (Year: {year}, Section: {section}, Score: {score:.3f}) ==="
        return f"{header}\n{doc.text}\n"

    def _get_system_prompt(self) -> str:
        """
        System prompt for extraction

        Returns:
            System prompt string
        """
        return """You are a financial data extraction expert. Your task is to extract structured numerical data from financial documents for visualization.

Key requirements:
1. Extract ONLY exact numbers from the documents - do NOT make up or estimate values
2. Pay attention to units (Million, Billion, Thousand, etc.) and convert consistently
3. Ensure all data series have the same length as x_values
4. Preserve the original precision of numbers
5. If a value is not found, do NOT include it in the output
6. For financial statements, prioritize tabular data over narrative text
7. Include proper data source citations

Data quality rules:
- All values must be actual numbers from the documents
- Units must be consistent within each series
- X-axis values must be in logical order
- Series names should be clear and descriptive

If you cannot find sufficient data to answer the question, indicate this in the title."""

    def _get_user_prompt(self, question: str, context: str) -> str:
        """
        User prompt for extraction

        Args:
            question: User question
            context: Document context

        Returns:
            User prompt string
        """
        return f"""Based on the following financial documents, extract structured data to answer this question:

Question: {question}

Documents:
{context}

Extract the data using the extract_chart_data function. Ensure:
1. All numbers are extracted exactly as they appear
2. Units are properly identified and consistent
3. The chart type hint matches the data characteristics
4. Data source clearly indicates which documents/sections were used

If the documents don't contain sufficient data, set the title to indicate this (e.g., "Insufficient data for [question]")."""

    def _validate_chart_data(self, chart_data: ChartDataSchema) -> None:
        """
        Validate extracted chart data for consistency

        Args:
            chart_data: Extracted chart data

        Raises:
            ValueError: If data is invalid
        """
        x_len = len(chart_data.x_values)

        if x_len == 0:
            raise ValueError("No x_values extracted")

        if not chart_data.series:
            raise ValueError("No data series extracted")

        for series in chart_data.series:
            if len(series.values) != x_len:
                raise ValueError(
                    f"Series '{series.name}' has {len(series.values)} values "
                    f"but x_values has {x_len} values"
                )

        logger.debug(f"Validated chart data with {len(chart_data.series)} series "
                    f"and {x_len} x-values")
