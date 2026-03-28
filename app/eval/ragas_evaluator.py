"""
Evaluation utilities for retrieval and generation quality.
检索与生成评估工具
"""
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from app.core.config import settings
from app.generate.answer_generator import AnswerGenerator, PromptTemplate, QueryTranslator
from app.retrieve.bm25_retriever import BM25Retriever
from app.retrieve.chroma_retriever import ChromaRetriever
from app.retrieve.hybrid_retriever import HybridRetriever, QueryParser
from app.schemas.models import Citation, RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Single evaluation sample."""

    question: str
    filters: Dict[str, Any] = field(default_factory=dict)
    expected_doc_ids: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None


class RagasEvaluator:
    """Evaluate retrieval and answer quality for the local RAG pipeline."""

    def __init__(
        self,
        mode: str = "hybrid",
        top_k: int = 5,
        metrics: Optional[Sequence[str]] = None,
    ):
        self.mode = mode
        self.top_k = top_k
        self.metrics = list(metrics or ["hit_rate", "mrr", "faithfulness"])
        self._prepare_client_env()
        self.generator = self._create_generator(use_llm="faithfulness" in self.metrics)
        self.retriever = self._load_retriever(mode)
        self.query_parser = QueryParser()

    def evaluate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Run evaluation for all samples in the dataset."""
        samples = self._load_samples(dataset_path)
        sample_results = []
        faithfulness_rows = []

        for sample in samples:
            sample_result = self._evaluate_sample(sample)
            sample_results.append(sample_result)

            if "faithfulness" in self.metrics:
                faithfulness_rows.append(
                    {
                        "user_input": sample.question,
                        "response": self._truncate_text(sample_result["answer"], max_chars=600),
                        "retrieved_contexts": [
                            self._truncate_text(context, max_chars=1200)
                            for context in sample_result["retrieved_contexts"]
                        ],
                    }
                )

        summary = {}
        if "hit_rate" in self.metrics:
            summary["hit_rate"] = self._mean_metric(sample_results, "hit_rate")
        if "mrr" in self.metrics:
            summary["mrr"] = self._mean_metric(sample_results, "mrr")
        if "faithfulness" in self.metrics:
            faithfulness_scores = self._evaluate_faithfulness(faithfulness_rows)
            for result, score in zip(sample_results, faithfulness_scores):
                result["faithfulness"] = score
            summary["faithfulness"] = self._safe_average(faithfulness_scores)

        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "metrics": self.metrics,
            "sample_count": len(sample_results),
            "summary": summary,
            "samples": sample_results,
        }

    def _evaluate_sample(self, sample: EvaluationSample) -> Dict[str, Any]:
        """Evaluate a single question end to end."""
        retrieval_query = self.generator.build_retrieval_query(sample.question)
        retrieved_docs, debug_info = self._retrieve(
            query=retrieval_query,
            filters=sample.filters,
        )

        query_type = debug_info.get("query_type", "factual")
        answer, citations = self.generator.generate(
            question=sample.question,
            retrieved_docs=retrieved_docs,
            query_type=query_type,
            debug_info=debug_info,
        )

        rank = self._first_relevant_rank(
            retrieved_docs=retrieved_docs,
            expected_doc_ids=sample.expected_doc_ids,
            expected_chunk_ids=sample.expected_chunk_ids,
        )

        return {
            "question": sample.question,
            "filters": sample.filters,
            "retrieval_query": retrieval_query,
            "query_type": query_type,
            "answer": answer,
            "reference_answer": sample.reference_answer,
            "hit_rate": 1.0 if rank else 0.0,
            "mrr": 1.0 / rank if rank else 0.0,
            "first_relevant_rank": rank,
            "expected_doc_ids": sample.expected_doc_ids,
            "expected_chunk_ids": sample.expected_chunk_ids,
            "retrieved_doc_ids": [doc.doc_id for doc in retrieved_docs],
            "retrieved_chunk_ids": [self._get_chunk_id(doc) for doc in retrieved_docs],
            "retrieved_contexts": [doc.text for doc in retrieved_docs],
            "citations": self._citations_to_dicts(citations),
            "retrieval_debug": debug_info,
        }

    def _load_retriever(self, mode: str):
        """Load retriever for the requested evaluation mode."""
        if mode == "bm25":
            return BM25Retriever()
        if mode == "vector":
            return ChromaRetriever()
        return HybridRetriever()

    def _create_generator(self, use_llm: bool) -> AnswerGenerator:
        """Create answer generator with optional remote client initialization."""
        if use_llm:
            return AnswerGenerator()

        translator = QueryTranslator.__new__(QueryTranslator)
        translator.client = None

        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = None
        generator.query_translator = translator
        generator.templates = PromptTemplate()
        return generator

    def _prepare_client_env(self) -> None:
        """Avoid forcing SOCKS proxy initialization when HTTP proxies are already set."""
        if os.environ.get("http_proxy") or os.environ.get("https_proxy"):
            os.environ.pop("ALL_PROXY", None)
            os.environ.pop("all_proxy", None)

    def _retrieve(
        self,
        query: str,
        filters: Dict[str, Any],
    ) -> tuple[List[RetrievedDocument], Dict[str, Any]]:
        """Normalize retriever outputs across modes."""
        if self.mode == "hybrid":
            return self.retriever.retrieve(
                query=query,
                k=self.top_k,
                filters=filters or None,
            )

        parsed_query = self.query_parser.parse(query)
        results = self.retriever.retrieve(
            query=query,
            k=self.top_k,
            filters=filters or None,
        )
        debug_info = {
            "mode": self.mode,
            "original_query": query,
            "normalized_query": parsed_query["normalized_query"],
            "query_type": parsed_query["query_type"],
            "extracted_years": parsed_query["years"],
            "extracted_item_types": parsed_query["item_types"],
            "query_hints": parsed_query["query_hints"],
            "filters_applied": filters,
            "final_count": len(results),
        }
        return results, debug_info

    def _load_samples(self, dataset_path: str) -> List[EvaluationSample]:
        """Load evaluation samples from JSON."""
        path = Path(dataset_path)
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)

        samples = []
        for row in rows:
            samples.append(
                EvaluationSample(
                    question=row["question"],
                    filters=row.get("filters", {}),
                    expected_doc_ids=row.get("expected_doc_ids", []),
                    expected_chunk_ids=row.get("expected_chunk_ids", []),
                    reference_answer=row.get("reference_answer"),
                )
            )
        return samples

    def _first_relevant_rank(
        self,
        retrieved_docs: List[RetrievedDocument],
        expected_doc_ids: Sequence[str],
        expected_chunk_ids: Sequence[str],
    ) -> Optional[int]:
        """Return rank of the first relevant retrieval result."""
        expected_doc_id_set = set(expected_doc_ids)
        expected_chunk_id_set = set(expected_chunk_ids)

        for rank, doc in enumerate(retrieved_docs, start=1):
            chunk_id = self._get_chunk_id(doc)
            if doc.doc_id in expected_doc_id_set or chunk_id in expected_chunk_id_set:
                return rank

        return None

    def _get_chunk_id(self, doc: RetrievedDocument) -> str:
        """Build normalized chunk identifier."""
        return f"{doc.doc_id}_{doc.metadata.get('chunk_id', 0)}"

    def _citations_to_dicts(self, citations: List[Citation]) -> List[Dict[str, Any]]:
        """Convert citation models to dictionaries."""
        return [citation.model_dump() for citation in citations]

    def _mean_metric(self, sample_results: List[Dict[str, Any]], key: str) -> float:
        """Compute mean for a numeric metric across samples."""
        values = [float(result[key]) for result in sample_results]
        return self._safe_average(values)

    def _safe_average(self, values: Sequence[Optional[float]]) -> float:
        """Average non-null numeric values."""
        valid_values = []
        for value in values:
            if value is None:
                continue
            numeric_value = float(value)
            if math.isnan(numeric_value):
                continue
            valid_values.append(numeric_value)
        if not valid_values:
            return 0.0
        return sum(valid_values) / len(valid_values)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Trim long text before sending it to evaluation models."""
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _evaluate_faithfulness(self, rows: List[Dict[str, Any]]) -> List[Optional[float]]:
        """Evaluate faithfulness with RAGAS."""
        if not rows:
            return []

        if settings.llm_provider != "deepseek" or not settings.deepseek_api_key:
            raise RuntimeError(
                "Faithfulness evaluation requires DeepSeek credentials in the current implementation."
            )

        try:
            from datasets import Dataset
            from openai import AsyncOpenAI
            from ragas import evaluate
            from ragas.llms import llm_factory
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS faithfulness evaluation requires `ragas` and `datasets`. "
                "Please install project dependencies again."
            ) from exc

        evaluator_llm = llm_factory(
            "deepseek-chat",
            client=AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            ),
        )

        try:
            from ragas.metrics import Faithfulness
            metric = Faithfulness(llm=evaluator_llm)
        except ImportError:
            try:
                from ragas.metrics.collections import Faithfulness as CollectionFaithfulness
            except ImportError as exc:
                raise RuntimeError("Unable to import RAGAS faithfulness metric.") from exc

            scorer = CollectionFaithfulness(llm=evaluator_llm)
            scores = []
            for row in rows:
                result = scorer.score(
                    user_input=row["user_input"],
                    response=row["response"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
                score_value = getattr(result, "value", result)
                scores.append(float(score_value) if score_value is not None else None)
            return scores

        dataset = Dataset.from_list(rows)
        result = evaluate(
            dataset=dataset,
            metrics=[metric],
            llm=evaluator_llm,
        )

        if hasattr(result, "to_pandas"):
            frame = result.to_pandas()
        else:
            frame = pd.DataFrame(getattr(result, "scores", []))

        metric_column = self._resolve_metric_column(frame, "faithfulness")
        return frame[metric_column].tolist()

    def _resolve_metric_column(self, frame: pd.DataFrame, metric_name: str) -> str:
        """Resolve metric column name across RAGAS versions."""
        normalized_target = metric_name.lower()
        for column in frame.columns:
            if column.lower() == normalized_target:
                return column

        reserved_columns = {
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
            "reference_answer",
            "question",
            "answer",
            "contexts",
        }
        metric_columns = [column for column in frame.columns if column not in reserved_columns]
        if not metric_columns:
            raise RuntimeError("Unable to locate faithfulness column in RAGAS result.")
        return metric_columns[0]
