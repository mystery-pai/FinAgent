"""
Answer generation using LLM with retrieved context.
基于检索上下文的LLM答案生成
"""
import logging
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

from app.schemas.models import RetrievedDocument, Citation, ConversationTurn
from app.core.config import settings

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Prompt templates for different question types"""

    SYSTEM_TEMPLATE = """You are a senior financial analyst specializing in analyzing 10-K reports. Your task is to answer questions based on the retrieved document fragments.

**IMPORTANT RULES:**
1. ONLY use information from the provided context fragments
2. If the context doesn't contain enough information, explicitly state "Insufficient information"
3. NEVER make up numbers or facts
4. Always cite your sources with year and section
5. For comparative questions, organize your answer by year
6. Keep answers concise but comprehensive

**Answer Format:**
结论：[Your conclusion in Chinese]
依据：
- [Year], [Section Title]: [Brief relevant content]
补充说明：[Additional context if needed]
"""

    FACTUAL_TEMPLATE = """Based on the following 10-K report fragments and recent conversation history, answer the question.

**Question:** {question}

**Recent Conversation:**
{conversation_history}

**Context Fragments:**
{context}

**Requirements:**
- Provide a direct answer in Chinese
- List each source citation
- If information is insufficient, state it clearly

**Answer:**"""

    COMPARATIVE_TEMPLATE = """Based on the following 10-K report fragments and recent conversation history, provide a comparative analysis.

**Question:** {question}

**Recent Conversation:**
{conversation_history}

**Context Fragments:**
{context}

**Requirements:**
- Compare information across different years
- Organize by year chronologically
- Highlight trends and changes
- Provide source citations for each year

**Answer:**"""

    SUMMARY_TEMPLATE = """Based on the following 10-K report fragments and recent conversation history, provide a comprehensive summary.

**Topic:** {question}

**Recent Conversation:**
{conversation_history}

**Context Fragments:**
{context}

**Requirements:**
- Synthesize information from multiple sources
- Cover key aspects comprehensively
- Maintain factual accuracy
- Provide complete citations

**Answer:**"""

    QUERY_REWRITE_TEMPLATE = """Rewrite the current question into a standalone English retrieval query.

**Recent Conversation:**
{conversation_history}

**Current Question:**
{question}

**Requirements:**
- Keep the rewritten query concise
- Resolve pronouns or omitted subjects using the conversation
- Preserve years, metrics and company names
- Return English only

**Standalone Query:**"""


class QueryTranslator:
    """Translate Chinese queries to English for better retrieval"""

    TRANSLATION_PROMPT = """Translate the following Chinese financial question to English.
Keep technical terms accurate (e.g., "risk factors", "revenue", "net income").

Question: {chinese_question}

English translation:"""

    def __init__(self, client: OpenAI = None):
        """Initialize translator"""
        self.client = client or self._get_default_client()

    def _get_default_client(self) -> Optional[OpenAI]:
        """Get default OpenAI client if configured"""
        if settings.llm_provider == "deepseek" and settings.deepseek_api_key:
            return OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
        elif settings.llm_provider == "ollama":
            return OpenAI(
                base_url=settings.ollama_base_url + "/v1",
                api_key="ollama",
            )
        return None

    def translate(self, chinese_query: str) -> str:
        """
        Translate Chinese query to English.

        Args:
            chinese_query: Query in Chinese

        Returns:
            Query in English (or original if translation fails)
        """
        if not re.search(r"[\u4e00-\u9fff]", chinese_query):
            return chinese_query

        if not self.client:
            logger.warning("No LLM client configured, returning original query")
            return chinese_query

        try:
            model = (
                settings.ollama_model
                if settings.llm_provider == "ollama"
                else "deepseek-chat"
            )
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial translator."},
                    {"role": "user", "content": self.TRANSLATION_PROMPT.format(chinese_question=chinese_query)},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            english_query = response.choices[0].message.content.strip()
            logger.info(f"Translated query: {chinese_query} -> {english_query}")
            return english_query
        except Exception as e:
            logger.error(f"Translation failed: {e}, using original query")
            return chinese_query


class AnswerGenerator:
    """Generate answers using LLM with retrieved context"""

    def __init__(self, client: OpenAI = None):
        """
        Initialize answer generator.

        Args:
            client: OpenAI client (optional)
        """
        self.client = client or self._get_client()
        self.query_translator = QueryTranslator(client)
        self.templates = PromptTemplate()

        logger.info("Answer generator initialized")

    def _get_client(self) -> Optional[OpenAI]:
        """Get configured LLM client"""
        if settings.llm_provider == "deepseek" and settings.deepseek_api_key:
            return OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
        elif settings.llm_provider == "ollama":
            # Ollama uses OpenAI-compatible API
            return OpenAI(
                base_url=settings.ollama_base_url + "/v1",
                api_key="ollama",  # Required but not used by Ollama
            )
        logger.warning(f"No LLM client configured for provider: {settings.llm_provider}")
        return None

    def generate(
        self,
        question: str,
        retrieved_docs: List[RetrievedDocument],
        query_type: str = "factual",
        debug_info: Dict[str, Any] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> tuple[str, List[Citation]]:
        """
        Generate answer based on retrieved documents.

        Args:
            question: User question
            retrieved_docs: Retrieved relevant documents
            query_type: Type of query (factual, comparative, summary)
            debug_info: Debug information from retrieval

        Returns:
            Tuple of (answer, citations)
        """
        if not retrieved_docs:
            return "抱歉，未找到相关信息。无法回答该问题。", []

        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        history_text = self._build_conversation_history(conversation_history)

        # Select template based on query type
        if query_type == "comparative":
            prompt = self.templates.COMPARATIVE_TEMPLATE
        elif query_type == "summary":
            prompt = self.templates.SUMMARY_TEMPLATE
        else:
            prompt = self.templates.FACTUAL_TEMPLATE

        # Build full prompt
        full_prompt = prompt.format(
            question=question,
            context=context,
            conversation_history=history_text,
        )

        # Generate answer
        if not self.client:
            # Fallback: simple concatenation without LLM
            answer = self._generate_simple_answer(question, retrieved_docs)
        else:
            answer = self._generate_with_llm(full_prompt)

        # Extract citations
        citations = self._extract_citations(retrieved_docs)

        return answer, citations

    def build_retrieval_query(
        self,
        question: str,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> str:
        """
        Build retrieval query with optional conversation-aware rewriting.

        Args:
            question: Current user question
            conversation_history: Recent conversation turns

        Returns:
            Retrieval-friendly English query
        """
        translated_question = self.translate_query(question)
        history = conversation_history or []

        if not history or not self._looks_like_follow_up(question):
            return translated_question

        history_text = self._build_conversation_history(history[-3:])
        if not self.client:
            last_question = history[-1].question
            return f"{last_question} {translated_question}".strip()

        try:
            model = (
                settings.ollama_model
                if settings.llm_provider == "ollama"
                else "deepseek-chat"
            )
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You rewrite follow-up questions for retrieval."},
                    {
                        "role": "user",
                        "content": self.templates.QUERY_REWRITE_TEMPLATE.format(
                            conversation_history=history_text,
                            question=question,
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=200,
            )
            rewritten_query = response.choices[0].message.content.strip()
            return rewritten_query or translated_question
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}, falling back to translated question")
            last_question = history[-1].question
            return f"{last_question} {translated_question}".strip()

    def _build_context(self, docs: List[RetrievedDocument]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []

        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            source = f"[{i}] Year: {metadata.get('year', 'N/A')}, Section: {metadata.get('section_title', 'N/A')}"

            # Truncate very long documents
            text = doc.text
            if len(text) > 1000:
                text = text[:1000] + "..."

            context_parts.append(f"{source}\n{text}")

        return "\n\n".join(context_parts)

    def _build_conversation_history(
        self,
        turns: Optional[List[ConversationTurn]],
    ) -> str:
        """Build a compact conversation history string."""
        if not turns:
            return "None"

        history_parts = []
        for index, turn in enumerate(turns[-settings.conversation_window_size :], start=1):
            history_parts.append(
                f"[{index}] User: {turn.question}\nAssistant: {turn.answer}"
            )

        return "\n\n".join(history_parts)

    def _looks_like_follow_up(self, question: str) -> bool:
        """Detect if a question depends on previous turns."""
        normalized_question = question.strip().lower()
        if len(normalized_question.split()) <= 6:
            return True

        follow_up_patterns = [
            r"\b(what about|how about|and what|and how|same for|compare that|compare it|those|them|it)\b",
            r"(那|那么|这个|这个呢|那个|那个呢|它|它们|还有呢|对比一下|相比呢|上一年|前一年)",
        ]
        return any(re.search(pattern, normalized_question) for pattern in follow_up_patterns)

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate answer using LLM"""
        try:
            # Use configured model based on provider
            model = (
                settings.ollama_model
                if settings.llm_provider == "ollama"
                else "deepseek-chat"
            )
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.templates.SYSTEM_TEMPLATE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "抱歉，生成答案时出错。请稍后重试。"

    def _generate_simple_answer(
        self,
        question: str,
        docs: List[RetrievedDocument],
    ) -> str:
        """Generate simple answer without LLM"""
        answer_parts = ["结论：\n"]

        # Group documents by year
        by_year = {}
        for doc in docs:
            year = doc.metadata.get("year", "N/A")
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(doc)

        # Build answer by year
        for year in sorted(by_year.keys(), reverse=True):
            year_docs = by_year[year]
            answer_parts.append(f"\n{year}年：")
            for doc in year_docs[:2]:  # Limit to 2 docs per year
                text = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                answer_parts.append(f"- {text}")

        answer_parts.append("\n\n依据：请参考上述引用的文档片段。")

        return "\n".join(answer_parts)

    def _extract_citations(self, docs: List[RetrievedDocument]) -> List[Citation]:
        """Extract citation information from retrieved documents"""
        citations = []

        for doc in docs:
            metadata = doc.metadata
            citation = Citation(
                year=metadata.get("year", 0),
                section_title=metadata.get("section_title", ""),
                chunk_id=f"{metadata.get('doc_id', '')}_{metadata.get('chunk_id', 0)}",
                relevance_score=doc.score,
            )
            citations.append(citation)

        return citations

    def translate_query(self, chinese_query: str) -> str:
        """
        Translate Chinese query to English for retrieval.

        Args:
            chinese_query: Query in Chinese

        Returns:
            Query in English
        """
        return self.query_translator.translate(chinese_query)
