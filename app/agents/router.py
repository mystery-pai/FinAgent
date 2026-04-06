"""
Router Agent for intent classification

Determines user intent to route queries to appropriate agents
"""

from typing import Optional
from pydantic import BaseModel, Field
import logging

from app.agents.base import BaseAgent, AgentConfig, AgentResult

logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    """Intent classification result"""
    intent: str = Field(
        description="Detected user intent: qa, visualization, or report"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score (0-1)"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Why this intent was chosen"
    )


class RouterAgent(BaseAgent):
    """
    Routes user queries to appropriate agents based on intent

    Strategy:
    1. Rule-based detection (fast, high precision)
    2. LLM-based classification (fallback, high recall) - future enhancement
    3. Hybrid scoring (combine both) - future enhancement

    Current implementation: Rule-based only
    """

    # Visualization keywords (English and Chinese)
    VIZ_KEYWORDS = [
        "show", "plot", "chart", "graph", "visualize", "visualization",
        "trend", "compare", "comparison", "display", "draw", "illustrate",
        "图表", "显示", "展示", "可视化", "趋势", "对比", "绘制"
    ]

    # Report keywords
    REPORT_KEYWORDS = [
        "report", "summary", "summarize", "analysis", "analyze", "overview", "breakdown",
        "报告", "总结", "分析", "概览", "综述"
    ]

    def __init__(self, config: AgentConfig, llm_client=None):
        """
        Initialize router agent

        Args:
            config: Agent configuration
            llm_client: Optional LLM client for advanced classification
        """
        super().__init__(config)
        self.llm_client = llm_client

    async def execute(self, question: str, **kwargs) -> AgentResult:
        """
        Classify user intent

        Args:
            question: User question

        Returns:
            AgentResult with intent classification
        """
        try:
            # Validate
            self.validate_input(question=question)

            # Try rule-based first
            rule_result = self._rule_based_classification(question)

            if rule_result.confidence >= 0.8:
                # High confidence, use rule result
                logger.info(
                    f"Intent detected: {rule_result.intent} "
                    f"(confidence: {rule_result.confidence:.2f})"
                )
                return self._create_result(
                    success=True,
                    data=rule_result.dict(),
                    metadata={"method": "rule_based"}
                )

            # Fallback to LLM if available (future enhancement)
            if self.llm_client:
                llm_result = await self._llm_based_classification(question)
                logger.info(
                    f"Intent detected (LLM): {llm_result.intent} "
                    f"(confidence: {llm_result.confidence:.2f})"
                )
                return self._create_result(
                    success=True,
                    data=llm_result.dict(),
                    metadata={"method": "llm_based"}
                )

            # Default to rule result
            logger.info(
                f"Intent detected (fallback): {rule_result.intent} "
                f"(confidence: {rule_result.confidence:.2f})"
            )
            return self._create_result(
                success=True,
                data=rule_result.dict(),
                metadata={"method": "rule_based_fallback"}
            )

        except Exception as e:
            logger.error(f"Router agent failed: {str(e)}", exc_info=True)
            return self._create_result(
                success=False,
                data={},
                error=str(e)
            )

    def validate_input(self, question: str, **kwargs) -> bool:
        """
        Validate input

        Args:
            question: User question

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        return True

    def _rule_based_classification(self, question: str) -> IntentClassification:
        """
        Rule-based intent detection

        Logic:
        1. Check for visualization keywords
        2. Check for report keywords
        3. Default to QA

        Args:
            question: User question

        Returns:
            IntentClassification result
        """
        question_lower = question.lower()

        # Check visualization keywords
        viz_matches = sum(
            1 for kw in self.VIZ_KEYWORDS
            if kw in question_lower
        )
        viz_score = viz_matches / len(self.VIZ_KEYWORDS)

        # Check report keywords
        report_matches = sum(
            1 for kw in self.REPORT_KEYWORDS
            if kw in question_lower
        )
        report_score = report_matches / len(self.REPORT_KEYWORDS)

        # Decision logic
        if viz_matches > 0:
            # Detected visualization intent
            confidence = min(0.6 + viz_matches * 0.15, 0.95)
            return IntentClassification(
                intent="visualization",
                confidence=confidence,
                reasoning=f"Detected {viz_matches} visualization keyword(s) in query"
            )
        elif report_matches > 0:
            # Detected report intent
            confidence = min(0.6 + report_matches * 0.15, 0.95)
            return IntentClassification(
                intent="report",
                confidence=confidence,
                reasoning=f"Detected {report_matches} report keyword(s) in query"
            )
        else:
            # Default to QA
            return IntentClassification(
                intent="qa",
                confidence=0.7,
                reasoning="No specific intent keywords detected, defaulting to QA"
            )

    async def _llm_based_classification(self, question: str) -> IntentClassification:
        """
        LLM-based intent classification (future enhancement)

        Uses LLM to classify intent with higher accuracy

        Args:
            question: User question

        Returns:
            IntentClassification result
        """
        # TODO: Implement LLM classification using DeepSeek
        # For now, fallback to rule-based
        logger.warning("LLM classification not implemented yet, using rule-based fallback")
        return self._rule_based_classification(question)
