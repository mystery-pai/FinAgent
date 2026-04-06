"""
Agent Orchestrator for coordinating multiple agents

Manages the execution flow and routing between different agents
"""

from typing import Optional, Any
import logging

from app.agents.base import AgentResult
from app.agents.router import RouterAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple agents to handle user queries

    For Sprint 1:
    - Supports direct visualization agent calls
    - Router agent for intent detection (preparatory)

    For Sprint 2+:
    - Full routing logic (QA vs Visualization vs Report)
    - Multi-agent collaboration
    - State management
    """

    def __init__(
        self,
        router: RouterAgent,
        visualization_agent: Optional[Any] = None,
        qa_agent: Optional[Any] = None,
        report_agent: Optional[Any] = None
    ):
        """
        Initialize orchestrator with agents

        Args:
            router: Router agent for intent classification
            visualization_agent: Visualization agent (optional for Sprint 1)
            qa_agent: QA agent (to be added in Sprint 2)
            report_agent: Report agent (future)
        """
        self.router = router
        self.visualization_agent = visualization_agent
        self.qa_agent = qa_agent
        self.report_agent = report_agent

        logger.info(
            f"Orchestrator initialized with: "
            f"router={router is not None}, "
            f"viz={visualization_agent is not None}, "
            f"qa={qa_agent is not None}, "
            f"report={report_agent is not None}"
        )

    async def execute(
        self,
        question: str,
        session_id: Optional[str] = None,
        auto_route: bool = True,
        **kwargs
    ) -> AgentResult:
        """
        Execute query with automatic routing

        Args:
            question: User question
            session_id: Conversation session ID
            auto_route: Whether to auto-detect intent and route
            **kwargs: Additional parameters

        Returns:
            AgentResult from appropriate agent
        """
        try:
            if not auto_route:
                # No routing, assume QA
                if self.qa_agent:
                    return await self.qa_agent.execute(
                        question=question,
                        session_id=session_id,
                        **kwargs
                    )
                else:
                    return AgentResult(
                        agent_name="orchestrator",
                        success=False,
                        data={},
                        error="QA agent not available"
                    )

            # Auto-route based on intent
            intent_result = await self.router.execute(question=question)

            if not intent_result.success:
                logger.error(f"Intent detection failed: {intent_result.error}")
                # Fallback to QA
                intent = "qa"
            else:
                intent = intent_result.data["intent"]

            logger.info(f"Routing to: {intent}")

            # Route to appropriate agent
            if intent == "visualization":
                if self.visualization_agent:
                    return await self.visualization_agent.execute(
                        question=question,
                        session_id=session_id,
                        **kwargs
                    )
                else:
                    return AgentResult(
                        agent_name="orchestrator",
                        success=False,
                        data={},
                        error="Visualization agent not available"
                    )

            elif intent == "report":
                if self.report_agent:
                    return await self.report_agent.execute(
                        question=question,
                        session_id=session_id,
                        **kwargs
                    )
                else:
                    return AgentResult(
                        agent_name="orchestrator",
                        success=False,
                        data={},
                        error="Report agent not available"
                    )

            else:  # qa
                if self.qa_agent:
                    return await self.qa_agent.execute(
                        question=question,
                        session_id=session_id,
                        **kwargs
                    )
                else:
                    return AgentResult(
                        agent_name="orchestrator",
                        success=False,
                        data={},
                        error="QA agent not available"
                    )

        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}", exc_info=True)
            return AgentResult(
                agent_name="orchestrator",
                success=False,
                data={},
                error=str(e)
            )

    async def execute_visualization(
        self,
        question: str,
        chart_type: str = "auto",
        session_id: Optional[str] = None,
        max_results: int = 10,
        **kwargs
    ) -> AgentResult:
        """
        Execute visualization agent directly (Sprint 1 implementation)

        This is a convenience method for direct visualization calls
        without going through intent detection

        Args:
            question: User question
            chart_type: Chart type or "auto"
            session_id: Session ID
            max_results: Max documents to retrieve
            **kwargs: Additional parameters

        Returns:
            AgentResult from visualization agent
        """
        if not self.visualization_agent:
            return AgentResult(
                agent_name="orchestrator",
                success=False,
                data={},
                error="Visualization agent not initialized"
            )

        logger.info(f"Executing visualization for: {question}")

        try:
            result = await self.visualization_agent.execute(
                question=question,
                chart_type=chart_type,
                session_id=session_id,
                max_results=max_results,
                **kwargs
            )
            return result

        except Exception as e:
            logger.error(f"Visualization execution failed: {str(e)}", exc_info=True)
            return AgentResult(
                agent_name="orchestrator",
                success=False,
                data={},
                error=str(e)
            )
