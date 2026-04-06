"""
Agent base classes and abstractions

This module provides the foundational abstractions for all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Agent configuration model"""
    name: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True


class AgentResult(BaseModel):
    """
    Unified agent result format

    All agents return this standardized result format for consistency
    """
    agent_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class BaseAgent(ABC):
    """
    Abstract base class for all agents

    All agents must implement:
    - execute(): Main execution logic
    - validate_input(): Input validation

    Design principles:
    - Single responsibility: Each agent does one thing well
    - Composability: Agents can be orchestrated together
    - Observability: Results include metadata for debugging
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute agent logic

        Args:
            **kwargs: Agent-specific parameters

        Returns:
            AgentResult with execution results

        Raises:
            ValueError: If input validation fails
        """
        pass

    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails with specific error message
        """
        pass

    def _create_result(
        self,
        success: bool,
        data: Dict[str, Any],
        error: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> AgentResult:
        """
        Helper method to create AgentResult

        Args:
            success: Whether execution succeeded
            data: Result data
            error: Error message if failed
            metadata: Additional metadata

        Returns:
            AgentResult instance
        """
        return AgentResult(
            agent_name=self.name,
            success=success,
            data=data,
            error=error,
            metadata=metadata or {}
        )
