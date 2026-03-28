"""
Conversation state management for multi-turn Q&A.
多轮对话状态管理
"""
from collections import deque
from threading import Lock
from typing import Deque, Dict, List, Optional
from uuid import uuid4

from app.core.config import settings
from app.schemas.models import ConversationTurn


class ConversationManager:
    """Store recent conversation turns per session in memory."""

    def __init__(self, window_size: int = None):
        self.window_size = window_size or settings.conversation_window_size
        self._sessions: Dict[str, Deque[ConversationTurn]] = {}
        self._lock = Lock()

    def create_session(self) -> str:
        """Create a new conversation session identifier."""
        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = deque(maxlen=self.window_size)
        return session_id

    def ensure_session(self, session_id: Optional[str]) -> str:
        """Ensure a session exists and return its identifier."""
        if not session_id:
            return self.create_session()

        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = deque(maxlen=self.window_size)
        return session_id

    def get_history(self, session_id: str) -> List[ConversationTurn]:
        """Return recent turns for the given session."""
        with self._lock:
            turns = self._sessions.get(session_id)
            if not turns:
                return []
            return list(turns)

    def append_turn(self, session_id: str, question: str, answer: str) -> List[ConversationTurn]:
        """Append a new turn and return the updated window."""
        normalized_session_id = self.ensure_session(session_id)
        turn = ConversationTurn(question=question, answer=answer)

        with self._lock:
            self._sessions[normalized_session_id].append(turn)
            return list(self._sessions[normalized_session_id])
