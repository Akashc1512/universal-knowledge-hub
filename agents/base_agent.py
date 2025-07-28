"""
Base Agent Class - Common interface and functionality for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging
import os
from collections import defaultdict
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the system."""

    ORCHESTRATOR = "orchestrator"
    RETRIEVAL = "retrieval"
    FACT_CHECK = "fact_check"
    SYNTHESIS = "synthesis"
    CITATION = "citation"


class MessageType(Enum):
    """Types of messages for inter-agent communication."""

    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"


class TaskPriority(Enum):
    """Priority levels for tasks."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


@dataclass
class AgentMessage:
    """
    Standard message format for inter-agent communication.

    Attributes:
        header: Message metadata including routing and timing information
        payload: Actual task data, results, or control information
    """

    header: Dict[str, Any] = field(
        default_factory=lambda: {
            "message_id": str(uuid.uuid4()),
            "correlation_id": None,
            "timestamp": time.time(),
            "sender_agent": None,
            "recipient_agent": None,
            "message_type": MessageType.TASK,
            "priority": TaskPriority.MEDIUM.value,
            "ttl": int(os.getenv("MESSAGE_TTL_MS", "30000")),  # milliseconds
            "retry_count": 0,
            "trace_id": None,
        }
    )
    payload: Dict[str, Any] = field(
        default_factory=lambda: {
            "task": None,
            "result": None,
            "error": None,
            "metadata": {},
            "token_usage": {"prompt": 0, "completion": 0},
        }
    )


@dataclass
class QueryContext:
    """Context information for query processing."""

    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    domains: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    token_budget: int = int(os.getenv("DEFAULT_TOKEN_BUDGET", "1000"))
    timeout_ms: int = int(os.getenv("AGENT_TIMEOUT_MS", "5000"))
    user_context: Optional[Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None


@dataclass
class AgentResult:
    """Standard result format from agent execution."""

    success: bool
    data: Any
    confidence: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    execution_time_ms: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Provides common functionality for message handling, health checks, and metrics.
    """

    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        # Initialize message queue lazily to avoid event loop issues
        self._message_queue = None
        self.metrics = defaultdict(int)
        self.is_running = False
        self.health_status = "healthy"
        self.start_time = time.time()

    @property
    def message_queue(self):
        """Lazy initialization of message queue."""
        if self._message_queue is None:
            try:
                self._message_queue = asyncio.Queue()
            except RuntimeError:
                # If no event loop is running, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._message_queue = asyncio.Queue()
        return self._message_queue

    @abstractmethod
    async def process_task(self, task: Dict[str, Any], context: QueryContext) -> AgentResult:
        """
        Process a specific task. Must be implemented by each agent.

        Args:
            task: Task data to process
            context: Query context information

        Returns:
            AgentResult with processing results
        """
        pass

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages and route to appropriate handlers."""
        try:
            if message.header["message_type"] == MessageType.TASK:
                result = await self.process_task(
                    message.payload["task"], message.payload.get("context", QueryContext(query=""))
                )
                return self._create_result_message(message, result)
            elif message.header["message_type"] == MessageType.CONTROL:
                return await self._handle_control_message(message)
            elif message.header["message_type"] == MessageType.HEARTBEAT:
                return self._create_heartbeat_response(message)
        except Exception as e:
            logger.error(f"Error handling message in {self.agent_id}: {str(e)}")
            return self._create_error_message(message, str(e))

    async def _handle_control_message(self, message: AgentMessage) -> AgentMessage:
        """Handle control messages (pause, resume, shutdown)."""
        control_action = message.payload.get("action", "")

        if control_action == "pause":
            self.is_running = False
            logger.info(f"Agent {self.agent_id} paused")
            return self._create_control_response(message, "paused", "Agent paused successfully")

        elif control_action == "resume":
            self.is_running = True
            logger.info(f"Agent {self.agent_id} resumed")
            return self._create_control_response(message, "resumed", "Agent resumed successfully")

        elif control_action == "shutdown":
            self.is_running = False
            logger.info(f"Agent {self.agent_id} shutting down")
            return self._create_control_response(message, "shutdown", "Agent shutdown initiated")

        else:
            logger.warning(f"Unknown control action: {control_action}")
            return self._create_error_message(message, f"Unknown control action: {control_action}")

    def _create_control_response(
        self, original: AgentMessage, status: str, message: str
    ) -> AgentMessage:
        """Create a control response message."""
        return AgentMessage(
            header={
                **original.header,
                "message_type": MessageType.RESULT,
                "timestamp": time.time(),
                "sender_agent": self.agent_id,
                "recipient_agent": original.header.get("sender_agent"),
            },
            payload={
                "control_status": status,
                "message": message,
                "agent_id": self.agent_id,
                "metadata": {"control_action": original.payload.get("action", "")},
            },
        )

    def _create_result_message(self, original: AgentMessage, result: AgentResult) -> AgentMessage:
        """Create a result message from an agent result."""
        return AgentMessage(
            header={
                **original.header,
                "message_type": MessageType.RESULT,
                "timestamp": time.time(),
                "sender_agent": self.agent_id,
                "recipient_agent": original.header.get("sender_agent"),
            },
            payload={
                "result": result,
                "metadata": result.metadata,
                "token_usage": result.token_usage,
            },
        )

    def _create_error_message(self, original: AgentMessage, error: str) -> AgentMessage:
        """Create an error message."""
        return AgentMessage(
            header={
                **original.header,
                "message_type": MessageType.ERROR,
                "timestamp": time.time(),
                "sender_agent": self.agent_id,
                "recipient_agent": original.header.get("sender_agent"),
            },
            payload={"error": error, "metadata": {"agent_id": self.agent_id}},
        )

    def _create_heartbeat_response(self, original: AgentMessage) -> AgentMessage:
        """Create a heartbeat response."""
        return AgentMessage(
            header={
                **original.header,
                "message_type": MessageType.HEARTBEAT,
                "timestamp": time.time(),
                "sender_agent": self.agent_id,
                "recipient_agent": original.header.get("sender_agent"),
            },
            payload={
                "health_status": self.health_status,
                "uptime": time.time() - self.start_time,
                "metrics": dict(self.metrics),
            },
        )

    async def start(self):
        """Start the agent's message processing loop."""
        self.is_running = True
        logger.info(f"Agent {self.agent_id} started")

        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(timeout=1.0), timeout=1.0)
                response = await self.handle_message(message)
                if response:
                    # TODO: Send response back through message broker
                    logger.debug(f"Processed message: {message.header['message_id']}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in agent loop: {str(e)}")

    async def stop(self):
        """Stop the agent's message processing loop."""
        self.is_running = False
        logger.info(f"Agent {self.agent_id} stopped")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status and metrics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.health_status,
            "uptime": time.time() - self.start_time,
            "metrics": dict(self.metrics),
            "is_running": self.is_running,
        }

    def record_metric(self, metric_name: str, value: int = 1):
        """Record a metric for monitoring."""
        self.metrics[metric_name] += value

    async def health_check(self) -> bool:
        """Perform a health check on the agent."""
        try:
            # Basic health check - can be overridden by subclasses
            return self.is_running and self.health_status == "healthy"
        except Exception as e:
            logger.error(f"Health check failed for {self.agent_id}: {str(e)}")
            return False
