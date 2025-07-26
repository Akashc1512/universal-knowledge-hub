"""
Message Broker - Handles inter-agent messaging and communication.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
from agents.base_agent import AgentMessage, MessageType

logger = logging.getLogger(__name__)


class MessageBroker:
    """
    Handles inter-agent messaging and communication.
    Provides publish/subscribe pattern for agent communication.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queues: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
        self.is_running = False
        self.message_history: List[AgentMessage] = []
        self.max_history = 1000
        
    async def publish(self, message: AgentMessage):
        """
        Publish a message to the broker.
        
        Args:
            message: The message to publish
        """
        # Store in history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # Route to subscribers
        recipient = message.header.get('recipient_agent')
        if recipient:
            # Direct message to specific agent
            if recipient in self.message_queues:
                await self.message_queues[recipient].put(message)
                logger.debug(f"Published message to {recipient}: {message.header['message_id']}")
            else:
                logger.warning(f"No queue found for recipient: {recipient}")
        else:
            # Broadcast to all subscribers of this message type
            message_type = message.header.get('message_type', MessageType.TASK)
            for callback in self.subscribers.get(str(message_type), []):
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error in message callback: {str(e)}")
    
    async def subscribe(self, agent_id: str, message_types: List[MessageType]):
        """
        Subscribe an agent to specific message types.
        
        Args:
            agent_id: The agent ID to subscribe
            message_types: List of message types to subscribe to
        """
        for msg_type in message_types:
            self.subscribers[str(msg_type)].append(
                lambda msg, aid=agent_id: self._route_to_agent(aid, msg)
            )
        
        logger.info(f"Agent {agent_id} subscribed to {len(message_types)} message types")
    
    async def get_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """
        Get the next message for an agent.
        
        Args:
            agent_id: The agent ID to get messages for
            timeout: Timeout in seconds
            
        Returns:
            The next message or None if timeout
        """
        queue = self.message_queues[agent_id]
        
        try:
            if timeout:
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                message = await queue.get()
            
            logger.debug(f"Agent {agent_id} received message: {message.header['message_id']}")
            return message
            
        except asyncio.TimeoutError:
            logger.debug(f"Timeout waiting for message for agent {agent_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting message for agent {agent_id}: {str(e)}")
            return None
    
    def unsubscribe(self, agent_id: str, message_types: List[MessageType]):
        """
        Unsubscribe an agent from specific message types.
        
        Args:
            agent_id: The agent ID to unsubscribe
            message_types: List of message types to unsubscribe from
        """
        for msg_type in message_types:
            msg_type_str = str(msg_type)
            if msg_type_str in self.subscribers:
                # Remove callbacks for this agent
                self.subscribers[msg_type_str] = [
                    cb for cb in self.subscribers[msg_type_str]
                    if not hasattr(cb, '__name__') or cb.__name__ != f'_route_to_agent_{agent_id}'
                ]
        
        # Clear agent's message queue
        if agent_id in self.message_queues:
            del self.message_queues[agent_id]
        
        logger.info(f"Agent {agent_id} unsubscribed from {len(message_types)} message types")
    
    async def broadcast(self, message: AgentMessage, exclude_agent: Optional[str] = None):
        """
        Broadcast a message to all subscribed agents.
        
        Args:
            message: The message to broadcast
            exclude_agent: Optional agent ID to exclude from broadcast
        """
        for agent_id in list(self.message_queues.keys()):
            if agent_id != exclude_agent:
                await self.message_queues[agent_id].put(message)
        
        logger.debug(f"Broadcasted message to {len(self.message_queues)} agents")
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get message broker statistics."""
        total_subscribers = sum(len(callbacks) for callbacks in self.subscribers.values())
        total_queues = len(self.message_queues)
        queue_sizes = {agent_id: queue.qsize() for agent_id, queue in self.message_queues.items()}
        
        return {
            'total_subscribers': total_subscribers,
            'active_queues': total_queues,
            'queue_sizes': queue_sizes,
            'message_history_size': len(self.message_history),
            'subscriber_types': {msg_type: len(callbacks) for msg_type, callbacks in self.subscribers.items()}
        }
    
    def clear_history(self):
        """Clear message history."""
        self.message_history.clear()
        logger.info("Message history cleared")
    
    async def _route_to_agent(self, agent_id: str, message: AgentMessage):
        """Route a message to a specific agent."""
        if agent_id in self.message_queues:
            await self.message_queues[agent_id].put(message)
        else:
            logger.warning(f"Attempted to route message to non-existent agent: {agent_id}")
    
    async def start(self):
        """Start the message broker."""
        self.is_running = True
        logger.info("Message broker started")
        
        # Keep the broker running
        while self.is_running:
            await asyncio.sleep(1)
    
    async def stop(self):
        """Stop the message broker."""
        self.is_running = False
        logger.info("Message broker stopped") 