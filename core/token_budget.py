"""
Token Budget Controller - Manages API token usage per query/agent to control costs.
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from agents.base_agent import AgentType

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when token budget is exceeded."""
    pass


@dataclass
class TokenUsage:
    """Represents token usage for a specific period."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timestamp: float = field(default_factory=time.time)


class BudgetPeriod(Enum):
    """Budget periods for token tracking."""
    QUERY = "query"
    SESSION = "session"
    DAILY = "daily"
    MONTHLY = "monthly"


class TokenBudgetController:
    """
    Manages token budgets and usage tracking for cost control.
    """
    
    def __init__(self, daily_budget: int = 1_000_000, query_budget: int = 10_000):
        self.daily_budget = daily_budget
        self.query_budget = query_budget
        self.daily_usage: Dict[str, TokenUsage] = {}
        self.query_usage: Dict[str, TokenUsage] = {}
        self.agent_budgets: Dict[AgentType, float] = {
            AgentType.RETRIEVAL: 0.3,      # 30% of query budget
            AgentType.FACT_CHECK: 0.25,    # 25% of query budget
            AgentType.SYNTHESIS: 0.35,     # 35% of query budget
            AgentType.CITATION: 0.1        # 10% of query budget
        }
        
    def allocate_budget_for_query(self, query: str) -> int:
        """
        Allocate token budget for a query based on complexity.
        
        Args:
            query: The query to allocate budget for
            
        Returns:
            Allocated token budget
        """
        # Simple heuristic: longer queries get more tokens
        base_budget = min(len(query.split()) * 10, self.query_budget)
        
        # Adjust based on query complexity indicators
        complexity_multiplier = self._calculate_complexity_multiplier(query)
        allocated_budget = int(base_budget * complexity_multiplier)
        
        logger.info(f"Allocated {allocated_budget} tokens for query: {query[:50]}...")
        return allocated_budget
    
    def get_agent_budget(self, agent_type: AgentType, total_budget: int) -> int:
        """
        Get the budget allocation for a specific agent.
        
        Args:
            agent_type: The type of agent
            total_budget: Total available budget
            
        Returns:
            Budget allocation for the agent
        """
        if agent_type not in self.agent_budgets:
            logger.warning(f"No budget allocation found for agent type: {agent_type}")
            return total_budget // 4  # Default equal split
        
        allocation = self.agent_budgets[agent_type]
        return int(total_budget * allocation)
    
    def track_usage(self, agent_type: AgentType, tokens_used: int, 
                   prompt_tokens: int = 0, completion_tokens: int = 0):
        """
        Track token usage for an agent.
        
        Args:
            agent_type: The agent that used tokens
            tokens_used: Total tokens used
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
        """
        today = time.strftime("%Y-%m-%d")
        
        # Track daily usage
        if today not in self.daily_usage:
            self.daily_usage[today] = TokenUsage()
        
        self.daily_usage[today].total_tokens += tokens_used
        self.daily_usage[today].prompt_tokens += prompt_tokens
        self.daily_usage[today].completion_tokens += completion_tokens
        
        logger.debug(f"Tracked {tokens_used} tokens for {agent_type.value}")
        
        # Check daily budget
        if self.daily_usage[today].total_tokens > self.daily_budget:
            logger.warning(f"Daily budget exceeded: {self.daily_usage[today].total_tokens}/{self.daily_budget}")
    
    def check_budget_available(self, query_id: str, required_tokens: int) -> bool:
        """
        Check if budget is available for a query.
        
        Args:
            query_id: Unique identifier for the query
            required_tokens: Tokens required
            
        Returns:
            True if budget is available, False otherwise
        """
        if query_id not in self.query_usage:
            return required_tokens <= self.query_budget
        
        used_tokens = self.query_usage[query_id].total_tokens
        available_tokens = self.query_budget - used_tokens
        
        return required_tokens <= available_tokens
    
    def reserve_tokens(self, query_id: str, tokens: int) -> bool:
        """
        Reserve tokens for a query.
        
        Args:
            query_id: Unique identifier for the query
            tokens: Tokens to reserve
            
        Returns:
            True if reservation successful, False otherwise
        """
        if not self.check_budget_available(query_id, tokens):
            logger.warning(f"Cannot reserve {tokens} tokens for query {query_id}")
            return False
        
        if query_id not in self.query_usage:
            self.query_usage[query_id] = TokenUsage()
        
        self.query_usage[query_id].total_tokens += tokens
        logger.debug(f"Reserved {tokens} tokens for query {query_id}")
        return True
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        today = time.strftime("%Y-%m-%d")
        daily_usage = self.daily_usage.get(today, TokenUsage())
        
        return {
            'daily_budget': self.daily_budget,
            'query_budget': self.query_budget,
            'daily_used': daily_usage.total_tokens,
            'daily_remaining': self.daily_budget - daily_usage.total_tokens,
            'daily_usage_percentage': (daily_usage.total_tokens / self.daily_budget) * 100,
            'active_queries': len(self.query_usage),
            'agent_budgets': {agent.value: allocation for agent, allocation in self.agent_budgets.items()}
        }
    
    def reset_query_usage(self, query_id: str):
        """Reset usage tracking for a specific query."""
        if query_id in self.query_usage:
            del self.query_usage[query_id]
            logger.debug(f"Reset usage tracking for query {query_id}")
    
    def _calculate_complexity_multiplier(self, query: str) -> float:
        """
        Calculate complexity multiplier based on query characteristics.
        
        Args:
            query: The query to analyze
            
        Returns:
            Complexity multiplier
        """
        # Simple heuristics for complexity
        words = query.split()
        complexity_score = 1.0
        
        # Longer queries are more complex
        if len(words) > 20:
            complexity_score *= 1.5
        elif len(words) > 10:
            complexity_score *= 1.2
        
        # Technical terms indicate complexity
        technical_terms = ['algorithm', 'implementation', 'architecture', 'framework', 
                          'protocol', 'specification', 'methodology', 'paradigm']
        tech_count = sum(1 for word in words if word.lower() in technical_terms)
        if tech_count > 0:
            complexity_score *= (1 + tech_count * 0.1)
        
        # Question words indicate research queries
        question_words = ['how', 'why', 'what', 'when', 'where', 'which']
        if any(word.lower() in question_words for word in words):
            complexity_score *= 1.3
        
        return min(complexity_score, 3.0)  # Cap at 3x 