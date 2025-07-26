"""
Unified LLM client for OpenAI and Anthropic APIs.
Provides a consistent interface for both providers.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import openai
import anthropic
from core.config import config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    model: str
    provider: str
    token_usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any] = None


class UnifiedLLMClient:
    """
    Unified client for OpenAI and Anthropic APIs.
    Automatically selects the best available provider.
    """
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients."""
        try:
            if config.openai_api_key:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=config.openai_api_key)
                logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        try:
            if config.anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
                logger.info("✅ Anthropic client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        providers = []
        if self.openai_client:
            providers.append("openai")
        if self.anthropic_client:
            providers.append("anthropic")
        return providers
    
    async def generate_text(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate text using the specified or best available provider.
        
        Args:
            prompt: The input prompt
            provider: Specific provider to use ("openai" or "anthropic")
            model: Specific model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_message: System message for the conversation
        
        Returns:
            LLMResponse with generated content and metadata
        """
        # Determine provider
        if provider:
            if provider == "openai" and not self.openai_client:
                raise ValueError("OpenAI client not available")
            elif provider == "anthropic" and not self.anthropic_client:
                raise ValueError("Anthropic client not available")
        else:
            # Auto-select best available provider
            providers = self.get_available_providers()
            if not providers:
                raise ValueError("No LLM providers available")
            provider = providers[0]  # Use first available
        
        # Set defaults
        if model is None:
            model = config.openai_model if provider == "openai" else config.anthropic_model
        if max_tokens is None:
            max_tokens = config.openai_max_tokens if provider == "openai" else config.anthropic_max_tokens
        if temperature is None:
            temperature = config.openai_temperature
        
        try:
            if provider == "openai":
                return await self._generate_openai(prompt, model, max_tokens, temperature, system_message)
            elif provider == "anthropic":
                return await self._generate_anthropic(prompt, model, max_tokens, temperature, system_message)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            logger.error(f"Error generating text with {provider}: {e}")
            raise
    
    async def _generate_openai(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_message: Optional[str] = None
    ) -> LLMResponse:
        """Generate text using OpenAI."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = await asyncio.to_thread(
            self.openai_client.chat.completions.create,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider="openai",
            token_usage={
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason,
            metadata={"response_id": response.id}
        )
    
    async def _generate_anthropic(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_message: Optional[str] = None
    ) -> LLMResponse:
        """Generate text using Anthropic."""
        # Anthropic uses system parameter instead of system message in messages
        response = await asyncio.to_thread(
            self.anthropic_client.messages.create,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message or "",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=model,
            provider="anthropic",
            token_usage={
                "prompt": response.usage.input_tokens,
                "completion": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason,
            metadata={"response_id": response.id}
        )
    
    async def generate_embeddings(
        self,
        text: str,
        provider: Optional[str] = None
    ) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            provider: Specific provider to use
        
        Returns:
            List of embedding values
        """
        if provider and provider not in self.get_available_providers():
            raise ValueError(f"Provider {provider} not available")
        
        if not provider:
            provider = "openai" if self.openai_client else "anthropic"
        
        try:
            if provider == "openai":
                return await self._generate_openai_embeddings(text)
            elif provider == "anthropic":
                return await self._generate_anthropic_embeddings(text)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            logger.error(f"Error generating embeddings with {provider}: {e}")
            raise
    
    async def _generate_openai_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        response = await asyncio.to_thread(
            self.openai_client.embeddings.create,
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    
    async def _generate_anthropic_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Anthropic."""
        # Anthropic doesn't have embeddings in the current version
        # For now, we'll use OpenAI embeddings as fallback
        logger.warning("Anthropic embeddings not available, using OpenAI as fallback")
        return await self._generate_openai_embeddings(text)


# Global LLM client instance
llm_client = UnifiedLLMClient() 