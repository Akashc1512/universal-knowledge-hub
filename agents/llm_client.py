# --- LLMClient: Unified OpenAI + Anthropic (Claude) LLM Integration ---
# Required environment variables:
#   LLM_PROVIDER=openai|anthropic
#   OPENAI_API_KEY (if using OpenAI)
#   OPENAI_LLM_MODEL (default: gpt-3.5-turbo)
#   ANTHROPIC_API_KEY (if using Anthropic)
#   ANTHROPIC_MODEL (default: claude-3-opus-20240229)

import os
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if self.provider == "openai":
            import openai

            self.openai = openai
            self.openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo")
        elif self.provider == "anthropic":
            import anthropic

            self.anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def get_provider(self):
        return self.provider

    def get_model(self):
        return self.model

    def get_llm_name(self):
        if self.provider == "openai":
            return f"openai:{self.model}"
        elif self.provider == "anthropic":
            return f"anthropic:{self.model}"
        return "unknown"

    def synthesize(self, prompt: str, max_tokens: int = 500, temperature: float = 0.2) -> str:
        if self.provider == "openai":
            try:
                response = self.openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful, factual assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.error(f"OpenAI LLM error: {e}")
                raise
        elif self.provider == "anthropic":
            try:
                # Claude v3 API (anthropic>=0.3.11)
                response = self.anthropic.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip() if response.content else ""
            except Exception as e:
                logger.error(f"Anthropic LLM error: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")
