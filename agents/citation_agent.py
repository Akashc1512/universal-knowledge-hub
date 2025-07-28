"""
Advanced citation agent that generates proper citations for sources.
"""

import asyncio
import logging
import time
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents.base_agent import BaseAgent, AgentType, AgentMessage, AgentResult, QueryContext
from agents.data_models import CitationResult, CitationModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
DEFAULT_TOKEN_BUDGET = int(os.getenv("DEFAULT_TOKEN_BUDGET", "1000"))


@dataclass
class Citation:
    """Represents a citation with metadata."""

    id: str
    text: str
    url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "author": self.author,
            "date": self.date,
            "source": self.source,
            "confidence": self.confidence,
        }


class CitationAgent(BaseAgent):
    """
    CitationAgent that generates proper citations and integrates them into the answer text.
    """

    def __init__(self):
        """Initialize the citation agent."""
        super().__init__(agent_id="citation_agent", agent_type=AgentType.CITATION)

        # Initialize citation formats - using generic format for all styles
        self.citation_formats = {
            "academic": self._format_citation,
            "apa": self._format_citation,
            "mla": self._format_citation,
            "chicago": self._format_citation,
            "url": self._format_citation,
        }

        logger.info("✅ CitationAgent initialized successfully")

    async def process_task(self, task: Dict[str, Any], context: QueryContext) -> AgentResult:
        """
        Process citation task by generating proper citations for content and sources.

        Args:
            task: Task data containing content and sources
            context: Query context

        Returns:
            AgentResult with cited content
        """
        start_time = time.time()

        try:
            # Extract task data
            content = task.get("content", "")
            sources = task.get("sources", [])
            citation_format = task.get(
                "format",
                context.citation_format if hasattr(context, "citation_format") else "academic",
            )

            logger.info(f"Generating citations in {citation_format} format")
            logger.info(f"Content length: {len(content)} characters")
            logger.info(f"Number of sources: {len(sources)}")

            # Validate input
            if not content:
                return AgentResult(
                    success=False, error="No content provided for citation", confidence=0.0
                )

            # Generate citations
            citations = await self._generate_citations(sources, citation_format)

            # Integrate citations into content
            cited_content = await self._integrate_citations(content, sources, citations)

            processing_time = time.time() - start_time

            # Create standardized citation result
            citation_data = CitationResult(
                cited_content=cited_content,
                citations=citations,
                citation_format=citation_format,
                total_sources=len(sources),
                metadata={
                    "agent_id": self.agent_id,
                    "processing_time_ms": int(processing_time * 1000),
                },
            )

            return AgentResult(
                success=True,
                data=citation_data.dict(),
                confidence=1.0,  # Citations are deterministic
                execution_time_ms=int(processing_time * 1000),
            )

        except Exception as e:
            logger.error(f"Citation generation failed: {str(e)}")
            return AgentResult(
                success=False, error=f"Citation generation failed: {str(e)}", confidence=0.0
            )

    async def _generate_citations(self, sources: List[Dict], style: str) -> List[Citation]:
        """
        Generate citations from sources.

        Args:
            sources: List of source documents
            style: Citation style (APA, MLA, etc.)

        Returns:
            List of Citation objects
        """
        citations = []

        for i, source in enumerate(sources):
            try:
                # Extract metadata from source
                source_id = source.get("doc_id", f"source_{i+1}")
                content = source.get("content", "")
                metadata = source.get("metadata", {})

                # Generate citation text
                citation_text = await self._format_citation(source, style)

                citation = Citation(
                    id=source_id,
                    text=citation_text,
                    url=metadata.get("url"),
                    title=metadata.get("title"),
                    author=metadata.get("author"),
                    date=metadata.get("date"),
                    source=metadata.get("source"),
                    confidence=source.get("score", 1.0),
                )

                citations.append(citation)

            except Exception as e:
                logger.warning(f"Failed to generate citation for source {i}: {e}")
                continue

        return citations

    async def _format_citation(self, source: Dict, style: str) -> str:
        """
        Format citation according to specified style.

        Args:
            source: Source document
            style: Citation style

        Returns:
            Formatted citation text
        """
        metadata = source.get("metadata", {})

        if style.upper() == "APA":
            author = metadata.get("author", "Unknown Author")
            title = metadata.get("title", "Untitled")
            date = metadata.get("date", "n.d.")
            url = metadata.get("url", "")

            if url:
                return f"{author} ({date}). {title}. Retrieved from {url}"
            else:
                return f"{author} ({date}). {title}."

        elif style.upper() == "MLA":
            author = metadata.get("author", "Unknown Author")
            title = metadata.get("title", "Untitled")
            date = metadata.get("date", "n.d.")
            url = metadata.get("url", "")

            if url:
                return f'"{title}." {author}, {date}, {url}.'
            else:
                return f'"{title}." {author}, {date}.'

        else:
            # Default format
            author = metadata.get("author", "Unknown Author")
            title = metadata.get("title", "Untitled")
            date = metadata.get("date", "n.d.")
            return f"{author} ({date}). {title}."

    async def _integrate_citations(
        self, content: str, sources: List[Dict], citations: List[Citation]
    ) -> str:
        """
        Integrate citations into content text.

        Args:
            content: Original content
            sources: List of source documents
            citations: List of citations

        Returns:
            Tuple of (cited_content, in_text_citations)
        """
        if not citations:
            return content

        cited_content = content

        # Simple citation integration - add citation numbers to sentences
        sentences = re.split(r"[.!?]+", content)
        cited_sentences = []

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Find relevant citations for this sentence
                relevant_citations = self._find_relevant_citations(sentence, citations)

                if relevant_citations:
                    # Add citation numbers to sentence
                    citation_numbers = [str(cit.id) for cit in relevant_citations]
                    cited_sentence = f"{sentence.strip()} [{', '.join(citation_numbers)}]."

                    # Track in-text citations
                    for citation in relevant_citations:
                        # Implement in-text citation tracking
                        citation_text = f" [{citation.source}]"
                        cited_sentence = f"{sentence.strip()}{citation_text}."
                        
                        # Track citation usage for analytics
                        logger.info(
                            "Citation used in sentence",
                            citation_id=citation.id,
                            source=citation.source,
                            sentence_preview=sentence[:100]
                        )
                else:
                    cited_sentence = f"{sentence.strip()}."

                cited_sentences.append(cited_sentence)

        cited_content = " ".join(cited_sentences)

        return cited_content

    def _find_relevant_citations(self, sentence: str, citations: List[Citation]) -> List[Citation]:
        """
        Find citations relevant to a sentence.

        Args:
            sentence: Sentence to find citations for
            citations: List of available citations

        Returns:
            List of relevant citations
        """
        relevant_citations = []

        # Simple keyword matching
        sentence_lower = sentence.lower()

        for citation in citations:
            # Check if citation text contains keywords from sentence
            citation_text_lower = citation.text.lower()

            # Extract key terms from sentence (simple approach)
            words = sentence_lower.split()
            key_words = [w for w in words if len(w) > 3]  # Focus on longer words

            # Check if any key words appear in citation
            for word in key_words:
                if word in citation_text_lower:
                    relevant_citations.append(citation)
                    break

        return relevant_citations[:3]  # Limit to 3 citations per sentence

    async def _generate_bibliography(self, citations: List[Citation], style: str) -> List[str]:
        """
        Generate bibliography from citations.

        Args:
            citations: List of citations
            style: Citation style

        Returns:
            List of bibliography entries
        """
        bibliography = []

        for citation in citations:
            bibliography.append(citation.text)

        return bibliography


# Example usage
async def main():
    """Example usage of CitationAgent."""
    agent = CitationAgent()

    # Example content and sources
    content = "The Earth orbits around the Sun. The Sun is a star."
    sources = [
        {
            "doc_id": "source_1",
            "content": "Information about Earth's orbit",
            "metadata": {
                "author": "NASA",
                "title": "Solar System Overview",
                "date": "2023",
                "url": "https://nasa.gov/solar-system",
            },
        },
        {
            "doc_id": "source_2",
            "content": "Information about the Sun",
            "metadata": {
                "author": "Astronomy Society",
                "title": "Stars and Stellar Evolution",
                "date": "2023",
                "url": "https://astronomy.org/stars",
            },
        },
    ]

    task = {"content": content, "sources": sources, "format": "APA"}

    context = QueryContext(query="What is the relationship between Earth and the Sun?")

    result = await agent.process_task(task, context)
    print(f"Success: {result.success}")
    print(f"Cited Content: {result.data.get('cited_content', '')}")
    print(f"Bibliography: {result.data.get('bibliography', [])}")


if __name__ == "__main__":
    asyncio.run(main())
