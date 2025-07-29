"""
Advanced fact-checking agent that verifies claims against retrieved documents.
"""

import asyncio
import logging
import time
import os
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents.base_agent import BaseAgent, AgentType, AgentMessage, AgentResult, QueryContext
from agents.data_models import FactCheckResult, VerifiedFactModel, CitationModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
DEFAULT_TOKEN_BUDGET = int(os.getenv("DEFAULT_TOKEN_BUDGET", "1000"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
DATABASE_NAME = os.getenv("DATABASE_NAME", "knowledge_base")


@dataclass
class Claim:
    """Represents a claim to be verified."""

    text: str
    confidence: float = 0.5
    source: str = "extracted"
    metadata: Dict[str, Any] = None


@dataclass
class Verification:
    """Represents the verification result of a claim."""

    claim: str
    is_supported: bool
    confidence: float
    evidence: List[str]
    contradicting_evidence: List[str]
    source_documents: List[str]
    verification_method: str


class FactCheckAgent(BaseAgent):
    """
    FactCheckAgent that verifies claims against retrieved documents.
    """

    def __init__(self):
        """Initialize the fact-checking agent."""
        super().__init__(agent_id="factcheck_agent", agent_type=AgentType.FACT_CHECK)
        self.manual_review_callback: Optional[Callable] = None
        logger.info("✅ FactCheckAgent initialized successfully")

    def set_manual_review_callback(self, callback: Callable):
        """Set callback for manual review of contested claims."""
        self.manual_review_callback = callback

    async def process_task(self, task: Dict[str, Any], context: QueryContext) -> AgentResult:
        """
        Process fact-checking task by verifying claims against documents.

        Args:
            task: Task data containing documents and query
            context: Query context

        Returns:
            AgentResult with verified facts
        """
        start_time = time.time()

        try:
            # Extract task data
            documents = task.get("documents", [])
            query = task.get("query", "")

            logger.info(f"Fact-checking for query: {query[:50]}...")
            logger.info(f"Number of documents: {len(documents)}")

            # Validate input
            if not documents:
                return AgentResult(
                    success=False, error="No documents provided for fact-checking", confidence=0.0
                )

            # Extract claims from query and documents
            claims = await self._extract_claims(query, documents)

            # Verify claims against documents
            verifications = await self._verify_claims(claims, documents)

            # Filter verified facts
            verified_facts = self._filter_verified_facts(verifications)

            # Handle contested claims
            contested_claims = self._identify_contested_claims(verifications)
            if contested_claims and self.manual_review_callback:
                await self._request_manual_review(contested_claims)

            processing_time = time.time() - start_time

            # FIXED: Return verified_facts directly in data to match orchestrator expectations
            return AgentResult(
                success=True,
                data={
                    "verified_facts": verified_facts,  # Direct access for orchestrator
                    "contested_claims": contested_claims,
                    "verification_method": "rule_based", 
                    "total_claims": len(verifications),
                    "metadata": {
                        "agent_id": self.agent_id,
                        "processing_time_ms": int(processing_time * 1000),
                    }
                },
                confidence=self._calculate_verification_confidence(verifications),
                execution_time_ms=int(processing_time * 1000),
            )

        except Exception as e:
            logger.error(f"Fact-checking failed: {str(e)}")
            return AgentResult(
                success=False, error=f"Fact-checking failed: {str(e)}", confidence=0.0
            )

    async def _extract_claims(self, query: str, documents: List[Dict]) -> List[Claim]:
        """
        Extract claims from query and documents.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            List of claims to verify
        """
        claims = []

        # Extract claims from query
        query_claims = self._extract_claims_from_text(query)
        claims.extend(query_claims)

        # Extract claims from documents
        for doc in documents:
            content = doc.get("content", "")
            if content:
                doc_claims = self._extract_claims_from_text(content)
                claims.extend(doc_claims)

        # Remove duplicates and low-confidence claims
        unique_claims = self._deduplicate_claims(claims)

        return unique_claims[:10]  # Limit to top 10 claims

    def _extract_claims_from_text(self, text: str) -> List[Claim]:
        """
        Extract claims from text using pattern matching.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted claims
        """
        claims = []

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            # Look for factual statements
            if self._is_factual_statement(sentence):
                confidence = self._calculate_claim_confidence(sentence)
                claim = Claim(text=sentence, confidence=confidence, source="extracted")
                claims.append(claim)

        return claims

    def _is_factual_statement(self, sentence: str) -> bool:
        """
        Determine if a sentence is a factual statement.

        Args:
            sentence: Sentence to analyze

        Returns:
            True if sentence appears to be factual
        """
        sentence_lower = sentence.lower()

        # Look for factual indicators
        factual_indicators = [
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "contains",
            "includes",
            "consists",
            "comprises",
            "located",
            "found",
            "discovered",
            "identified",
            "according to",
            "research shows",
            "studies indicate",
        ]

        # Look for opinion indicators (negative)
        opinion_indicators = [
            "i think",
            "i believe",
            "in my opinion",
            "i feel",
            "probably",
            "maybe",
            "perhaps",
            "might",
            "could",
            "seems",
            "appears",
            "looks like",
        ]

        # Check for factual indicators
        has_factual = any(indicator in sentence_lower for indicator in factual_indicators)

        # Check for opinion indicators
        has_opinion = any(indicator in sentence_lower for indicator in opinion_indicators)

        return has_factual and not has_opinion

    def _calculate_claim_confidence(self, sentence: str) -> float:
        """
        Calculate confidence for a claim based on its characteristics.

        Args:
            sentence: Claim sentence

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence

        # Boost confidence for specific patterns
        if re.search(r"\d{4}", sentence):  # Contains year
            confidence += 0.1

        if re.search(r"according to|research shows|studies indicate", sentence.lower()):
            confidence += 0.2

        if len(sentence.split()) > 10:  # Longer sentences tend to be more specific
            confidence += 0.1

        return min(1.0, confidence)

    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """
        Remove duplicate claims and sort by confidence.

        Args:
            claims: List of claims

        Returns:
            Deduplicated and sorted claims
        """
        seen_texts = set()
        unique_claims = []

        for claim in claims:
            # Normalize text for comparison
            normalized_text = re.sub(r"\s+", " ", claim.text.lower().strip())

            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_claims.append(claim)

        # Sort by confidence
        unique_claims.sort(key=lambda x: x.confidence, reverse=True)

        return unique_claims

    async def _verify_claims(
        self, claims: List[Claim], documents: List[Dict]
    ) -> List[Verification]:
        """
        Verify claims against documents.

        Args:
            claims: List of claims to verify
            documents: Retrieved documents

        Returns:
            List of verification results
        """
        verifications = []

        for claim in claims:
            verification = await self._verify_single_claim(claim, documents)
            verifications.append(verification)

        return verifications

    async def _verify_single_claim(self, claim: Claim, documents: List[Dict]) -> Verification:
        """
        Verify a single claim against documents using LLM-based analysis.

        Args:
            claim: Claim to verify
            documents: Documents to check against

        Returns:
            Verification result
        """
        supporting_evidence = []
        contradicting_evidence = []
        source_docs = []

        claim_keywords = self._extract_keywords(claim.text)

        for doc in documents:
            doc_content = doc.get("content", "").lower()
            doc_score = doc.get("score", 0)

            # Calculate relevance to claim
            relevance_score = self._calculate_relevance(claim_keywords, doc_content)

            if relevance_score > 0.3:  # Threshold for relevance
                source_docs.append(doc.get("doc_id", "unknown"))

                # Use LLM to analyze evidence
                evidence_analysis = await self._analyze_evidence_with_llm(claim.text, doc_content)
                
                if evidence_analysis["supports"]:
                    supporting_evidence.append(doc_content[:200])
                elif evidence_analysis["contradicts"]:
                    contradicting_evidence.append(doc_content[:200])

        # Determine if claim is supported based on evidence analysis
        is_supported = len(supporting_evidence) > len(contradicting_evidence)

        # Calculate confidence based on evidence quality and quantity
        total_evidence = len(supporting_evidence) + len(contradicting_evidence)
        if total_evidence == 0:
            confidence = 0.1  # Low confidence if no evidence
        else:
            support_ratio = len(supporting_evidence) / total_evidence
            confidence = min(0.9, support_ratio + claim.confidence * 0.3)

        return Verification(
            claim=claim.text,
            is_supported=is_supported,
            confidence=confidence,
            evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            source_documents=source_docs,
            verification_method="llm_analysis",
        )

    async def _analyze_evidence_with_llm(self, claim: str, document_content: str) -> Dict[str, bool]:
        """
        Use LLM to analyze whether document content supports or contradicts a claim.
        
        Args:
            claim: The claim to verify
            document_content: Document content to analyze
            
        Returns:
            Dict with 'supports' and 'contradicts' boolean flags
        """
        try:
            from agents.llm_client import LLMClient
            
            # Create prompt for LLM analysis
            prompt = f"""
            Analyze whether the following document content supports or contradicts the given claim.
            
            Claim: "{claim}"
            
            Document Content: "{document_content[:1000]}"
            
            Please respond with only:
            - "SUPPORTS" if the document content provides evidence that supports the claim
            - "CONTRADICTS" if the document content provides evidence that contradicts the claim  
            - "NEUTRAL" if the document content is not relevant or provides no clear evidence
            
            Response:"""
            
            llm_client = LLMClient()
            response = await llm_client.generate_text(prompt, max_tokens=50, temperature=0.1)
            
            response_upper = response.strip().upper()
            
            return {
                "supports": "SUPPORTS" in response_upper,
                "contradicts": "CONTRADICTS" in response_upper,
                "neutral": "NEUTRAL" in response_upper or response_upper not in ["SUPPORTS", "CONTRADICTS"]
            }
            
        except Exception as e:
            logger.error(f"LLM evidence analysis failed: {e}")
            # Fallback to keyword-based analysis
            return self._fallback_evidence_analysis(claim, document_content)
    
    def _fallback_evidence_analysis(self, claim: str, document_content: str) -> Dict[str, bool]:
        """
        Fallback evidence analysis using keyword matching when LLM is unavailable.
        
        Args:
            claim: The claim to verify
            document_content: Document content to analyze
            
        Returns:
            Dict with 'supports' and 'contradicts' boolean flags
        """
        claim_lower = claim.lower()
        content_lower = document_content.lower()
        
        # Extract key terms from claim
        claim_terms = set(re.findall(r'\b\w+\b', claim_lower))
        claim_terms = {term for term in claim_terms if len(term) > 3}  # Filter short words
        
        # Check for supporting evidence
        supports = False
        contradicts = False
        
        # Look for supporting indicators
        support_indicators = ["confirm", "support", "evidence", "prove", "demonstrate", "show", "indicate"]
        for indicator in support_indicators:
            if indicator in content_lower and any(term in content_lower for term in claim_terms):
                supports = True
                break
        
        # Look for contradicting indicators  
        contradict_indicators = ["contradict", "refute", "disprove", "false", "incorrect", "wrong", "disagree"]
        for indicator in contradict_indicators:
            if indicator in content_lower and any(term in content_lower for term in claim_terms):
                contradicts = True
                break
        
        return {
            "supports": supports,
            "contradicts": contradicts,
            "neutral": not supports and not contradicts
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        # Simple keyword extraction
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out common words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        keywords = [word for word in words if word not in stop_words and len(word) > 3]

        return keywords[:10]  # Limit to top 10 keywords

    def _calculate_relevance(self, keywords: List[str], content: str) -> float:
        """
        Calculate relevance between keywords and content.

        Args:
            keywords: List of keywords
            content: Content to check against

        Returns:
            Relevance score between 0 and 1
        """
        if not keywords or not content:
            return 0.0

        # Simple keyword matching
        matches = sum(1 for keyword in keywords if keyword in content)
        relevance = matches / len(keywords)

        return relevance

    def _find_supporting_evidence(self, claim: str, content: str) -> bool:
        """
        Check if content supports the claim.

        Args:
            claim: Claim to check
            content: Content to check against

        Returns:
            True if content supports the claim
        """
        claim_lower = claim.lower()
        content_lower = content.lower()

        # Extract key terms from claim
        claim_terms = set(re.findall(r"\b\w+\b", claim_lower))

        # Check if key terms appear in content
        content_terms = set(re.findall(r"\b\w+\b", content_lower))

        # Calculate overlap
        overlap = len(claim_terms.intersection(content_terms))
        overlap_ratio = overlap / len(claim_terms) if claim_terms else 0

        return overlap_ratio > 0.3  # Threshold for support

    def _find_contradicting_evidence(self, claim: str, content: str) -> bool:
        """
        Check if content contradicts the claim.

        Args:
            claim: Claim to check
            content: Content to check against

        Returns:
            True if content contradicts the claim
        """
        # Simple contradiction detection
        contradiction_indicators = [
            "however",
            "but",
            "although",
            "despite",
            "nevertheless",
            "on the other hand",
            "in contrast",
            "unlike",
            "different from",
        ]

        content_lower = content.lower()

        # Check for contradiction indicators
        has_contradiction_indicators = any(
            indicator in content_lower for indicator in contradiction_indicators
        )

        # Check for negation of claim terms
        claim_terms = set(re.findall(r"\b\w+\b", claim.lower()))
        negation_words = {"not", "no", "never", "none", "neither", "nor"}

        has_negation = any(term in content_lower for term in negation_words)

        return has_contradiction_indicators or has_negation

    def _filter_verified_facts(self, verifications: List[Verification]) -> List[VerifiedFactModel]:
        """
        Filter verified facts from verifications.

        Args:
            verifications: List of verification results

        Returns:
            List of verified facts
        """
        verified_facts = []

        for verification in verifications:
            if verification.is_supported and verification.confidence > 0.6:
                verified_facts.append(
                    VerifiedFactModel(
                        claim=verification.claim,
                        confidence=verification.confidence,
                        source="fact_check_agent",
                        evidence=verification.evidence,
                        contradicting_evidence=verification.contradicting_evidence,
                        verification_method=verification.verification_method,
                        metadata={
                            "source_documents": verification.source_documents,
                        }
                    )
                )

        return verified_facts

    def _identify_contested_claims(self, verifications: List[Verification]) -> List[Dict]:
        """
        Identify claims that need manual review.

        Args:
            verifications: List of verification results

        Returns:
            List of contested claims
        """
        contested_claims = []

        for verification in verifications:
            # Claims with mixed evidence or low confidence
            if (
                len(verification.evidence) > 0 and len(verification.contradicting_evidence) > 0
            ) or verification.confidence < 0.5:
                contested_claims.append(
                    {
                        "claim": verification.claim,
                        "confidence": verification.confidence,
                        "supporting_evidence": verification.evidence,
                        "contradicting_evidence": verification.contradicting_evidence,
                        "source_documents": verification.source_documents,
                    }
                )

        return contested_claims

    async def _request_manual_review(self, contested_claims: List[Dict]):
        """
        Request manual review for contested claims.
        
        Args:
            contested_claims: List of claims that need expert review
        """
        try:
            if self.manual_review_callback:
                logger.info(f"Requesting manual review for {len(contested_claims)} contested claims")
                await self.manual_review_callback(contested_claims)
            else:
                # Log contested claims for manual review
                logger.warning(
                    "Contested claims detected but no manual review callback configured",
                    contested_claims=contested_claims
                )
                
                # Store for later review
                await self._store_contested_claims(contested_claims)
                
        except Exception as e:
            logger.error(f"Manual review callback failed: {e}")
    
    async def _store_contested_claims(self, contested_claims: List[Dict]):
        """
        Store contested claims for later manual review.
        
        Args:
            contested_claims: List of claims to store
        """
        try:
            # Create review request
            review_request = {
                "id": f"review_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "claims": contested_claims,
                "status": "pending",
                "assigned_expert": None,
                "review_notes": None,
                "final_decision": None
            }
            
            # Store in database or file system
            await self._save_review_request(review_request)
            
            logger.info(f"Stored {len(contested_claims)} contested claims for manual review")
            
        except Exception as e:
            logger.error(f"Failed to store contested claims: {e}")
    
    async def _save_review_request(self, review_request: Dict):
        """
        Save review request to persistent storage.
        
        Args:
            review_request: Review request to save
        """
        try:
            # In production, save to database
            # For now, save to file system
            import json
            import os
            
            review_dir = "data/manual_reviews"
            os.makedirs(review_dir, exist_ok=True)
            
            filename = f"{review_request['id']}.json"
            filepath = os.path.join(review_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(review_request, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save review request: {e}")
    
    async def get_pending_reviews(self) -> List[Dict]:
        """
        Get list of pending manual reviews.
        
        Returns:
            List of pending review requests
        """
        try:
            import json
            import os
            import glob
            
            review_dir = "data/manual_reviews"
            if not os.path.exists(review_dir):
                return []
            
            pending_reviews = []
            for filepath in glob.glob(os.path.join(review_dir, "*.json")):
                try:
                    with open(filepath, 'r') as f:
                        review = json.load(f)
                        if review.get("status") == "pending":
                            pending_reviews.append(review)
                except Exception as e:
                    logger.error(f"Failed to load review from {filepath}: {e}")
            
            return pending_reviews
            
        except Exception as e:
            logger.error(f"Failed to get pending reviews: {e}")
            return []
    
    async def update_review_decision(self, review_id: str, decision: Dict):
        """
        Update a manual review with expert decision.
        
        Args:
            review_id: ID of the review to update
            decision: Expert decision with notes and final verdict
        """
        try:
            import json
            import os
            
            review_dir = "data/manual_reviews"
            filepath = os.path.join(review_dir, f"{review_id}.json")
            
            if not os.path.exists(filepath):
                raise ValueError(f"Review {review_id} not found")
            
            # Load existing review
            with open(filepath, 'r') as f:
                review = json.load(f)
            
            # Update with expert decision
            review.update({
                "status": "completed",
                "assigned_expert": decision.get("expert_id"),
                "review_notes": decision.get("notes"),
                "final_decision": decision.get("verdict"),
                "completed_at": datetime.now().isoformat()
            })
            
            # Save updated review
            with open(filepath, 'w') as f:
                json.dump(review, f, indent=2)
            
            logger.info(f"Updated review {review_id} with expert decision")
            
        except Exception as e:
            logger.error(f"Failed to update review decision: {e}")
            raise

    def _calculate_verification_confidence(self, verifications: List[Verification]) -> float:
        """
        Calculate overall verification confidence.

        Args:
            verifications: List of verification results

        Returns:
            Overall confidence score
        """
        if not verifications:
            return 0.0

        # Calculate average confidence
        avg_confidence = sum(v.confidence for v in verifications) / len(verifications)

        # Boost confidence based on number of high-confidence verifications
        high_conf_verifications = [v for v in verifications if v.confidence > 0.8]
        high_conf_boost = min(0.1, len(high_conf_verifications) * 0.02)

        final_confidence = min(1.0, avg_confidence + high_conf_boost)

        return final_confidence


# Example usage
async def main():
    """Example usage of FactCheckAgent."""
    agent = FactCheckAgent()

    # Example documents and query
    documents = [
        {
            "doc_id": "doc1",
            "content": "The Earth orbits around the Sun. This is a well-established fact in astronomy.",
            "score": 0.9,
        },
        {
            "doc_id": "doc2",
            "content": "The Sun is a star located at the center of our solar system.",
            "score": 0.8,
        },
    ]

    task = {"documents": documents, "query": "What is the relationship between Earth and the Sun?"}

    context = QueryContext(query="What is the relationship between Earth and the Sun?")

    result = await agent.process_task(task, context)
    print(f"Success: {result.success}")
    print(f"Verified facts: {len(result.data.get('verified_facts', []))}")
    print(f"Confidence: {result.confidence}")


if __name__ == "__main__":
    asyncio.run(main())
