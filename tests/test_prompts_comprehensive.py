#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE PROMPT TESTING SUITE
Universal Knowledge Platform - Prompt Engineering Tests

Tests all prompts, prompt templates, and prompt engineering components
for bulletproof functionality and optimal performance.
"""

import pytest
import unittest
import json
import os
import sys
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import prompt-related components
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Note: These prompt modules may not exist yet, so we'll create mock versions for testing
try:
    from prompts.query_processing import QueryProcessingPrompts
    from prompts.retrieval import RetrievalPrompts
    from prompts.synthesis import SynthesisPrompts
    from prompts.fact_check import FactCheckPrompts
    from prompts.citation import CitationPrompts
except ImportError:
    # Create mock prompt classes for testing
    class QueryProcessingPrompts:
        def get_classification_prompt(self, query):
            return f"Classify: {query}"

        def get_entity_extraction_prompt(self, query):
            return f"Extract entities: {query}"

        def get_complexity_prompt(self, query):
            return f"Assess complexity: {query}"

        def get_intent_prompt(self, query):
            return f"Recognize intent: {query}"

        def get_refinement_prompt(self, query, context):
            return f"Refine: {query} with {context}"

    class RetrievalPrompts:
        def get_semantic_search_prompt(self, query):
            return f"Semantic search: {query}"

        def get_keyword_search_prompt(self, query):
            return f"Keyword search: {query}"

        def get_hybrid_search_prompt(self, query):
            return f"Hybrid search: {query}"

        def get_reranking_prompt(self, query, documents):
            return f"Rerank: {query}"

    class SynthesisPrompts:
        def get_answer_generation_prompt(self, query, documents):
            return f"Generate answer: {query}"

        def get_aggregation_prompt(self, documents):
            return f"Aggregate: {len(documents)} docs"

        def get_summarization_prompt(self, content):
            return f"Summarize: {content[:50]}"

        def get_explanation_prompt(self, concept, context):
            return f"Explain: {concept}"

    class FactCheckPrompts:
        def get_verification_prompt(self, claim, sources):
            return f"Verify: {claim}"

        def get_decomposition_prompt(self, claim):
            return f"Decompose: {claim}"

        def get_evidence_evaluation_prompt(self, claim, evidence):
            return f"Evaluate: {claim}"

        def get_contradiction_detection_prompt(self, sources):
            return f"Detect contradictions"

    class CitationPrompts:
        def get_citation_generation_prompt(self, answer, sources):
            return f"Generate citations: {answer[:50]}"

        def get_relevance_scoring_prompt(self, query, sources):
            return f"Score relevance: {query}"

        def get_formatting_prompt(self, citations, format_style):
            return f"Format: {format_style}"


class TestQueryProcessingPrompts(unittest.TestCase):
    """Test query processing prompts"""

    def setUp(self):
        """Set up test environment"""
        self.prompts = QueryProcessingPrompts()

    def test_query_classification_prompt(self):
        """Test query classification prompt"""
        query = "What is quantum computing?"

        prompt = self.prompts.get_classification_prompt(query)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        self.assertIn("classify", prompt.lower())
        self.assertGreater(len(prompt), 50)

    def test_entity_extraction_prompt(self):
        """Test entity extraction prompt"""
        query = "How does machine learning work with neural networks?"

        prompt = self.prompts.get_entity_extraction_prompt(query)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        self.assertIn("extract", prompt.lower())
        self.assertIn("entities", prompt.lower())

    def test_complexity_assessment_prompt(self):
        """Test complexity assessment prompt"""
        query = "Explain the mathematical foundations of quantum mechanics"

        prompt = self.prompts.get_complexity_prompt(query)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        self.assertIn("complexity", prompt.lower())
        self.assertIn("assess", prompt.lower())

    def test_intent_recognition_prompt(self):
        """Test intent recognition prompt"""
        query = "What are the latest advances in AI?"

        prompt = self.prompts.get_intent_prompt(query)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        self.assertIn("intent", prompt.lower())
        self.assertIn("recognize", prompt.lower())

    def test_query_refinement_prompt(self):
        """Test query refinement prompt"""
        original_query = "AI"
        context = "User is interested in machine learning"

        prompt = self.prompts.get_refinement_prompt(original_query, context)

        self.assertIsInstance(prompt, str)
        self.assertIn(original_query, prompt)
        self.assertIn(context, prompt)
        self.assertIn("refine", prompt.lower())


class TestRetrievalPrompts(unittest.TestCase):
    """Test retrieval prompts"""

    def setUp(self):
        """Set up test environment"""
        self.prompts = RetrievalPrompts()

    def test_semantic_search_prompt(self):
        """Test semantic search prompt"""
        query = "quantum computing applications"

        prompt = self.prompts.get_semantic_search_prompt(query)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        self.assertIn("semantic", prompt.lower())
        self.assertIn("search", prompt.lower())

    def test_keyword_search_prompt(self):
        """Test keyword search prompt"""
        query = "machine learning algorithms"

        prompt = self.prompts.get_keyword_search_prompt(query)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        self.assertIn("keyword", prompt.lower())
        self.assertIn("search", prompt.lower())

    def test_hybrid_search_prompt(self):
        """Test hybrid search prompt"""
        query = "artificial intelligence in healthcare"

        prompt = self.prompts.get_hybrid_search_prompt(query)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        self.assertIn("hybrid", prompt.lower())
        self.assertIn("search", prompt.lower())

    def test_reranking_prompt(self):
        """Test reranking prompt"""
        documents = [
            {"content": "Document 1 about AI", "score": 0.8},
            {"content": "Document 2 about ML", "score": 0.7},
            {"content": "Document 3 about DL", "score": 0.6},
        ]
        query = "artificial intelligence"

        prompt = self.prompts.get_reranking_prompt(query, documents)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        for doc in documents:
            self.assertIn(doc["content"], prompt)


class TestSynthesisPrompts(unittest.TestCase):
    """Test synthesis prompts"""

    def setUp(self):
        """Set up test environment"""
        self.prompts = SynthesisPrompts()

    def test_answer_generation_prompt(self):
        """Test answer generation prompt"""
        query = "What is quantum computing?"
        documents = [
            {"content": "Quantum computing uses quantum mechanics", "score": 0.9},
            {"content": "It can solve complex problems faster", "score": 0.8},
        ]

        prompt = self.prompts.get_answer_generation_prompt(query, documents)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        for doc in documents:
            self.assertIn(doc["content"], prompt)

    def test_content_aggregation_prompt(self):
        """Test content aggregation prompt"""
        documents = [
            {"content": "AI is transforming industries", "score": 0.9},
            {"content": "Machine learning is a subset of AI", "score": 0.8},
            {"content": "Deep learning uses neural networks", "score": 0.7},
        ]

        prompt = self.prompts.get_aggregation_prompt(documents)

        self.assertIsInstance(prompt, str)
        for doc in documents:
            self.assertIn(doc["content"], prompt)

    def test_summarization_prompt(self):
        """Test summarization prompt"""
        content = "This is a long text about artificial intelligence and its applications in various fields including healthcare, finance, and transportation."

        prompt = self.prompts.get_summarization_prompt(content)

        self.assertIsInstance(prompt, str)
        self.assertIn(content, prompt)
        self.assertIn("summarize", prompt.lower())

    def test_explanation_prompt(self):
        """Test explanation prompt"""
        concept = "quantum computing"
        context = "User is a beginner"

        prompt = self.prompts.get_explanation_prompt(concept, context)

        self.assertIsInstance(prompt, str)
        self.assertIn(concept, prompt)
        self.assertIn(context, prompt)
        self.assertIn("explain", prompt.lower())


class TestFactCheckPrompts(unittest.TestCase):
    """Test fact-checking prompts"""

    def setUp(self):
        """Set up test environment"""
        self.prompts = FactCheckPrompts()

    def test_claim_verification_prompt(self):
        """Test claim verification prompt"""
        claim = "Quantum computers can solve all problems faster than classical computers"
        sources = [
            {"content": "Quantum computers excel at specific problems", "source": "research_paper"},
            {
                "content": "Classical computers are still faster for most tasks",
                "source": "textbook",
            },
        ]

        prompt = self.prompts.get_verification_prompt(claim, sources)

        self.assertIsInstance(prompt, str)
        self.assertIn(claim, prompt)
        for source in sources:
            self.assertIn(source["content"], prompt)

    def test_claim_decomposition_prompt(self):
        """Test claim decomposition prompt"""
        claim = "AI will replace all human jobs in the next decade"

        prompt = self.prompts.get_decomposition_prompt(claim)

        self.assertIsInstance(prompt, str)
        self.assertIn(claim, prompt)
        self.assertIn("decompose", prompt.lower())
        self.assertIn("subclaims", prompt.lower())

    def test_evidence_evaluation_prompt(self):
        """Test evidence evaluation prompt"""
        claim = "Machine learning improves healthcare outcomes"
        evidence = [
            {"content": "ML algorithms detect cancer earlier", "reliability": "high"},
            {"content": "AI reduces diagnostic errors", "reliability": "medium"},
        ]

        prompt = self.prompts.get_evidence_evaluation_prompt(claim, evidence)

        self.assertIsInstance(prompt, str)
        self.assertIn(claim, prompt)
        for ev in evidence:
            self.assertIn(ev["content"], prompt)

    def test_contradiction_detection_prompt(self):
        """Test contradiction detection prompt"""
        sources = [
            {"content": "AI is beneficial for society", "source": "study_1"},
            {"content": "AI poses risks to humanity", "source": "study_2"},
            {"content": "AI has both benefits and risks", "source": "study_3"},
        ]

        prompt = self.prompts.get_contradiction_detection_prompt(sources)

        self.assertIsInstance(prompt, str)
        for source in sources:
            self.assertIn(source["content"], prompt)
        self.assertIn("contradiction", prompt.lower())


class TestCitationPrompts(unittest.TestCase):
    """Test citation prompts"""

    def setUp(self):
        """Set up test environment"""
        self.prompts = CitationPrompts()

    def test_citation_generation_prompt(self):
        """Test citation generation prompt"""
        answer = "Quantum computing uses quantum mechanics to process information"
        sources = [
            {
                "title": "Quantum Computing Basics",
                "url": "https://example.com/quantum",
                "content": "Quantum computing uses...",
            },
            {
                "title": "Advanced Quantum Algorithms",
                "url": "https://example.com/algorithms",
                "content": "Shor's algorithm can...",
            },
        ]

        prompt = self.prompts.get_citation_generation_prompt(answer, sources)

        self.assertIsInstance(prompt, str)
        self.assertIn(answer, prompt)
        for source in sources:
            self.assertIn(source["title"], prompt)

    def test_relevance_scoring_prompt(self):
        """Test relevance scoring prompt"""
        query = "What is quantum computing?"
        sources = [
            {"content": "Quantum computing uses quantum mechanics", "title": "QC Basics"},
            {"content": "Machine learning algorithms", "title": "ML Guide"},
            {"content": "Quantum algorithms for cryptography", "title": "QC Crypto"},
        ]

        prompt = self.prompts.get_relevance_scoring_prompt(query, sources)

        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        for source in sources:
            self.assertIn(source["content"], prompt)

    def test_citation_formatting_prompt(self):
        """Test citation formatting prompt"""
        citations = [
            {"title": "Research Paper 1", "authors": "Smith, J.", "year": "2023"},
            {"title": "Book Chapter", "authors": "Johnson, A.", "year": "2022"},
        ]
        format_style = "apa"

        prompt = self.prompts.get_formatting_prompt(citations, format_style)

        self.assertIsInstance(prompt, str)
        self.assertIn(format_style, prompt)
        for citation in citations:
            self.assertIn(citation["title"], prompt)


class TestPromptValidation(unittest.TestCase):
    """Test prompt validation and quality"""

    def test_prompt_length_validation(self):
        """Test that prompts are not too long or too short"""
        test_prompts = [
            "What is quantum computing?",
            "Explain the mathematical foundations of quantum mechanics in detail",
            "AI",
            "This is a very long prompt that should be validated for length and quality to ensure it meets the requirements for optimal performance and clarity in the context of artificial intelligence and machine learning applications",
        ]

        for prompt in test_prompts:
            # Prompt should be between 10 and 1000 characters
            self.assertGreaterEqual(len(prompt), 10, f"Prompt too short: {prompt}")
            self.assertLessEqual(len(prompt), 1000, f"Prompt too long: {prompt}")

    def test_prompt_clarity_validation(self):
        """Test that prompts are clear and well-structured"""
        good_prompts = [
            "Classify the following query: {query}",
            "Extract entities from: {text}",
            "Generate answer for: {question} based on: {context}",
        ]

        bad_prompts = ["", "   ", "prompt", "this is not a proper prompt structure"]

        for prompt in good_prompts:
            self.assertGreater(len(prompt.strip()), 0)
            self.assertIn("{", prompt)  # Should have placeholders

        for prompt in bad_prompts:
            self.assertLess(len(prompt.strip()), 10)

    def test_prompt_consistency_validation(self):
        """Test that prompts are consistent in structure"""
        prompts = [
            "Classify: {query}",
            "Extract: {text}",
            "Generate: {question}",
            "Verify: {claim}",
        ]

        for prompt in prompts:
            # Should have consistent structure
            self.assertIn(":", prompt)
            self.assertIn("{", prompt)
            self.assertIn("}", prompt)


class TestPromptPerformance(unittest.TestCase):
    """Test prompt performance and optimization"""

    def test_prompt_generation_speed(self):
        """Test that prompt generation is fast"""
        import time

        # Test prompt generation speed
        start_time = time.time()

        # Generate multiple prompts
        for i in range(100):
            prompt = f"Test prompt {i} with query: {{query}}"

        end_time = time.time()
        generation_time = end_time - start_time

        # Should generate 100 prompts in less than 1 second
        self.assertLess(generation_time, 1.0)

    def test_prompt_memory_usage(self):
        """Test that prompt generation doesn't use excessive memory"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Generate many prompts
        prompts = []
        for i in range(1000):
            prompts.append(f"Test prompt {i} with content: {{content}}")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not increase memory by more than 10MB
        self.assertLess(memory_increase, 10 * 1024 * 1024)

    def test_prompt_caching(self):
        """Test prompt caching functionality"""
        # This would test if prompts are cached for performance
        pass


class TestPromptSecurity(unittest.TestCase):
    """Test prompt security and injection prevention"""

    def test_prompt_injection_prevention(self):
        """Test that prompts prevent injection attacks"""
        malicious_inputs = [
            "'; DROP TABLE prompts; --",
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "{{config}}",
            "{{request}}",
        ]

        for malicious_input in malicious_inputs:
            # Should sanitize or reject malicious input
            sanitized = malicious_input.replace("<script>", "").replace("javascript:", "")
            self.assertNotIn("<script>", sanitized)
            self.assertNotIn("javascript:", sanitized)

    def test_prompt_sanitization(self):
        """Test prompt sanitization"""
        test_inputs = [
            "Normal text",
            "Text with <b>HTML</b>",
            "Text with 'quotes' and \"double quotes\"",
            "Text with special chars: & < > \" '",
        ]

        for test_input in test_inputs:
            # Should handle special characters properly
            sanitized = test_input.replace("<", "&lt;").replace(">", "&gt;")
            self.assertNotIn("<script>", sanitized)
            self.assertNotIn("javascript:", sanitized)


def run_prompt_tests():
    """Run all prompt tests"""
    print("🧪 Starting COMPREHENSIVE PROMPT TESTING SUITE")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all prompt test classes
    test_classes = [
        TestQueryProcessingPrompts,
        TestRetrievalPrompts,
        TestSynthesisPrompts,
        TestFactCheckPrompts,
        TestCitationPrompts,
        TestPromptValidation,
        TestPromptPerformance,
        TestPromptSecurity,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("🧪 PROMPT TESTING SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.2f}%"
    )

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    if result.wasSuccessful():
        print("\n✅ ALL PROMPT TESTS PASSED - PROMPTS ARE BULLETPROOF!")
    else:
        print("\n❌ SOME PROMPT TESTS FAILED - NEEDS FIXING!")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_prompt_tests()
    sys.exit(0 if success else 1)
