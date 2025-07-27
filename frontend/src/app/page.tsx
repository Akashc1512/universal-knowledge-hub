'use client';

import { useState } from 'react';
import { QueryResponse } from '@/types/api';
import api from '@/lib/api';
import QueryForm from '@/components/QueryForm';
import AnswerDisplay from '@/components/AnswerDisplay';
import FeedbackForm from '@/components/FeedbackForm';

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  const handleSubmitQuery = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.submitQuery(query);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'An error occurred while processing your question.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = async (type: 'helpful' | 'not-helpful', details?: string) => {
    if (result?.query_id) {
      try {
        await api.submitFeedback({
          query_id: result.query_id,
          feedback_type: type,
          details
        });
      } catch (error) {
        console.error('Failed to submit feedback:', error);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                Universal Knowledge Platform
              </h1>
            </div>
            <nav className="flex items-center space-x-4">
              <a
                href="/about"
                className="text-gray-600 hover:text-gray-900 transition-colors"
              >
                About
              </a>
              <a
                href="/help"
                className="text-gray-600 hover:text-gray-900 transition-colors"
              >
                Help
              </a>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Ask Anything, Get Accurate Answers
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Our advanced AI system uses multiple agents to provide comprehensive, 
            well-cited answers to your questions. Get reliable information with 
            confidence scores and source citations.
          </p>
        </div>

        {/* Query Form */}
        <div className="mb-8">
          <QueryForm
            onSubmit={handleSubmitQuery}
            isLoading={isLoading}
            placeholder="Ask any question..."
            maxLength={10000}
          />
        </div>

        {/* Answer Display */}
        {result && (
          <div className="mb-6">
            <AnswerDisplay
              answer={result.answer}
              confidence={result.confidence}
              citations={result.citations}
              isLoading={false}
              queryId={result.query_id}
            />
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mb-6">
            <AnswerDisplay
              answer=""
              confidence={0}
              citations={[]}
              isLoading={false}
              error={error}
            />
          </div>
        )}

        {/* Feedback Form */}
        {result && !isLoading && (
          <div className="mb-8">
            <FeedbackForm
              queryId={result.query_id || 'unknown'}
              onFeedback={handleFeedback}
              disabled={isLoading}
            />
          </div>
        )}

        {/* Features Section */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="bg-blue-100 rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
              <svg className="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Multi-Agent Intelligence
            </h3>
            <p className="text-gray-600">
              Our system uses specialized AI agents for retrieval, fact-checking, 
              synthesis, and citation to ensure accurate answers.
            </p>
          </div>

          <div className="text-center">
            <div className="bg-green-100 rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
              <svg className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Verified Sources
            </h3>
            <p className="text-gray-600">
              Every answer includes citations from reliable sources, 
              giving you confidence in the information provided.
            </p>
          </div>

          <div className="text-center">
            <div className="bg-purple-100 rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
              <svg className="h-8 w-8 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Fast & Reliable
            </h3>
            <p className="text-gray-600">
              Get comprehensive answers quickly with confidence scores 
              that help you understand the reliability of each response.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p>&copy; 2025 Universal Knowledge Platform. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
