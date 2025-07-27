'use client'

import { useState } from 'react'
import QueryForm from '@/components/QueryForm'
import AnswerDisplay from '@/components/AnswerDisplay'
import FeedbackForm from '@/components/FeedbackForm'
import { QueryResponse } from '@/types/api'

export default function Home() {
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleQuerySubmit = async (query: string) => {
    setIsLoading(true)
    setError(null)
    setResponse(null)

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          max_tokens: 1000,
          confidence_threshold: 0.8,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResponse(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const handleFeedback = async (type: 'helpful' | 'not-helpful', details?: string) => {
    // TODO: Implement feedback submission to backend
    console.log('Feedback submitted:', { type, details })
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <div className="mb-6">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-full mb-4">
            <span className="text-white text-3xl">🧠</span>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Welcome to <span className="text-blue-600">SarvanOM</span>
          </h1>
          <p className="text-xl text-gray-600 mb-6">
            Your Own Knowledge Hub Powered by AI
          </p>
          <p className="text-gray-500 max-w-2xl mx-auto">
            Get accurate, verifiable answers with source citations and confidence scores. 
            Powered by advanced AI agents that search, verify, and synthesize information from multiple sources.
          </p>
        </div>
      </div>

      {/* Main Interface */}
      <div className="space-y-8">
        <QueryForm onSubmit={handleQuerySubmit} isLoading={isLoading} />
        
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <div className="mt-2 text-sm text-red-700">
                  <p>{error}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {response && (
          <>
            <AnswerDisplay 
              answer={response.answer}
              confidence={response.confidence}
              citations={response.citations}
              isLoading={false}
              queryId={response.query_id}
            />
            <FeedbackForm 
              queryId={response.query_id || 'unknown'}
              onFeedback={handleFeedback}
              disabled={isLoading}
            />
          </>
        )}
      </div>

      {/* Features Section */}
      <div className="mt-16 bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
          Why Choose SarvanOM?
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">AI-Powered Intelligence</h3>
            <p className="text-gray-600">
              Advanced multi-agent AI system that searches, verifies, and synthesizes information from multiple sources.
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Verified Sources</h3>
            <p className="text-gray-600">
              Every answer includes source citations and confidence scores for complete transparency and trust.
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Lightning Fast</h3>
            <p className="text-gray-600">
              Optimized for speed with intelligent caching and parallel processing for quick, accurate responses.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
