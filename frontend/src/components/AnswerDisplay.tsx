'use client';

import { QueryResponse, Citation } from '@/types/api';
import CitationList from './CitationList';
import ConfidenceBadge from './ConfidenceBadge';

interface AnswerDisplayProps {
  answer: string;
  confidence: number;
  citations: Citation[];
  isLoading: boolean;
  error?: string;
  queryId?: string;
}

export default function AnswerDisplay({
  answer,
  confidence,
  citations,
  isLoading,
  error,
  queryId
}: AnswerDisplayProps) {
  if (isLoading) {
    return (
      <div className="w-full max-w-4xl mx-auto" aria-live="polite" aria-busy="true">
        <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
          <div className="animate-pulse space-y-4">
            <div className="flex items-center space-x-3">
              <div className="w-4 h-4 bg-blue-200 rounded-full animate-pulse" />
              <div className="h-4 bg-gray-200 rounded w-32" />
            </div>
            <div className="space-y-3">
              <div className="h-4 bg-gray-200 rounded w-full" />
              <div className="h-4 bg-gray-200 rounded w-5/6" />
              <div className="h-4 bg-gray-200 rounded w-4/6" />
            </div>
            <div className="h-4 bg-gray-200 rounded w-1/3" />
          </div>
          <div className="mt-4 text-sm text-gray-500">
            Analyzing your question...
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full max-w-4xl mx-auto" role="alert">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                Unable to process your question
              </h3>
              <div className="mt-2 text-sm text-red-700">
                {error}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!answer) {
    return null;
  }

  return (
    <div className="w-full max-w-4xl mx-auto" aria-live="polite">
      <article className="bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
        {/* Answer Header */}
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">
              Answer
            </h2>
            <ConfidenceBadge confidence={confidence} />
          </div>
        </div>

        {/* Answer Content */}
        <div className="px-6 py-6">
          <div className="prose prose-lg max-w-none">
            <div 
              className="text-gray-800 leading-relaxed"
              dangerouslySetInnerHTML={{ 
                __html: formatAnswerWithCitations(answer, citations) 
              }}
            />
          </div>
        </div>

        {/* Citations */}
        {citations && citations.length > 0 && (
          <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
            <CitationList citations={citations} />
          </div>
        )}

        {/* Processing Info */}
        {queryId && (
          <div className="px-6 py-2 bg-gray-50 border-t border-gray-200">
            <p className="text-xs text-gray-500">
              Query ID: {queryId}
            </p>
          </div>
        )}
      </article>
    </div>
  );
}

// Helper function to format answer with citation markers
function formatAnswerWithCitations(answer: string, citations: Citation[]): string {
  if (!citations || citations.length === 0) {
    return answer;
  }

  // Simple formatting - in a real implementation, you might want more sophisticated parsing
  let formattedAnswer = answer;
  
  // Add citation markers if they don't exist
  citations.forEach((citation, index) => {
    const marker = `[${index + 1}]`;
    // This is a simple implementation - in practice, you'd want more sophisticated parsing
    // based on how the backend formats citations
    if (!formattedAnswer.includes(marker)) {
      // Add marker at the end of sentences that might need citations
      formattedAnswer = formattedAnswer.replace(/\./g, `${marker}.`);
    }
  });

  return formattedAnswer;
} 