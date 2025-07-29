'use client';

import { useState, useRef, useEffect } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface QueryFormProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  placeholder?: string;
  maxLength?: number;
}

const exampleQuestions = [
  "What is the capital of France?",
  "How does photosynthesis work?",
  "What are the benefits of renewable energy?",
  "Explain quantum computing in simple terms"
];

export default function QueryForm({ 
  onSubmit, 
  isLoading, 
  placeholder = "Ask any question...",
  maxLength = 10000 
}: QueryFormProps) {
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-focus on mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const validateQuery = (text: string): string => {
    if (!text.trim()) {
      return 'Please enter a question.';
    }
    if (text.length > maxLength) {
      return `Question is too long. Maximum ${maxLength} characters allowed.`;
    }
    if (text.trim().length < 3) {
      return 'Question is too short. Please provide more detail.';
    }
    return '';
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const validationError = validateQuery(query);
    
    if (validationError) {
      setError(validationError);
      return;
    }

    setError('');
    onSubmit(query.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    setError('');
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  const characterCount = query.length;
  const isOverLimit = characterCount > maxLength;

  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-4" role="search">
        {/* Query Input */}
        <div className="relative">
          <label htmlFor="query-input" className="sr-only">
            Ask a question
          </label>
          <div className="relative">
            <textarea
              ref={inputRef}
              id="query-input"
              name="query"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                if (error) setError('');
              }}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={isLoading}
              aria-describedby={error ? 'query-error' : 'query-help'}
              aria-invalid={error ? 'true' : 'false'}
              className={`
                w-full px-4 py-3 pr-12 text-lg border-2 rounded-lg resize-none
                focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                transition-all duration-200
                ${error ? 'border-red-500 bg-red-50' : 'border-gray-300'}
                ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
                min-h-[60px] max-h-[200px]
              `}
              rows={3}
              maxLength={maxLength}
            />
            <div className="absolute right-3 top-3">
              {isLoading ? (
                <div 
                  className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"
                  aria-label="Processing your question"
                />
              ) : (
                <MagnifyingGlassIcon className="h-6 w-6 text-gray-400" aria-hidden="true" />
              )}
            </div>
          </div>

          {/* Character Count */}
          <div className="flex justify-between items-center mt-2 text-sm text-gray-500">
            <span id="query-help">
              {characterCount} / {maxLength} characters
            </span>
            {isOverLimit && (
              <span className="text-red-500 font-medium" role="alert">
                Over character limit
              </span>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div id="query-error" className="mt-2 text-sm text-red-600" role="alert">
              {error}
            </div>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading || !query.trim() || isOverLimit}
          className={`
            w-full sm:w-auto px-8 py-3 text-lg font-medium text-white rounded-lg
            transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2
            ${isLoading || !query.trim() || isOverLimit
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500'
            }
          `}
          aria-describedby={isLoading ? 'loading-description' : undefined}
        >
          {isLoading ? 'Analyzing...' : 'Ask Question'}
        </button>

        {isLoading && (
          <div id="loading-description" className="sr-only">
            Processing your question, please wait
          </div>
        )}
      </form>

      {/* Example Questions */}
      <div className="mt-8">
        <h3 className="text-lg font-medium text-gray-700 mb-3">
          Try these example questions:
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {exampleQuestions.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              disabled={isLoading}
              className="
                text-left p-3 text-sm text-gray-600 bg-gray-50 rounded-lg
                hover:bg-gray-100 hover:text-gray-800 transition-colors duration-200
                focus:outline-none focus:ring-2 focus:ring-blue-500
                disabled:opacity-50 disabled:cursor-not-allowed
              "
              aria-label={`Try example question: ${example}`}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
} 