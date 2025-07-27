'use client';

import { useState } from 'react';
import { HandThumbUpIcon, HandThumbDownIcon } from '@heroicons/react/24/outline';

interface FeedbackFormProps {
  queryId: string;
  onFeedback: (type: 'helpful' | 'not-helpful', details?: string) => void;
  disabled?: boolean;
}

export default function FeedbackForm({ queryId, onFeedback, disabled = false }: FeedbackFormProps) {
  const [feedbackType, setFeedbackType] = useState<'helpful' | 'not-helpful' | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [details, setDetails] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleFeedback = (type: 'helpful' | 'not-helpful') => {
    setFeedbackType(type);
    
    if (type === 'not-helpful') {
      setShowDetails(true);
    } else {
      submitFeedback(type);
    }
  };

  const submitFeedback = (type: 'helpful' | 'not-helpful') => {
    onFeedback(type, details);
    setSubmitted(true);
  };

  const handleDetailsSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (feedbackType) {
      submitFeedback(feedbackType);
    }
  };

  if (submitted) {
    return (
      <div className="w-full max-w-4xl mx-auto">
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-green-800">
                Thank you for your feedback!
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-gray-900">
              Was this answer helpful?
            </h3>
            <p className="text-sm text-gray-500 mt-1">
              Your feedback helps us improve our responses.
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              type="button"
              onClick={() => handleFeedback('helpful')}
              disabled={disabled}
              className={`
                inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md
                transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2
                ${disabled
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-green-100 text-green-700 hover:bg-green-200 focus:ring-green-500'
                }
              `}
              aria-label="Mark answer as helpful"
            >
              <HandThumbUpIcon className="h-4 w-4 mr-1" />
              Helpful
            </button>
            
            <button
              type="button"
              onClick={() => handleFeedback('not-helpful')}
              disabled={disabled}
              className={`
                inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md
                transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2
                ${disabled
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-red-100 text-red-700 hover:bg-red-200 focus:ring-red-500'
                }
              `}
              aria-label="Mark answer as not helpful"
            >
              <HandThumbDownIcon className="h-4 w-4 mr-1" />
              Not Helpful
            </button>
          </div>
        </div>

        {/* Details Form */}
        {showDetails && (
          <form onSubmit={handleDetailsSubmit} className="mt-4">
            <div>
              <label htmlFor="feedback-details" className="block text-sm font-medium text-gray-700">
                What could we improve? (Optional)
              </label>
              <textarea
                id="feedback-details"
                value={details}
                onChange={(e) => setDetails(e.target.value)}
                rows={3}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="Please share any specific details about what was wrong or how we could improve..."
              />
            </div>
            <div className="mt-3 flex items-center space-x-3">
              <button
                type="submit"
                className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Submit Feedback
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowDetails(false);
                  setFeedbackType(null);
                  setDetails('');
                }}
                className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Cancel
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
} 