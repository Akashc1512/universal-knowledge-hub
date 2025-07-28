'use client';

import { XCircleIcon, ExclamationTriangleIcon, ArrowPathIcon } from '@heroicons/react/24/outline';

interface ErrorDisplayProps {
  error: string;
  type?: 'error' | 'warning';
  onRetry?: () => void;
  requestId?: string;
}

export default function ErrorDisplay({ 
  error, 
  type = 'error',
  onRetry,
  requestId 
}: ErrorDisplayProps) {
  const isError = type === 'error';
  
  return (
    <div 
      className={`
        w-full max-w-4xl mx-auto rounded-lg p-6 
        ${isError ? 'bg-red-50 border border-red-200' : 'bg-yellow-50 border border-yellow-200'}
      `}
      role="alert"
      aria-live="assertive"
    >
      <div className="flex">
        <div className="flex-shrink-0">
          {isError ? (
            <XCircleIcon className="h-6 w-6 text-red-400" aria-hidden="true" />
          ) : (
            <ExclamationTriangleIcon className="h-6 w-6 text-yellow-400" aria-hidden="true" />
          )}
        </div>
        <div className="ml-3 flex-1">
          <h3 className={`text-sm font-medium ${isError ? 'text-red-800' : 'text-yellow-800'}`}>
            {isError ? 'Error processing your request' : 'Warning'}
          </h3>
          <div className={`mt-2 text-sm ${isError ? 'text-red-700' : 'text-yellow-700'}`}>
            <p>{error}</p>
            {requestId && (
              <p className="mt-1 text-xs opacity-75">
                Request ID: {requestId}
              </p>
            )}
          </div>
          {onRetry && (
            <div className="mt-4">
              <button
                type="button"
                onClick={onRetry}
                className={`
                  inline-flex items-center px-3 py-2 border border-transparent 
                  text-sm leading-4 font-medium rounded-md text-white 
                  ${isError 
                    ? 'bg-red-600 hover:bg-red-700 focus:ring-red-500' 
                    : 'bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-500'
                  }
                  focus:outline-none focus:ring-2 focus:ring-offset-2
                  transition-colors duration-200
                `}
              >
                <ArrowPathIcon className="h-4 w-4 mr-1.5" />
                Try Again
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 