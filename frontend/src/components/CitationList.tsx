'use client';

import React from 'react';
import { Citation } from '@/types/api';
import { ArrowTopRightOnSquareIcon, DocumentTextIcon } from '@heroicons/react/24/outline';

interface CitationListProps {
  citations: Citation[];
}

export default function CitationList({ citations }: CitationListProps) {
  if (!citations || citations.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic flex items-center space-x-2">
        <DocumentTextIcon className="h-4 w-4" />
        <span>No citations available for this answer.</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-2">
        <DocumentTextIcon className="h-5 w-5 text-gray-600" />
        <h3 className="text-sm font-semibold text-gray-900">
          Sources ({citations.length})
        </h3>
      </div>
      
      <div className="space-y-3">
        {citations.map((citation, index) => (
          <div 
            key={citation.id || index} 
            className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition-colors"
          >
            <span 
              className="text-xs font-bold text-gray-600 bg-white rounded-full w-6 h-6 flex items-center justify-center border border-gray-300 flex-shrink-0 mt-0.5"
              aria-label={`Citation ${index + 1}`}
            >
              {index + 1}
            </span>
            
            <div className="flex-1 min-w-0">
              <div className="text-sm text-gray-800 leading-relaxed">
                {citation.text}
              </div>
              
              {(citation.title || citation.author || citation.date) && (
                <div className="mt-2 text-xs text-gray-600 space-y-1">
                  {citation.title && (
                    <div className="font-medium text-gray-700">
                      {citation.title}
                    </div>
                  )}
                  {(citation.author || citation.date) && (
                    <div className="text-gray-500">
                      {citation.author && <span>{citation.author}</span>}
                      {citation.author && citation.date && <span> • </span>}
                      {citation.date && <span>{citation.date}</span>}
                    </div>
                  )}
                </div>
              )}
            </div>
            
            {citation.url && (
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-shrink-0 text-blue-600 hover:text-blue-800 transition-colors p-1 rounded"
                title="Visit source"
                aria-label={`Visit source for citation ${index + 1}`}
              >
                <ArrowTopRightOnSquareIcon className="h-4 w-4" />
                <span className="sr-only">Visit source</span>
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  );
} 