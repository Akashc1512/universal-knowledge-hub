'use client';

import React from 'react';
import { Citation } from '@/types/api';
import { ArrowUpRightIcon } from '@heroicons/react/24/outline';

interface CitationListProps {
  citations: Citation[];
}

export default function CitationList({ citations }: CitationListProps) {
  if (!citations || citations.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic">
        No citations available for this answer.
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-sm font-medium text-gray-900 mb-3">
        Sources ({citations.length})
      </h3>
      <div className="space-y-2">
        {citations.map((citation, index) => (
          <div key={citation.id || index} className="flex items-start space-x-2">
            <span className="text-xs font-medium text-gray-500 mt-1">
              [{index + 1}]
            </span>
            <div className="flex-1 text-sm text-gray-700">
              <div className="flex items-start justify-between">
                <span className="flex-1">{citation.text}</span>
                {citation.url && (
                  <a
                    href={citation.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-2 flex-shrink-0 text-blue-600 hover:text-blue-800 transition-colors"
                    title="Visit source"
                  >
                    <ArrowUpRightIcon className="h-4 w-4" />
                    <span className="sr-only">Visit source</span>
                  </a>
                )}
              </div>
              {citation.title && (
                <div className="text-xs text-gray-500 mt-1">
                  {citation.title}
                  {citation.author && ` - ${citation.author}`}
                  {citation.date && ` (${citation.date})`}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 