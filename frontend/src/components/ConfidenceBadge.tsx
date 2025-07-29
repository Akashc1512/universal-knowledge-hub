'use client';

interface ConfidenceBadgeProps {
  confidence: number;
}

export default function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'bg-green-100 text-green-800 border-green-200';
    if (conf >= 0.6) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  const getConfidenceLabel = (conf: number) => {
    if (conf >= 0.8) return 'High';
    if (conf >= 0.6) return 'Medium';
    return 'Low';
  };

  const getConfidenceDescription = (conf: number) => {
    if (conf >= 0.8) return 'Very confident in this answer';
    if (conf >= 0.6) return 'Moderately confident in this answer';
    return 'Low confidence in this answer';
  };

  const percentage = Math.round(confidence * 100);

  return (
    <div className="flex items-center space-x-2" role="status" aria-live="polite">
      <span className="text-sm text-gray-500 sr-only">Confidence level:</span>
      <span
        className={`
          inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border
          ${getConfidenceColor(confidence)}
        `}
        title={getConfidenceDescription(confidence)}
        aria-label={`Confidence: ${getConfidenceLabel(confidence)} (${percentage}%)`}
      >
        {getConfidenceLabel(confidence)} ({percentage}%)
      </span>
    </div>
  );
} 