// API Types for Universal Knowledge Platform

export interface Citation {
  id: string;
  text: string;
  url?: string;
  title?: string;
  author?: string;
  date?: string;
}

export interface QueryRequest {
  query: string;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  citations: Citation[];
  query_id?: string;
  processing_time?: number;
}

export interface ApiError {
  detail: string;
  status_code: number;
}

export interface FeedbackRequest {
  query_id: string;
  feedback_type: 'helpful' | 'not-helpful';
  details?: string;
}

export interface AnalyticsData {
  total_queries: number;
  average_response_time: number;
  cache_hit_rate: number;
  top_queries: string[];
}

export interface User {
  id: string;
  email: string;
  role: 'user' | 'expert' | 'admin';
  created_at: string;
}

export interface ExpertValidationTask {
  id: string;
  query: string;
  answer: string;
  confidence: number;
  citations: Citation[];
  status: 'pending' | 'approved' | 'rejected' | 'corrected';
  submitted_at: string;
  reviewed_at?: string;
  reviewer_id?: string;
} 