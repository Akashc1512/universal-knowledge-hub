// API Types for Universal Knowledge Platform

export interface Citation {
  id: string;
  text: string;
  url?: string;
  title?: string;
  author?: string;
  date?: string;
  source?: string;
  confidence?: number;
}

export interface QueryRequest {
  query: string;
  max_tokens?: number;
  confidence_threshold?: number;
  user_context?: Record<string, unknown>;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  citations: Citation[];
  query_id: string;
  processing_time?: number;
  metadata?: Record<string, unknown>;
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

export interface FeedbackResponse {
  success: boolean;
  message: string;
  feedback_id?: string;
}

export interface AnalyticsData {
  total_queries: number;
  successful_queries: number;
  failed_queries: number;
  average_confidence: number;
  average_response_time?: number;
  cache_hit_rate?: number;
  top_queries: Array<{query: string; count: number}>;
  user_activity: Record<string, number>;
  time_period: Record<string, unknown>;
}

// CRUD Operation Types
export interface QueryUpdateRequest {
  query?: string;
  max_tokens?: number;
  confidence_threshold?: number;
  user_context?: Record<string, unknown>;
  reprocess?: boolean;
}

export interface QueryListResponse {
  queries: QuerySummary[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
}

export interface QuerySummary {
  query_id: string;
  query: string;
  status: 'completed' | 'processing' | 'failed';
  confidence: number;
  created_at: string;
  processing_time: number;
}

export interface QueryDetailResponse {
  query_id: string;
  query: string;
  answer: string;
  confidence: number;
  citations: Citation[];
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at?: string;
  processing_time: number;
  user_id: string;
  status: 'completed' | 'processing' | 'failed';
}

export interface QueryStatusResponse {
  query_id: string;
  status: 'completed' | 'processing' | 'failed';
  message?: string;
  progress?: number;
  estimated_completion?: string;
}

export interface QueryListFilters {
  page?: number;
  page_size?: number;
  status_filter?: 'completed' | 'processing' | 'failed';
  user_filter?: string;
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

// ============================================================================
// SYSTEM MONITORING AND ANALYTICS TYPES
// ============================================================================

export interface SystemMetrics {
  sarvanom_version: string;
  sarvanom_uptime_seconds: number;
  sarvanom_requests_total: number;
  sarvanom_errors_total: number;
  sarvanom_cache_hits_total: number;
  sarvanom_cache_misses_total: number;
  sarvanom_average_response_time_seconds: number;
  sarvanom_active_users: number;
  sarvanom_partial_failures_total: number;
  sarvanom_complete_failures_total: number;
  sarvanom_integration_vector_db_status: number;
  sarvanom_integration_elasticsearch_status: number;
  sarvanom_integration_knowledge_graph_status: number;
  sarvanom_integration_llm_api_status: number;
  [key: string]: any;
}

export interface IntegrationInfo {
  status: 'healthy' | 'unhealthy' | 'not_configured';
  last_check: string;
  last_success?: string;
  error_count: number;
  response_time?: number;
  version?: string;
  details?: Record<string, any>;
}

export interface IntegrationStatus {
  timestamp: number;
  integrations: Record<string, IntegrationInfo>;
  summary: {
    total: number;
    healthy: number;
    unhealthy: number;
    not_configured: number;
  };
}

export interface ApiInfo {
  name: string;
  version: string;
  status: string;
  uptime: number;
  environment: string;
  timestamp?: number;
} 