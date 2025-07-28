import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { 
  QueryRequest, 
  QueryResponse, 
  QueryUpdateRequest,
  QueryListResponse,
  QueryDetailResponse,
  QueryStatusResponse,
  QueryListFilters,
  FeedbackRequest, 
  FeedbackResponse, 
  AnalyticsData 
} from '@/types/api';

// API Configuration from environment
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8002';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || 'user-key-456';
const DEFAULT_MAX_TOKENS = parseInt(process.env.NEXT_PUBLIC_DEFAULT_MAX_TOKENS || '1000');
const DEFAULT_CONFIDENCE_THRESHOLD = parseFloat(process.env.NEXT_PUBLIC_DEFAULT_CONFIDENCE_THRESHOLD || '0.8');
const REQUEST_TIMEOUT = parseInt(process.env.NEXT_PUBLIC_REQUEST_TIMEOUT || '30000');

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API Functions
export const api = {
  // Submit a query to the backend
  async submitQuery(query: string, options?: Partial<QueryRequest>): Promise<QueryResponse> {
    try {
      const requestData: QueryRequest = {
        query: query.trim(),
        max_tokens: options?.max_tokens || DEFAULT_MAX_TOKENS,
        confidence_threshold: options?.confidence_threshold || DEFAULT_CONFIDENCE_THRESHOLD,
        user_context: options?.user_context || {}
      };

      const response = await apiClient.post<QueryResponse>('/query', requestData);
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Submit feedback for a query
  async submitFeedback(feedback: FeedbackRequest): Promise<FeedbackResponse> {
    try {
      const response = await apiClient.post<FeedbackResponse>('/feedback', feedback);
      return response.data;
    } catch (error: unknown) {
      console.error('Feedback submission failed:', error);
      throw new Error(getErrorMessage(error));
    }
  },

  // Get analytics data
  async getAnalytics(): Promise<AnalyticsData> {
    try {
      const response = await apiClient.get<AnalyticsData>('/analytics');
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      await apiClient.get('/health');
      return true;
    } catch {
      return false;
    }
  },

  // ============================================================================
  // CRUD OPERATIONS FOR QUERIES
  // ============================================================================

  // List queries with pagination and filtering
  async listQueries(filters?: QueryListFilters): Promise<QueryListResponse> {
    try {
      const params = new URLSearchParams();
      
      if (filters?.page) params.append('page', filters.page.toString());
      if (filters?.page_size) params.append('page_size', filters.page_size.toString());
      if (filters?.status_filter) params.append('status_filter', filters.status_filter);
      if (filters?.user_filter) params.append('user_filter', filters.user_filter);

      const url = `/queries${params.toString() ? '?' + params.toString() : ''}`;
      const response = await apiClient.get<QueryListResponse>(url);
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Get detailed information about a specific query
  async getQuery(queryId: string): Promise<QueryDetailResponse> {
    try {
      const response = await apiClient.get<QueryDetailResponse>(`/queries/${queryId}`);
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Update an existing query
  async updateQuery(queryId: string, updateData: QueryUpdateRequest): Promise<QueryDetailResponse> {
    try {
      const response = await apiClient.put<QueryDetailResponse>(`/queries/${queryId}`, updateData);
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Delete a query
  async deleteQuery(queryId: string): Promise<{ message: string; query_id: string }> {
    try {
      const response = await apiClient.delete<{ message: string; query_id: string }>(`/queries/${queryId}`);
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Get query status
  async getQueryStatus(queryId: string): Promise<QueryStatusResponse> {
    try {
      const response = await apiClient.get<QueryStatusResponse>(`/queries/${queryId}/status`);
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Reprocess a query
  async reprocessQuery(queryId: string): Promise<{ message: string; query_id: string; processing_time: number; new_confidence: number }> {
    try {
      const response = await apiClient.patch<{ message: string; query_id: string; processing_time: number; new_confidence: number }>(`/queries/${queryId}/reprocess`);
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // ============================================================================
  // SYSTEM MONITORING AND ANALYTICS
  // ============================================================================

  // Get system metrics
  async getMetrics(): Promise<Record<string, any>> {
    try {
      const response = await apiClient.get<Record<string, any>>('/metrics');
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Get integration status
  async getIntegrationStatus(): Promise<{
    timestamp: number;
    integrations: Record<string, any>;
    summary: {
      total: number;
      healthy: number;
      unhealthy: number;
      not_configured: number;
    };
  }> {
    try {
      const response = await apiClient.get('/integrations');
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Get root API information
  async getApiInfo(): Promise<{
    name: string;
    version: string;
    status: string;
    uptime: number;
    environment: string;
  }> {
    try {
      const response = await apiClient.get('/');
      return response.data;
    } catch (error: unknown) {
      throw new Error(getErrorMessage(error));
    }
  }
};

// Error message helper
function getErrorMessage(error: unknown): string {
  if (error && typeof error === 'object' && 'response' in error) {
    const axiosError = error as { response?: { status: number; data?: { detail?: string } } };
    
    if (axiosError.response) {
      const status = axiosError.response.status;
      const detail = axiosError.response.data?.detail || 'Unknown error';
      
      switch (status) {
        case 400:
          return `Invalid request: ${detail}`;
        case 403:
          return 'Query blocked by content guidelines. Please rephrase your question.';
        case 408:
          return 'Query timeout. Please try a simpler question.';
        case 429:
          return 'Too many requests. Please wait a moment and try again.';
        case 500:
          return 'Server error. Please try again later.';
        default:
          return `Error ${status}: ${detail}`;
      }
    }
  }
  
  if (error && typeof error === 'object' && 'code' in error) {
    const codeError = error as { code: string };
    
    if (codeError.code === 'ECONNABORTED') {
      return 'Request timeout. Please try again.';
    }
    
    if (codeError.code === 'NETWORK_ERROR') {
      return 'Network error. Please check your connection.';
    }
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  return 'An unexpected error occurred.';
}

export default api; 