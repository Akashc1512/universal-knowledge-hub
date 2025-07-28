import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { QueryRequest, QueryResponse, ApiError, FeedbackRequest, AnalyticsData } from '@/types/api';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8002';

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'user-key-456',
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
  async submitQuery(query: string): Promise<QueryResponse> {
    try {
      const response = await apiClient.post<QueryResponse>('/query', {
        query: query.trim()
      });
      return response.data;
    } catch (error: any) {
      // Create an error that preserves the original response data
      const enhancedError = new Error(getErrorMessage(error)) as any;
      enhancedError.response = error.response;
      throw enhancedError;
    }
  },

  // Submit feedback for a query
  async submitFeedback(feedback: FeedbackRequest): Promise<void> {
    try {
      await apiClient.post('/feedback', feedback);
    } catch (error: any) {
      console.error('Feedback submission failed:', error);
      // Don't throw error for feedback - it's not critical
    }
  },

  // Get analytics data
  async getAnalytics(): Promise<AnalyticsData> {
    try {
      const response = await apiClient.get<AnalyticsData>('/analytics');
      return response.data;
    } catch (error: any) {
      throw new Error(getErrorMessage(error));
    }
  },

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      await apiClient.get('/health');
      return true;
    } catch (error) {
      return false;
    }
  }
};

// Error message helper
function getErrorMessage(error: any): string {
  if (error.response) {
    const status = error.response.status;
    // Backend returns error in 'error' field, not 'detail'
    const errorMessage = error.response.data?.error || error.response.data?.detail || 'Unknown error';
    
    switch (status) {
      case 400:
        return `Invalid request: ${errorMessage}`;
      case 401:
        return 'Authentication required. Please check your API key.';
      case 403:
        return 'Query blocked by content guidelines. Please rephrase your question.';
      case 408:
        return 'Query timeout. Please try a simpler question.';
      case 429:
        return 'Too many requests. Please wait a moment and try again.';
      case 500:
        return 'Server error. Please try again later.';
      case 503:
        return 'Service temporarily unavailable. Please try again later.';
      default:
        return `Error ${status}: ${errorMessage}`;
    }
  }
  
  if (error.code === 'ECONNABORTED') {
    return 'Request timeout. Please try again.';
  }
  
  if (error.code === 'NETWORK_ERROR') {
    return 'Network error. Please check your connection.';
  }
  
  return error.message || 'An unexpected error occurred.';
}

export default api; 