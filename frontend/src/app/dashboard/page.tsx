'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { api } from '@/lib/api'
import { 
  SystemMetrics, 
  IntegrationStatus, 
  ApiInfo, 
  AnalyticsData 
} from '@/types/api'

interface ErrorResponse {
  error: string;
}

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<SystemMetrics | ErrorResponse | null>(null)
  const [integrations, setIntegrations] = useState<IntegrationStatus | ErrorResponse | null>(null)
  const [apiInfo, setApiInfo] = useState<ApiInfo | null>(null)
  const [analytics, setAnalytics] = useState<AnalyticsData | ErrorResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())

  // Load all dashboard data
  useEffect(() => {
    loadDashboardData()
    
    // Set up auto-refresh every 30 seconds
    const interval = setInterval(loadDashboardData, 30000)
    return () => clearInterval(interval)
  }, [])

  const loadDashboardData = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      // Load all data in parallel
      const [metricsData, integrationsData, apiInfoData, analyticsData] = await Promise.all([
        api.getMetrics().catch(err => ({ error: err.message })),
        api.getIntegrationStatus().catch(err => ({ error: err.message })),
        api.getApiInfo().catch(err => ({ error: err.message })),
        api.getAnalytics().catch(err => ({ error: err.message }))
      ])

      setMetrics(metricsData as SystemMetrics | ErrorResponse)
      setIntegrations(integrationsData as IntegrationStatus | ErrorResponse)
      setApiInfo(apiInfoData as ApiInfo)
      setAnalytics(analyticsData as AnalyticsData | ErrorResponse)
      setLastRefresh(new Date())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data')
    } finally {
      setIsLoading(false)
    }
  }

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100'
      case 'unhealthy': return 'text-red-600 bg-red-100'
      case 'not_configured': return 'text-yellow-600 bg-yellow-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Navigation */}
      <div className="mb-8">
        <nav className="flex space-x-4">
          <Link 
            href="/" 
            className="px-4 py-2 border border-blue-600 text-blue-600 rounded-md hover:bg-blue-50"
          >
            New Query
          </Link>
          <Link 
            href="/queries" 
            className="px-4 py-2 border border-blue-600 text-blue-600 rounded-md hover:bg-blue-50"
          >
            Manage Queries
          </Link>
          <Link 
            href="/dashboard" 
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Dashboard
          </Link>
        </nav>
      </div>

      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">System Dashboard</h1>
            <p className="text-gray-600">Monitor system health, performance, and integrations</p>
          </div>
          <div className="text-right">
            <button
              onClick={loadDashboardData}
              disabled={isLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? 'Refreshing...' : 'Refresh'}
            </button>
            <p className="text-sm text-gray-500 mt-1">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <div className="mt-2 text-sm text-red-700">
                <p>{error}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* API Information */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">API Information</h2>
          {apiInfo ? (
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Name:</span>
                <span className="font-medium">{apiInfo.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Version:</span>
                <span className="font-medium">{apiInfo.version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Status:</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(apiInfo.status)}`}>
                  {apiInfo.status}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Uptime:</span>
                <span className="font-medium">{formatUptime(apiInfo.uptime)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Environment:</span>
                <span className="font-medium">{apiInfo.environment}</span>
              </div>
            </div>
          ) : (
            <div className="text-gray-500">Loading API information...</div>
          )}
        </div>

        {/* System Metrics */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">System Metrics</h2>
          {metrics && !('error' in metrics) ? (
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Requests:</span>
                <span className="font-medium">{metrics.sarvanom_requests_total || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Errors:</span>
                <span className="font-medium text-red-600">{metrics.sarvanom_errors_total || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Cache Hits:</span>
                <span className="font-medium text-green-600">{metrics.sarvanom_cache_hits_total || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Cache Misses:</span>
                <span className="font-medium text-orange-600">{metrics.sarvanom_cache_misses_total || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Response Time:</span>
                <span className="font-medium">{(metrics.sarvanom_average_response_time_seconds || 0).toFixed(3)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Active Users:</span>
                <span className="font-medium">{metrics.sarvanom_active_users || 0}</span>
              </div>
            </div>
          ) : (
            <div className="text-gray-500">
              {'error' in (metrics || {}) ? `Error: ${(metrics as ErrorResponse).error}` : 'Loading metrics...'}
            </div>
          )}
        </div>

        {/* Analytics */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Analytics</h2>
          {analytics && !('error' in analytics) ? (
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Queries:</span>
                <span className="font-medium">{analytics.total_queries}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Successful:</span>
                <span className="font-medium text-green-600">{analytics.successful_queries}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Failed:</span>
                <span className="font-medium text-red-600">{analytics.failed_queries}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Confidence:</span>
                <span className="font-medium">{(analytics.average_confidence * 100).toFixed(1)}%</span>
              </div>
              {analytics.cache_hit_rate !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Cache Hit Rate:</span>
                  <span className="font-medium">{(analytics.cache_hit_rate * 100).toFixed(1)}%</span>
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-500">
              {analytics?.error ? `Error: ${analytics.error}` : 'Loading analytics...'}
            </div>
          )}
        </div>

        {/* Integration Status */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Integration Status</h2>
          {integrations && !integrations.error ? (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4 text-center mb-4">
                <div>
                  <div className="text-2xl font-bold text-green-600">{integrations.summary.healthy}</div>
                  <div className="text-sm text-gray-600">Healthy</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-red-600">{integrations.summary.unhealthy}</div>
                  <div className="text-sm text-gray-600">Unhealthy</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-yellow-600">{integrations.summary.not_configured}</div>
                  <div className="text-sm text-gray-600">Not Configured</div>
                </div>
              </div>
              
              <div className="space-y-2">
                {Object.entries(integrations.integrations).map(([name, info]) => (
                  <div key={name} className="flex justify-between items-center p-2 border rounded">
                    <span className="font-medium capitalize">{name.replace('_', ' ')}</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(info.status)}`}>
                      {info.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-gray-500">
              {integrations?.error ? `Error: ${integrations.error}` : 'Loading integration status...'}
            </div>
          )}
        </div>

      </div>
    </div>
  )
} 