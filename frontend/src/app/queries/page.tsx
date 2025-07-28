'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { api } from '@/lib/api'
import { 
  QueryListResponse, 
  QueryDetailResponse, 
  QuerySummary, 
  QueryUpdateRequest,
  QueryListFilters 
} from '@/types/api'

export default function QueriesPage() {
  const [queries, setQueries] = useState<QueryListResponse | null>(null)
  const [selectedQuery, setSelectedQuery] = useState<QueryDetailResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showEditModal, setShowEditModal] = useState(false)
  const [editData, setEditData] = useState<QueryUpdateRequest>({})
  const [filters, setFilters] = useState<QueryListFilters>({ page: 1, page_size: 10 })
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  // Load queries on component mount and when filters change
  useEffect(() => {
    loadQueries()
  }, [filters, refreshTrigger])

  const loadQueries = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const data = await api.listQueries(filters)
      setQueries(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load queries')
    } finally {
      setIsLoading(false)
    }
  }

  const loadQueryDetails = async (queryId: string) => {
    try {
      const data = await api.getQuery(queryId)
      setSelectedQuery(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load query details')
    }
  }

  const handleUpdateQuery = async () => {
    if (!selectedQuery) return

    try {
      const updatedQuery = await api.updateQuery(selectedQuery.query_id, editData)
      setSelectedQuery(updatedQuery)
      setShowEditModal(false)
      setEditData({})
      setRefreshTrigger(prev => prev + 1) // Refresh the list
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update query')
    }
  }

  const handleDeleteQuery = async (queryId: string) => {
    if (!confirm('Are you sure you want to delete this query?')) return

    try {
      await api.deleteQuery(queryId)
      setRefreshTrigger(prev => prev + 1) // Refresh the list
      if (selectedQuery?.query_id === queryId) {
        setSelectedQuery(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete query')
    }
  }

  const handleReprocessQuery = async (queryId: string) => {
    try {
      await api.reprocessQuery(queryId)
      // Reload query details to show updated results
      if (selectedQuery?.query_id === queryId) {
        await loadQueryDetails(queryId)
      }
      setRefreshTrigger(prev => prev + 1) // Refresh the list
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reprocess query')
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100'
      case 'processing': return 'text-yellow-600 bg-yellow-100'
      case 'failed': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Navigation */}
      <div className="mb-6">
        <nav className="flex space-x-4">
          <Link 
            href="/" 
            className="px-4 py-2 border border-blue-600 text-blue-600 rounded-md hover:bg-blue-50"
          >
            New Query
          </Link>
          <Link 
            href="/queries" 
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Manage Queries
          </Link>
          <Link 
            href="/dashboard" 
            className="px-4 py-2 border border-blue-600 text-blue-600 rounded-md hover:bg-blue-50"
          >
            Dashboard
          </Link>
        </nav>
      </div>

      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Query Management</h1>
        <p className="text-gray-600">Manage and review your knowledge queries with full CRUD operations</p>
      </div>

      {/* Filters */}
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <div className="flex flex-wrap gap-4 items-center">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Status Filter</label>
            <select
              value={filters.status_filter || ''}
              onChange={(e) => setFilters({ ...filters, status_filter: e.target.value as 'completed' | 'processing' | 'failed' | undefined, page: 1 })}
              className="border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="processing">Processing</option>
              <option value="failed">Failed</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Page Size</label>
            <select
              value={filters.page_size}
              onChange={(e) => setFilters({ ...filters, page_size: parseInt(e.target.value), page: 1 })}
              className="border border-gray-300 rounded-md px-3 py-2"
            >
              <option value={5}>5 per page</option>
              <option value={10}>10 per page</option>
              <option value={20}>20 per page</option>
              <option value={50}>50 per page</option>
            </select>
          </div>

          <button
            onClick={() => setRefreshTrigger(prev => prev + 1)}
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          {error}
          <button 
            onClick={() => setError(null)} 
            className="float-right text-red-500 hover:text-red-700"
          >
            ×
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Queries List */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b">
            <h2 className="text-xl font-semibold">Your Queries</h2>
            {queries && (
              <p className="text-sm text-gray-600">
                Showing {queries.queries.length} of {queries.total} queries
              </p>
            )}
          </div>

          <div className="divide-y">
            {isLoading ? (
              <div className="p-4 text-center text-gray-500">Loading queries...</div>
            ) : queries?.queries.length === 0 ? (
              <div className="p-4 text-center text-gray-500">No queries found</div>
            ) : (
              queries?.queries.map((query: QuerySummary) => (
                <div 
                  key={query.query_id} 
                  className={`p-4 cursor-pointer hover:bg-gray-50 ${
                    selectedQuery?.query_id === query.query_id ? 'bg-blue-50 border-l-4 border-blue-500' : ''
                  }`}
                  onClick={() => loadQueryDetails(query.query_id)}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-medium text-gray-900 truncate flex-1 mr-2">
                      {query.query.length > 80 ? `${query.query.substring(0, 80)}...` : query.query}
                    </h3>
                    <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(query.status)}`}>
                      {query.status}
                    </span>
                  </div>
                  
                  <div className="text-sm text-gray-500 space-y-1">
                    <div>Confidence: {(query.confidence * 100).toFixed(1)}%</div>
                    <div>Created: {formatDate(query.created_at)}</div>
                    <div>Processing: {(query.processing_time * 1000).toFixed(0)}ms</div>
                  </div>

                  <div className="mt-2 flex gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleReprocessQuery(query.query_id)
                      }}
                      className="text-xs px-2 py-1 bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200"
                    >
                      Reprocess
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDeleteQuery(query.query_id)
                      }}
                      className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Pagination */}
          {queries && queries.total > 0 && (
            <div className="p-4 border-t flex justify-between items-center">
              <div className="text-sm text-gray-600">
                Page {queries.page} of {Math.ceil(queries.total / queries.page_size)}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setFilters({ ...filters, page: Math.max(1, (filters.page || 1) - 1) })}
                  disabled={(filters.page || 1) <= 1}
                  className="px-3 py-1 border rounded disabled:opacity-50"
                >
                  Previous
                </button>
                <button
                  onClick={() => setFilters({ ...filters, page: (filters.page || 1) + 1 })}
                  disabled={!queries.has_next}
                  className="px-3 py-1 border rounded disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Query Details */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b flex justify-between items-center">
            <h2 className="text-xl font-semibold">Query Details</h2>
            {selectedQuery && (
              <button
                onClick={() => {
                  setEditData({
                    query: selectedQuery.query,
                    max_tokens: 1000,
                    confidence_threshold: 0.8,
                    reprocess: false
                  })
                  setShowEditModal(true)
                }}
                className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Edit Query
              </button>
            )}
          </div>

          {selectedQuery ? (
            <div className="p-4 space-y-4">
              <div>
                <h3 className="font-medium text-gray-900 mb-2">Query</h3>
                <p className="text-gray-700 bg-gray-50 p-3 rounded">{selectedQuery.query}</p>
              </div>

              <div>
                <h3 className="font-medium text-gray-900 mb-2">Answer</h3>
                <div className="text-gray-700 bg-gray-50 p-3 rounded">
                  {selectedQuery.answer || <em className="text-gray-500">No answer generated</em>}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium text-gray-900">Status</h4>
                  <span className={`inline-block px-2 py-1 text-sm rounded ${getStatusColor(selectedQuery.status)}`}>
                    {selectedQuery.status}
                  </span>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Confidence</h4>
                  <p className="text-gray-700">{(selectedQuery.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium text-gray-900">Created</h4>
                  <p className="text-gray-700">{formatDate(selectedQuery.created_at)}</p>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Processing Time</h4>
                  <p className="text-gray-700">{(selectedQuery.processing_time * 1000).toFixed(0)}ms</p>
                </div>
              </div>

              {selectedQuery.updated_at && (
                <div>
                  <h4 className="font-medium text-gray-900">Last Updated</h4>
                  <p className="text-gray-700">{formatDate(selectedQuery.updated_at)}</p>
                </div>
              )}

              {selectedQuery.citations.length > 0 && (
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Citations</h4>
                  <div className="space-y-2">
                    {selectedQuery.citations.map((citation, index) => (
                      <div key={index} className="bg-gray-50 p-2 rounded text-sm">
                        <p className="text-gray-700">{citation.text}</p>
                        {citation.url && (
                          <a href={citation.url} target="_blank" rel="noopener noreferrer" 
                             className="text-blue-600 hover:underline">
                            {citation.url}
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-4 text-center text-gray-500">
              Select a query to view details
            </div>
          )}
        </div>
      </div>

      {/* Edit Modal */}
      {showEditModal && selectedQuery && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full p-6">
            <h3 className="text-lg font-semibold mb-4">Edit Query</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Query Text</label>
                <textarea
                  value={editData.query || selectedQuery.query}
                  onChange={(e) => setEditData({ ...editData, query: e.target.value })}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 h-24 resize-none"
                  placeholder="Enter your query..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Max Tokens</label>
                <input
                  type="number"
                  value={editData.max_tokens || 1000}
                  onChange={(e) => setEditData({ ...editData, max_tokens: parseInt(e.target.value) })}
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  min="100"
                  max="4000"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Confidence Threshold</label>
                <input
                  type="number"
                  value={editData.confidence_threshold || 0.8}
                  onChange={(e) => setEditData({ ...editData, confidence_threshold: parseFloat(e.target.value) })}
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  min="0"
                  max="1"
                  step="0.1"
                />
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="reprocess"
                  checked={editData.reprocess || false}
                  onChange={(e) => setEditData({ ...editData, reprocess: e.target.checked })}
                  className="mr-2"
                />
                <label htmlFor="reprocess" className="text-sm text-gray-700">
                  Reprocess query after update
                </label>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={handleUpdateQuery}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Update Query
              </button>
              <button
                onClick={() => {
                  setShowEditModal(false)
                  setEditData({})
                }}
                className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 