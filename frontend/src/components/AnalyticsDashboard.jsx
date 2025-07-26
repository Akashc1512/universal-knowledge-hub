/**
 * Analytics Dashboard Component
 * Universal Knowledge Platform - Real-time analytics and insights
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  People,
  Search,
  Bookmark,
  Share,
  Visibility,
  Speed,
  Analytics,
  Refresh,
  Download,
  FilterList
} from '@mui/icons-material';

const AnalyticsDashboard = () => {
  const [analyticsData, setAnalyticsData] = useState({
    overview: {
      totalUsers: 0,
      activeUsers: 0,
      totalSearches: 0,
      totalRecommendations: 0,
      averageResponseTime: 0,
      cacheHitRate: 0
    },
    trends: {
      dailySearches: [],
      userEngagement: [],
      recommendationAccuracy: [],
      systemPerformance: []
    },
    userBehavior: {
      topSearches: [],
      popularTopics: [],
      userJourneys: [],
      conversionRates: []
    },
    systemMetrics: {
      apiCalls: 0,
      errors: 0,
      uptime: 100,
      memoryUsage: 0,
      cpuUsage: 0
    }
  });

  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  
  const chartRefs = useRef({});
  const updateInterval = useRef(null);

  // Fetch analytics data
  const fetchAnalyticsData = async () => {
    try {
      setIsLoading(true);
      
      // Simulate API call - replace with actual endpoint
      const response = await fetch(`/api/analytics?range=${selectedTimeRange}`);
      const data = await response.json();
      
      setAnalyticsData(data);
      setLastUpdated(new Date());
      
    } catch (error) {
      console.error('Failed to fetch analytics data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Real-time updates
  useEffect(() => {
    fetchAnalyticsData();
    
    // Set up real-time updates every 30 seconds
    updateInterval.current = setInterval(fetchAnalyticsData, 30000);
    
    return () => {
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
      }
    };
  }, [selectedTimeRange]);

  // Chart rendering
  useEffect(() => {
    if (!isLoading && analyticsData.trends) {
      renderCharts();
    }
  }, [analyticsData, isLoading]);

  const renderCharts = () => {
    // Render search trends chart
    if (chartRefs.current.searchTrends) {
      renderSearchTrendsChart();
    }
    
    // Render user engagement chart
    if (chartRefs.current.userEngagement) {
      renderUserEngagementChart();
    }
    
    // Render recommendation accuracy chart
    if (chartRefs.current.recommendationAccuracy) {
      renderRecommendationAccuracyChart();
    }
  };

  const renderSearchTrendsChart = () => {
    const ctx = chartRefs.current.searchTrends.getContext('2d');
    const data = analyticsData.trends.dailySearches;
    
    // Create chart using Chart.js or similar library
    // This is a placeholder for actual chart implementation
    console.log('Rendering search trends chart with data:', data);
  };

  const renderUserEngagementChart = () => {
    const ctx = chartRefs.current.userEngagement.getContext('2d');
    const data = analyticsData.trends.userEngagement;
    
    console.log('Rendering user engagement chart with data:', data);
  };

  const renderRecommendationAccuracyChart = () => {
    const ctx = chartRefs.current.recommendationAccuracy.getContext('2d');
    const data = analyticsData.trends.recommendationAccuracy;
    
    console.log('Rendering recommendation accuracy chart with data:', data);
  };

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  const getTrendIcon = (trend) => {
    return trend > 0 ? <TrendingUp className="trend-up" /> : <TrendingDown className="trend-down" />;
  };

  const getMetricColor = (metric, value) => {
    if (metric === 'uptime' || metric === 'cacheHitRate') {
      return value >= 95 ? 'success' : value >= 80 ? 'warning' : 'error';
    }
    return 'default';
  };

  return (
    <div className="analytics-dashboard">
      {/* Dashboard Header */}
      <motion.div 
        className="dashboard-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="header-content">
          <h1>Analytics Dashboard</h1>
          <p>Real-time insights and performance metrics</p>
        </div>
        
        <div className="header-actions">
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={fetchAnalyticsData}
            className="refresh-button"
            disabled={isLoading}
          >
            <Refresh className={isLoading ? 'spinning' : ''} />
            Refresh
          </motion.button>
          
          <select 
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="time-range-selector"
          >
            <option value="1d">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
          </select>
        </div>
      </motion.div>

      {/* Overview Metrics */}
      <motion.div 
        className="overview-metrics"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <div className="metrics-grid">
          <motion.div 
            className="metric-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <div className="metric-icon">
              <People />
            </div>
            <div className="metric-content">
              <h3>Total Users</h3>
              <p className="metric-value">{formatNumber(analyticsData.overview.totalUsers)}</p>
              <span className="metric-trend">
                {getTrendIcon(12.5)} +12.5%
              </span>
            </div>
          </motion.div>

          <motion.div 
            className="metric-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <div className="metric-icon">
              <Search />
            </div>
            <div className="metric-content">
              <h3>Total Searches</h3>
              <p className="metric-value">{formatNumber(analyticsData.overview.totalSearches)}</p>
              <span className="metric-trend">
                {getTrendIcon(8.3)} +8.3%
              </span>
            </div>
          </motion.div>

          <motion.div 
            className="metric-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <div className="metric-icon">
              <Speed />
            </div>
            <div className="metric-content">
              <h3>Avg Response Time</h3>
              <p className="metric-value">{analyticsData.overview.averageResponseTime}ms</p>
              <span className="metric-trend">
                {getTrendIcon(-5.2)} -5.2%
              </span>
            </div>
          </motion.div>

          <motion.div 
            className="metric-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <div className="metric-icon">
              <Analytics />
            </div>
            <div className="metric-content">
              <h3>Cache Hit Rate</h3>
              <p className="metric-value">{analyticsData.overview.cacheHitRate}%</p>
              <span className="metric-trend">
                {getTrendIcon(2.1)} +2.1%
              </span>
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* Charts Section */}
      <motion.div 
        className="charts-section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="charts-grid">
          {/* Search Trends Chart */}
          <motion.div 
            className="chart-card"
            whileHover={{ scale: 1.01 }}
            transition={{ duration: 0.2 }}
          >
            <div className="chart-header">
              <h3>Search Trends</h3>
              <FilterList />
            </div>
            <canvas 
              ref={(el) => chartRefs.current.searchTrends = el}
              className="chart-canvas"
            />
          </motion.div>

          {/* User Engagement Chart */}
          <motion.div 
            className="chart-card"
            whileHover={{ scale: 1.01 }}
            transition={{ duration: 0.2 }}
          >
            <div className="chart-header">
              <h3>User Engagement</h3>
              <Visibility />
            </div>
            <canvas 
              ref={(el) => chartRefs.current.userEngagement = el}
              className="chart-canvas"
            />
          </motion.div>

          {/* Recommendation Accuracy Chart */}
          <motion.div 
            className="chart-card"
            whileHover={{ scale: 1.01 }}
            transition={{ duration: 0.2 }}
          >
            <div className="chart-header">
              <h3>Recommendation Accuracy</h3>
              <Bookmark />
            </div>
            <canvas 
              ref={(el) => chartRefs.current.recommendationAccuracy = el}
              className="chart-canvas"
            />
          </motion.div>
        </div>
      </motion.div>

      {/* User Behavior Insights */}
      <motion.div 
        className="insights-section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <h2>User Behavior Insights</h2>
        
        <div className="insights-grid">
          {/* Top Searches */}
          <motion.div 
            className="insight-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <h3>Top Searches</h3>
            <div className="insight-list">
              {analyticsData.userBehavior.topSearches.map((search, index) => (
                <div key={index} className="insight-item">
                  <span className="rank">{index + 1}</span>
                  <span className="search-term">{search.term}</span>
                  <span className="count">{search.count}</span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Popular Topics */}
          <motion.div 
            className="insight-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <h3>Popular Topics</h3>
            <div className="topic-tags">
              {analyticsData.userBehavior.popularTopics.map((topic, index) => (
                <motion.span
                  key={index}
                  className="topic-tag"
                  whileHover={{ scale: 1.1 }}
                  style={{ fontSize: `${Math.max(12, topic.weight * 2)}px` }}
                >
                  {topic.name}
                </motion.span>
              ))}
            </div>
          </motion.div>

          {/* Conversion Rates */}
          <motion.div 
            className="insight-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <h3>Conversion Rates</h3>
            <div className="conversion-metrics">
              {analyticsData.userBehavior.conversionRates.map((metric, index) => (
                <div key={index} className="conversion-item">
                  <span className="metric-name">{metric.name}</span>
                  <div className="progress-bar">
                    <motion.div
                      className="progress-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${metric.rate}%` }}
                      transition={{ duration: 1, delay: index * 0.1 }}
                    />
                  </div>
                  <span className="rate">{metric.rate}%</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* System Performance */}
      <motion.div 
        className="system-performance"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <h2>System Performance</h2>
        
        <div className="performance-grid">
          <motion.div 
            className="performance-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <h3>API Calls</h3>
            <p className="performance-value">{formatNumber(analyticsData.systemMetrics.apiCalls)}</p>
            <span className="performance-label">per minute</span>
          </motion.div>

          <motion.div 
            className="performance-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <h3>Error Rate</h3>
            <p className="performance-value">{analyticsData.systemMetrics.errors}</p>
            <span className="performance-label">errors</span>
          </motion.div>

          <motion.div 
            className="performance-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <h3>Uptime</h3>
            <p className={`performance-value ${getMetricColor('uptime', analyticsData.systemMetrics.uptime)}`}>
              {analyticsData.systemMetrics.uptime}%
            </p>
            <span className="performance-label">system uptime</span>
          </motion.div>

          <motion.div 
            className="performance-card"
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <h3>Memory Usage</h3>
            <p className="performance-value">{analyticsData.systemMetrics.memoryUsage}%</p>
            <span className="performance-label">RAM usage</span>
          </motion.div>
        </div>
      </motion.div>

      {/* Loading State */}
      <AnimatePresence>
        {isLoading && (
          <motion.div 
            className="loading-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="loading-spinner"></div>
            <p>Loading analytics data...</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Last Updated */}
      <motion.div 
        className="last-updated"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <p>Last updated: {lastUpdated.toLocaleTimeString()}</p>
      </motion.div>
    </div>
  );
};

export default AnalyticsDashboard; 