/**
 * Mobile-Optimized Search Component
 * Universal Knowledge Platform - Touch-friendly search interface
 */

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  Mic, 
  MicOff, 
  Filter, 
  Sort, 
  Bookmark,
  Share,
  Download,
  VoiceChat,
  TouchApp
} from '@mui/icons-material';

const MobileSearch = ({ onSearch, onVoiceSearch, recommendations = [] }) => {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [searchHistory, setSearchHistory] = useState([]);
  const [recentSearches, setRecentSearches] = useState([]);
  
  const searchInputRef = useRef(null);
  const voiceRecognitionRef = useRef(null);
  const touchStartRef = useRef(null);
  const touchEndRef = useRef(null);

  // Voice recognition setup
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      voiceRecognitionRef.current = new SpeechRecognition();
      voiceRecognitionRef.current.continuous = false;
      voiceRecognitionRef.current.interimResults = false;
      voiceRecognitionRef.current.lang = 'en-US';

      voiceRecognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setQuery(transcript);
        setIsListening(false);
        handleSearch(transcript);
      };

      voiceRecognitionRef.current.onerror = (event) => {
        console.error('Voice recognition error:', event.error);
        setIsListening(false);
      };
    }
  }, []);

  // Touch gesture handling
  const handleTouchStart = (e) => {
    touchStartRef.current = e.touches[0].clientX;
  };

  const handleTouchMove = (e) => {
    touchEndRef.current = e.touches[0].clientX;
  };

  const handleTouchEnd = () => {
    if (!touchStartRef.current || !touchEndRef.current) return;

    const distance = touchStartRef.current - touchEndRef.current;
    const isLeftSwipe = distance > 50;
    const isRightSwipe = distance < -50;

    if (isLeftSwipe) {
      // Swipe left - show filters
      setShowFilters(true);
    } else if (isRightSwipe) {
      // Swipe right - hide filters
      setShowFilters(false);
    }

    touchStartRef.current = null;
    touchEndRef.current = null;
  };

  const handleSearch = async (searchQuery = query) => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    
    try {
      // Add to search history
      const newHistory = [searchQuery, ...searchHistory.filter(q => q !== searchQuery)].slice(0, 10);
      setSearchHistory(newHistory);
      localStorage.setItem('searchHistory', JSON.stringify(newHistory));

      // Call search function
      await onSearch(searchQuery);
      
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleVoiceSearch = () => {
    if (!voiceRecognitionRef.current) {
      alert('Voice recognition not supported in this browser');
      return;
    }

    setIsListening(true);
    voiceRecognitionRef.current.start();
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleQuickSearch = (quickQuery) => {
    setQuery(quickQuery);
    handleSearch(quickQuery);
  };

  const handleShare = async (item) => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: item.title,
          text: item.summary,
          url: item.url
        });
      } catch (error) {
        console.error('Share failed:', error);
      }
    } else {
      // Fallback to clipboard
      navigator.clipboard.writeText(`${item.title}\n${item.url}`);
    }
  };

  const handleBookmark = (item) => {
    const bookmarks = JSON.parse(localStorage.getItem('bookmarks') || '[]');
    const newBookmarks = [...bookmarks, { ...item, bookmarkedAt: new Date().toISOString() }];
    localStorage.setItem('bookmarks', JSON.stringify(newBookmarks));
  };

  // Load search history on mount
  useEffect(() => {
    const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
    setSearchHistory(history);
  }, []);

  return (
    <div className="mobile-search-container">
      {/* Search Header */}
      <motion.div 
        className="search-header"
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="search-input-container">
          <Search className="search-icon" />
          <input
            ref={searchInputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Search knowledge base..."
            className="search-input"
            autoComplete="off"
          />
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={handleVoiceSearch}
            className={`voice-button ${isListening ? 'listening' : ''}`}
            disabled={isListening}
          >
            {isListening ? <MicOff /> : <Mic />}
          </motion.button>
        </div>

        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={() => setShowFilters(!showFilters)}
          className="filter-button"
        >
          <Filter />
        </motion.button>
      </motion.div>

      {/* Filters Panel */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            className="filters-panel"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="filter-options">
              <div className="filter-group">
                <label>Content Type</label>
                <div className="filter-chips">
                  {['All', 'Documents', 'Articles', 'Videos', 'Images'].map(type => (
                    <button key={type} className="filter-chip">
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              <div className="filter-group">
                <label>Date Range</label>
                <div className="filter-chips">
                  {['All Time', 'Last Week', 'Last Month', 'Last Year'].map(range => (
                    <button key={range} className="filter-chip">
                      {range}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Search History */}
      {!query && searchHistory.length > 0 && (
        <motion.div 
          className="search-history"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <h3>Recent Searches</h3>
          <div className="history-items">
            {searchHistory.slice(0, 5).map((item, index) => (
              <motion.button
                key={index}
                whileTap={{ scale: 0.95 }}
                onClick={() => handleQuickSearch(item)}
                className="history-item"
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: index * 0.1 }}
              >
                <Search className="history-icon" />
                <span>{item}</span>
              </motion.button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Quick Search Suggestions */}
      {!query && (
        <motion.div 
          className="quick-suggestions"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <h3>Popular Topics</h3>
          <div className="suggestion-chips">
            {['Technology', 'Business', 'Science', 'Health', 'Education'].map(topic => (
              <motion.button
                key={topic}
                whileTap={{ scale: 0.95 }}
                onClick={() => handleQuickSearch(topic)}
                className="suggestion-chip"
                whileHover={{ scale: 1.05 }}
              >
                {topic}
              </motion.button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Search Results */}
      {recommendations.length > 0 && (
        <motion.div 
          className="search-results"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="results-header">
            <h3>Search Results ({recommendations.length})</h3>
            <button className="sort-button">
              <Sort />
            </button>
          </div>

          <div className="results-list">
            {recommendations.map((item, index) => (
              <motion.div
                key={item.id}
                className="result-item"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
              >
                <div className="result-content">
                  <h4 className="result-title">{item.title}</h4>
                  <p className="result-summary">{item.summary}</p>
                  <div className="result-meta">
                    <span className="result-type">{item.content_type}</span>
                    <span className="result-date">{new Date(item.last_modified).toLocaleDateString()}</span>
                  </div>
                </div>

                <div className="result-actions">
                  <motion.button
                    whileTap={{ scale: 0.9 }}
                    onClick={() => handleBookmark(item)}
                    className="action-button"
                  >
                    <Bookmark />
                  </motion.button>
                  
                  <motion.button
                    whileTap={{ scale: 0.9 }}
                    onClick={() => handleShare(item)}
                    className="action-button"
                  >
                    <Share />
                  </motion.button>
                  
                  <motion.button
                    whileTap={{ scale: 0.9 }}
                    onClick={() => window.open(item.url, '_blank')}
                    className="action-button"
                  >
                    <Download />
                  </motion.button>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Loading State */}
      {isSearching && (
        <motion.div 
          className="loading-state"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="loading-spinner"></div>
          <p>Searching knowledge base...</p>
        </motion.div>
      )}

      {/* Voice Recognition Status */}
      {isListening && (
        <motion.div 
          className="voice-status"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="voice-indicator">
            <Mic className="pulse" />
            <p>Listening... Speak now</p>
          </div>
        </motion.div>
      )}

      {/* Touch Gesture Hint */}
      <motion.div 
        className="gesture-hint"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.7 }}
        transition={{ delay: 2 }}
      >
        <TouchApp />
        <span>Swipe left for filters, right to hide</span>
      </motion.div>
    </div>
  );
};

export default MobileSearch; 