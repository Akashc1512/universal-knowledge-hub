import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Chip,
  Grid,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Alert,
  Pagination,
  Divider,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  ExpandMore as ExpandMoreIcon,
  AccessTime as TimeIcon,
  Source as SourceIcon,
  Category as CategoryIcon,
} from '@mui/icons-material';
import { useSearchParams } from 'react-router-dom';

const SearchPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState(searchParams.get('q') || '');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    category: 'all',
    source: 'all',
    dateRange: [0, 100],
    confidence: [0, 100],
  });
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  // Mock search results
  const mockResults = [
    {
      id: 1,
      title: 'Introduction to Machine Learning',
      content: 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed...',
      source: 'AI Research Papers',
      category: 'Machine Learning',
      confidence: 0.95,
      timestamp: '2024-01-15T10:30:00Z',
      tags: ['AI', 'ML', 'Tutorial'],
    },
    {
      id: 2,
      title: 'Best Practices for API Design',
      content: 'When designing APIs, it\'s important to follow RESTful principles, use proper HTTP status codes, and provide comprehensive documentation...',
      source: 'Software Engineering Blog',
      category: 'Software Development',
      confidence: 0.88,
      timestamp: '2024-01-14T15:45:00Z',
      tags: ['API', 'REST', 'Design'],
    },
    {
      id: 3,
      title: 'Data Privacy and GDPR Compliance',
      content: 'The General Data Protection Regulation (GDPR) requires organizations to implement appropriate technical and organizational measures...',
      source: 'Legal Guidelines',
      category: 'Compliance',
      confidence: 0.92,
      timestamp: '2024-01-13T09:15:00Z',
      tags: ['GDPR', 'Privacy', 'Compliance'],
    },
  ];

  useEffect(() => {
    if (query) {
      performSearch();
    }
  }, [query, filters, page]);

  const performSearch = async () => {
    setLoading(true);
    setError(null);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock API response
      const filteredResults = mockResults.filter(result => {
        if (filters.category !== 'all' && result.category !== filters.category) return false;
        if (filters.source !== 'all' && result.source !== filters.source) return false;
        if (result.confidence < filters.confidence[0] / 100) return false;
        if (result.confidence > filters.confidence[1] / 100) return false;
        return true;
      });

      setResults(filteredResults);
      setTotalPages(Math.ceil(filteredResults.length / 10));
    } catch (err) {
      setError('Failed to perform search. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (event) => {
    event.preventDefault();
    if (query.trim()) {
      setSearchParams({ q: query.trim() });
      setPage(1);
    }
  };

  const handleFilterChange = (filterName, value) => {
    setFilters(prev => ({
      ...prev,
      [filterName]: value,
    }));
    setPage(1);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ display: 'flex', gap: 3, height: '100%' }}>
      {/* Filters Sidebar */}
      <Paper sx={{ p: 2, width: 300, height: 'fit-content', position: 'sticky', top: 20 }}>
        <Typography variant="h6" gutterBottom>
          <FilterIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Filters
        </Typography>
        
        <Divider sx={{ my: 2 }} />
        
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Category</InputLabel>
          <Select
            value={filters.category}
            label="Category"
            onChange={(e) => handleFilterChange('category', e.target.value)}
          >
            <MenuItem value="all">All Categories</MenuItem>
            <MenuItem value="Machine Learning">Machine Learning</MenuItem>
            <MenuItem value="Software Development">Software Development</MenuItem>
            <MenuItem value="Compliance">Compliance</MenuItem>
          </Select>
        </FormControl>

        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Source</InputLabel>
          <Select
            value={filters.source}
            label="Source"
            onChange={(e) => handleFilterChange('source', e.target.value)}
          >
            <MenuItem value="all">All Sources</MenuItem>
            <MenuItem value="AI Research Papers">AI Research Papers</MenuItem>
            <MenuItem value="Software Engineering Blog">Software Engineering Blog</MenuItem>
            <MenuItem value="Legal Guidelines">Legal Guidelines</MenuItem>
          </Select>
        </FormControl>

        <Typography gutterBottom>Confidence Score</Typography>
        <Slider
          value={filters.confidence}
          onChange={(e, newValue) => handleFilterChange('confidence', newValue)}
          valueLabelDisplay="auto"
          sx={{ mb: 2 }}
        />

        <Button
          variant="outlined"
          fullWidth
          onClick={() => {
            setFilters({
              category: 'all',
              source: 'all',
              dateRange: [0, 100],
              confidence: [0, 100],
            });
          }}
        >
          Clear Filters
        </Button>
      </Paper>

      {/* Main Content */}
      <Box sx={{ flexGrow: 1 }}>
        {/* Search Form */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <form onSubmit={handleSearch}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs>
                <TextField
                  fullWidth
                  placeholder="Search the knowledge base..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
                    ),
                  }}
                />
              </Grid>
              <Grid item>
                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
                >
                  Search
                </Button>
              </Grid>
            </Grid>
          </form>
        </Paper>

        {/* Results */}
        <Box>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : results.length > 0 ? (
            <>
              <Typography variant="h6" gutterBottom>
                {results.length} results found
              </Typography>
              
              {results.map((result) => (
                <Card key={result.id} sx={{ mb: 2 }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                      <Typography variant="h6" component="h2" gutterBottom>
                        {result.title}
                      </Typography>
                      <Chip
                        label={`${Math.round(result.confidence * 100)}%`}
                        color={getConfidenceColor(result.confidence)}
                        size="small"
                      />
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {result.content}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                      {result.tags.map((tag) => (
                        <Chip key={tag} label={tag} size="small" variant="outlined" />
                      ))}
                    </Box>
                    
                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', color: 'text.secondary' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <SourceIcon sx={{ fontSize: 16, mr: 0.5 }} />
                        <Typography variant="caption">{result.source}</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <CategoryIcon sx={{ fontSize: 16, mr: 0.5 }} />
                        <Typography variant="caption">{result.category}</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <TimeIcon sx={{ fontSize: 16, mr: 0.5 }} />
                        <Typography variant="caption">{formatDate(result.timestamp)}</Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              ))}
              
              {/* Pagination */}
              {totalPages > 1 && (
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                  <Pagination
                    count={totalPages}
                    page={page}
                    onChange={(e, value) => setPage(value)}
                    color="primary"
                  />
                </Box>
              )}
            </>
          ) : query && (
            <Alert severity="info">
              No results found for "{query}". Try adjusting your search terms or filters.
            </Alert>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default SearchPage; 