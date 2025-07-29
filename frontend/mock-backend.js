const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 8002;

app.use(cors());
app.use(express.json());

// Mock analytics data
const mockAnalytics = {
  total_requests: 42,
  total_errors: 3,
  average_response_time: 1.2,
  cache_hit_rate: 15.5,
  popular_queries: {
    "technology": 12,
    "science": 8,
    "education": 6,
    "general": 16
  },
  timestamp: new Date().toISOString()
};

// Mock query responses
const mockResponses = {
  "capital of france": {
    answer: "Paris is the capital of France. It is the largest city in France and serves as the country's political, economic, and cultural center.",
    confidence: 0.95,
    citations: [
      {
        id: "1",
        text: "Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture.",
        title: "Paris - Wikipedia",
        url: "https://en.wikipedia.org/wiki/Paris"
      }
    ],
    query_id: "mock-123",
    processing_time: 0.8,
    query_type: "fresh"
  },
  "photosynthesis": {
    answer: "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy. This process produces oxygen as a byproduct and is essential for life on Earth.",
    confidence: 0.88,
    citations: [
      {
        id: "2",
        text: "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy.",
        title: "Photosynthesis - National Geographic",
        url: "https://www.nationalgeographic.org/encyclopedia/photosynthesis/"
      }
    ],
    query_id: "mock-456",
    processing_time: 1.1,
    query_type: "fresh"
  }
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

// Query endpoint
app.post('/query', (req, res) => {
  const { query } = req.body;
  
  if (!query) {
    return res.status(400).json({
      error: 'Query is required'
    });
  }

  // Simulate processing delay
  setTimeout(() => {
    const queryLower = query.toLowerCase();
    let response;

    // Check for specific mock responses
    if (queryLower.includes('capital') && queryLower.includes('france')) {
      response = mockResponses["capital of france"];
    } else if (queryLower.includes('photosynthesis')) {
      response = mockResponses["photosynthesis"];
    } else {
      // Default response for other queries
      response = {
        answer: "This is a prototype system with limited knowledge. For this query, I would need to access real search engines and language models to provide an accurate answer. In the full version, this would be processed by multiple AI agents to retrieve and synthesize information from reliable sources.",
        confidence: 0.3,
        citations: [],
        query_id: `mock-${Date.now()}`,
        processing_time: 0.5,
        query_type: "fresh"
      };
    }

    res.json(response);
  }, 500 + Math.random() * 1000); // Random delay between 500ms and 1.5s
});

// Analytics endpoint
app.get('/analytics', (req, res) => {
  res.json(mockAnalytics);
});

// Feedback endpoint
app.post('/feedback', (req, res) => {
  const { query_id, feedback_type, details } = req.body;
  
  console.log('Feedback received:', { query_id, feedback_type, details });
  
  res.json({
    status: 'success',
    message: 'Feedback recorded successfully'
  });
});

app.listen(PORT, () => {
  console.log(`Mock backend server running on http://localhost:${PORT}`);
  console.log('Available endpoints:');
  console.log('  GET  /health');
  console.log('  POST /query');
  console.log('  GET  /analytics');
  console.log('  POST /feedback');
}); 