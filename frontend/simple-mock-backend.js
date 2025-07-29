const http = require('http');
const url = require('url');

const PORT = 8002;

// Mock data
const mockResponses = {
  '/health': {
    status: 'healthy',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  },
  '/query': {
    answer: 'This is a mock response from the Universal Knowledge Platform. The system is currently in prototype mode.',
    confidence: 0.85,
    citations: [
      {
        id: '1',
        text: 'Mock citation for demonstration purposes',
        title: 'Universal Knowledge Platform Documentation',
        author: 'System',
        date: '2025',
        url: 'https://sarvanom.com'
      }
    ],
    query_id: 'mock-query-123',
    processing_time: 0.5,
    query_type: 'fresh'
  },
  '/analytics': {
    total_requests: 42,
    total_errors: 2,
    average_response_time: 0.8,
    cache_hit_rate: 0.75,
    popular_queries: {
      'What is AI?': 15,
      'Machine learning basics': 12,
      'Python programming': 8,
      'Data science': 7
    },
    timestamp: new Date().toISOString()
  }
};

const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const path = parsedUrl.pathname;
  
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-API-Key');
  res.setHeader('Content-Type', 'application/json');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }
  
  console.log(`${req.method} ${path}`);
  
  if (path === '/health') {
    res.writeHead(200);
    res.end(JSON.stringify(mockResponses['/health']));
  } else if (path === '/query' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        console.log('Query received:', data.query);
        res.writeHead(200);
        res.end(JSON.stringify(mockResponses['/query']));
      } catch (error) {
        res.writeHead(400);
        res.end(JSON.stringify({ error: 'Invalid JSON' }));
      }
    });
  } else if (path === '/analytics') {
    res.writeHead(200);
    res.end(JSON.stringify(mockResponses['/analytics']));
  } else if (path === '/feedback' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      console.log('Feedback received');
      res.writeHead(200);
      res.end(JSON.stringify({ status: 'success', message: 'Feedback received' }));
    });
  } else {
    res.writeHead(404);
    res.end(JSON.stringify({ error: 'Not found' }));
  }
});

server.listen(PORT, () => {
  console.log(`Mock backend server running on http://localhost:${PORT}`);
  console.log('Available endpoints:');
  console.log('  GET  /health');
  console.log('  POST /query');
  console.log('  GET  /analytics');
  console.log('  POST /feedback');
}); 