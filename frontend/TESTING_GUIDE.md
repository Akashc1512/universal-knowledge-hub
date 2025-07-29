# Frontend Testing Guide

## 🚀 Quick Start

### 1. Start the Mock Backend
```bash
cd frontend
node mock-backend.js
```
This will start a mock backend server on `http://localhost:8002`

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```
This will start the Next.js development server on `http://localhost:3000`

## 🧪 Testing Checklist

### ✅ Basic Functionality
- [ ] **Homepage loads** - Visit `http://localhost:3000`
- [ ] **Prototype notice visible** - Blue info box at the top
- [ ] **Query form accessible** - Large text area with placeholder
- [ ] **Example questions clickable** - Try clicking the example questions

### ✅ Query Testing
- [ ] **Submit a query** - Enter "What is the capital of France?" and submit
- [ ] **Loading state** - Should see "Analyzing..." and spinner
- [ ] **Answer display** - Should show "Paris is the capital of France..."
- [ ] **Confidence badge** - Should show "High (95%)" in green
- [ ] **Citations** - Should show Wikipedia citation with link
- [ ] **Processing time** - Should show "Processed in 0.80s"
- [ ] **Cache indicator** - Should show "Fresh Result" (not cached)

### ✅ Error Handling
- [ ] **Empty query** - Try submitting empty query, should show error
- [ ] **Short query** - Try "Hi", should show "too short" error
- [ ] **Network error** - Stop backend, submit query, should show error message

### ✅ Feedback System
- [ ] **Helpful feedback** - Click "Helpful" button
- [ ] **Not helpful feedback** - Click "Not Helpful", should show details form
- [ ] **Feedback submission** - Submit feedback with details
- [ ] **Success message** - Should show "Thank you for your feedback!"

### ✅ Analytics Dashboard
- [ ] **Analytics button** - Click "Analytics" in header
- [ ] **Dashboard opens** - Should show modal with metrics
- [ ] **Key metrics** - Total requests, success rate, response time, cache hit rate
- [ ] **Popular queries** - Should show query categories
- [ ] **Close modal** - Click X or outside modal to close

### ✅ Accessibility
- [ ] **Keyboard navigation** - Tab through all interactive elements
- [ ] **Screen reader** - Test with screen reader (if available)
- [ ] **Focus indicators** - All interactive elements should have visible focus
- [ ] **ARIA labels** - Check browser dev tools for proper ARIA attributes

## 🔧 Mock Backend Features

### Query Responses
The mock backend provides realistic responses for testing:

1. **"What is the capital of France?"**
   - Answer: Detailed explanation about Paris
   - Confidence: 95%
   - Citations: Wikipedia source
   - Processing time: ~0.8s

2. **"How does photosynthesis work?"**
   - Answer: Scientific explanation
   - Confidence: 88%
   - Citations: National Geographic source
   - Processing time: ~1.1s

3. **Other queries**
   - Generic prototype response
   - Low confidence (30%)
   - No citations
   - Explains limitations

### Analytics Data
- Total requests: 42
- Total errors: 3
- Average response time: 1.2s
- Cache hit rate: 15.5%
- Popular query categories: technology, science, education, general

## 🐛 Troubleshooting

### Frontend Issues
- **Build errors**: Check TypeScript compilation with `npx tsc --noEmit`
- **Port conflicts**: Change port in `package.json` dev script
- **Module errors**: Run `npm install` to ensure dependencies

### Backend Issues
- **Port 8002 in use**: Kill existing process or change port in mock-backend.js
- **CORS errors**: Ensure mock backend has CORS enabled
- **Connection refused**: Check if mock backend is running

### Network Issues
- **API calls failing**: Check browser dev tools Network tab
- **CORS errors**: Verify backend CORS configuration
- **Timeout errors**: Check if backend is responding

## 📊 Expected Network Calls

### Successful Query
```
POST http://localhost:8002/query
Content-Type: application/json
X-API-Key: user-key-456

{
  "query": "What is the capital of France?"
}
```

**Response:**
```json
{
  "answer": "Paris is the capital of France...",
  "confidence": 0.95,
  "citations": [...],
  "query_id": "mock-123",
  "processing_time": 0.8,
  "query_type": "fresh"
}
```

### Analytics Request
```
GET http://localhost:8002/analytics
```

**Response:**
```json
{
  "total_requests": 42,
  "total_errors": 3,
  "average_response_time": 1.2,
  "cache_hit_rate": 15.5,
  "popular_queries": {...},
  "timestamp": "..."
}
```

## 🎯 Demo Scenarios

### Scenario 1: Basic Query
1. Open `http://localhost:3000`
2. Enter "What is the capital of France?"
3. Submit query
4. Verify answer, confidence, citations, and processing time

### Scenario 2: Feedback Flow
1. Submit any query
2. Click "Not Helpful"
3. Enter feedback details
4. Submit feedback
5. Verify success message

### Scenario 3: Analytics Dashboard
1. Click "Analytics" in header
2. Review all metrics
3. Check popular query categories
4. Close dashboard

### Scenario 4: Error Handling
1. Stop mock backend
2. Submit a query
3. Verify error message appears
4. Restart backend and retry

## 📝 Notes

- The mock backend simulates realistic delays (0.5-1.5s)
- All responses include proper confidence scores and citations
- The prototype notice sets proper expectations for users
- The UI is fully accessible and responsive
- Analytics dashboard shows realistic metrics for demonstration 