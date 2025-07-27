# Universal Knowledge Platform - Frontend

A modern, responsive web application built with Next.js, TypeScript, and Tailwind CSS for the Universal Knowledge Platform.

## 🚀 Features

- **Modern UI**: Clean, accessible interface built with Next.js and Tailwind CSS
- **TypeScript**: Full type safety and better development experience
- **Responsive Design**: Mobile-first design that works on all devices
- **Accessibility**: WCAG 2.1 AA compliant with proper ARIA labels
- **Real-time Feedback**: User feedback system for continuous improvement
- **Error Handling**: Comprehensive error handling and user feedback

## 🛠️ Tech Stack

- **Framework**: Next.js 14+ with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Heroicons
- **HTTP Client**: Axios
- **UI Components**: Headless UI

## 📦 Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Set up environment variables**:
   ```bash
   cp env.example .env.local
   ```
   
   Edit `.env.local` with your configuration:
   ```env
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8002
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🏗️ Project Structure

```
frontend/
├── src/
│   ├── app/                 # Next.js App Router pages
│   ├── components/          # Reusable UI components
│   ├── lib/                 # Utility functions and API client
│   └── types/               # TypeScript type definitions
├── public/                  # Static assets
└── env.example             # Environment variables template
```

## 🧩 Components

### Core Components

- **QueryForm**: Main query input with validation and examples
- **AnswerDisplay**: Renders answers with confidence scores and citations
- **CitationList**: Displays source citations with links
- **ConfidenceBadge**: Shows confidence level with color coding
- **FeedbackForm**: User feedback collection system

### API Integration

- **api.ts**: Axios-based API client with error handling
- **types/api.ts**: TypeScript interfaces for API responses

## 🎨 Design System

The application uses a consistent design system with:

- **Colors**: Blue primary, semantic colors for feedback
- **Typography**: Clear hierarchy with proper contrast
- **Spacing**: Consistent spacing using Tailwind utilities
- **Components**: Reusable, accessible components

## ♿ Accessibility

- **WCAG 2.1 AA Compliance**: Full accessibility support
- **Screen Reader Support**: Proper ARIA labels and roles
- **Keyboard Navigation**: All interactive elements keyboard accessible
- **High Contrast**: Proper color contrast ratios
- **Focus Management**: Clear focus indicators

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Run accessibility tests
npm run test:a11y
```

## 🚀 Deployment

### Vercel (Recommended)

1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Docker

```bash
# Build the Docker image
docker build -t universal-knowledge-frontend .

# Run the container
docker run -p 3000:3000 universal-knowledge-frontend
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_BASE_URL` | Backend API URL | `http://localhost:8002` |
| `NEXT_PUBLIC_APP_NAME` | Application name | `"Universal Knowledge Platform"` |
| `NEXT_PUBLIC_ENABLE_FEEDBACK` | Enable feedback system | `true` |
| `NEXT_PUBLIC_ENABLE_ANALYTICS` | Enable analytics | `true` |

### Backend Integration

The frontend integrates with the FastAPI backend:

- **Query Endpoint**: `POST /query` - Submit questions
- **Feedback Endpoint**: `POST /feedback` - Submit user feedback
- **Analytics Endpoint**: `GET /analytics` - Get usage statistics
- **Health Check**: `GET /health` - Backend health status

## 🐛 Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure backend has CORS configured for frontend domain
2. **API Connection**: Verify backend is running on correct port
3. **Environment Variables**: Check `.env.local` file exists and is configured

### Development Tips

- Use browser dev tools to inspect network requests
- Check console for API error messages
- Verify backend health endpoint is accessible

## 📈 Performance

- **Lighthouse Score**: Target > 90
- **First Contentful Paint**: < 1.5s
- **Largest Contentful Paint**: < 2.5s
- **Cumulative Layout Shift**: < 0.1

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is part of the Universal Knowledge Platform.
