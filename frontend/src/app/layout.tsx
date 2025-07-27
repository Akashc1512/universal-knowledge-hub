import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'SarvanOM - Your Own Knowledge Hub Powered by AI',
  description: 'Your Own Knowledge Hub Powered by AI - Get accurate, verifiable answers with source citations and confidence scores.',
  keywords: 'AI, knowledge hub, artificial intelligence, search, answers, citations, SarvanOM',
  authors: [{ name: 'SarvanOM Team' }],
  creator: 'SarvanOM',
  publisher: 'SarvanOM',
  robots: 'index, follow',
  openGraph: {
    title: 'SarvanOM - Your Own Knowledge Hub Powered by AI',
    description: 'Your Own Knowledge Hub Powered by AI - Get accurate, verifiable answers with source citations and confidence scores.',
    url: 'https://sarvanom.ai',
    siteName: 'SarvanOM',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'SarvanOM - Your Own Knowledge Hub Powered by AI',
    description: 'Your Own Knowledge Hub Powered by AI - Get accurate, verifiable answers with source citations and confidence scores.',
    creator: '@sarvanom',
  },
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#3B82F6',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
          <header className="bg-white shadow-sm border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center py-4">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-lg">🧠</span>
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">SarvanOM</h1>
                    <p className="text-sm text-gray-600">Your Own Knowledge Hub Powered by AI</p>
                  </div>
                </div>
                <nav className="hidden md:flex space-x-8">
                  <a href="https://sarvanom.ai" className="text-gray-600 hover:text-blue-600 transition-colors">
                    Home
                  </a>
                  <a href="https://docs.sarvanom.ai" className="text-gray-600 hover:text-blue-600 transition-colors">
                    Documentation
                  </a>
                  <a href="https://api.sarvanom.ai" className="text-gray-600 hover:text-blue-600 transition-colors">
                    API
                  </a>
                </nav>
              </div>
            </div>
          </header>
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </main>
          <footer className="bg-white border-t border-gray-200 mt-16">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              <div className="text-center">
                <p className="text-gray-600">
                  © 2024 SarvanOM. Your Own Knowledge Hub Powered by AI.
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Built with ❤️ using Next.js, FastAPI, and AI
                </p>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  )
}
