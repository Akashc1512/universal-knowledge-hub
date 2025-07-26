/**
 * Progressive Web App Service Worker
 * Universal Knowledge Platform - Offline-first caching and sync
 */

const CACHE_NAME = 'ukp-v1.0.0';
const STATIC_CACHE = 'ukp-static-v1.0.0';
const DYNAMIC_CACHE = 'ukp-dynamic-v1.0.0';
const API_CACHE = 'ukp-api-v1.0.0';

// Cache strategies
const CACHE_STRATEGIES = {
  STATIC: 'cache-first',
  DYNAMIC: 'stale-while-revalidate',
  API: 'network-first',
  OFFLINE: 'cache-only'
};

// Files to cache immediately
const STATIC_FILES = [
  '/',
  '/index.html',
  '/static/js/main.js',
  '/static/css/main.css',
  '/static/js/recommendations.js',
  '/static/js/search.js',
  '/static/js/analytics.js',
  '/static/images/logo.png',
  '/static/images/icons/icon-192x192.png',
  '/static/images/icons/icon-512x512.png',
  '/manifest.json'
];

// API endpoints to cache
const API_ENDPOINTS = [
  '/api/health',
  '/api/recommendations',
  '/api/search',
  '/api/analytics'
];

// Background sync queue
let syncQueue = [];

/**
 * Install event - Cache static files
 */
self.addEventListener('install', (event) => {
  console.log('🔧 Service Worker installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('📦 Caching static files');
        return cache.addAll(STATIC_FILES);
      })
      .then(() => {
        console.log('✅ Static files cached successfully');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('❌ Failed to cache static files:', error);
      })
  );
});

/**
 * Activate event - Clean up old caches
 */
self.addEventListener('activate', (event) => {
  console.log('🚀 Service Worker activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== STATIC_CACHE && 
                cacheName !== DYNAMIC_CACHE && 
                cacheName !== API_CACHE) {
              console.log('🗑️ Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('✅ Service Worker activated');
        return self.clients.claim();
      })
  );
});

/**
 * Fetch event - Handle all network requests
 */
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Handle different types of requests
  if (isStaticFile(request)) {
    event.respondWith(handleStaticFile(request));
  } else if (isAPIRequest(request)) {
    event.respondWith(handleAPIRequest(request));
  } else {
    event.respondWith(handleDynamicRequest(request));
  }
});

/**
 * Check if request is for a static file
 */
function isStaticFile(request) {
  const url = new URL(request.url);
  return STATIC_FILES.includes(url.pathname) ||
         url.pathname.startsWith('/static/') ||
         url.pathname.startsWith('/images/');
}

/**
 * Check if request is for an API endpoint
 */
function isAPIRequest(request) {
  const url = new URL(request.url);
  return url.pathname.startsWith('/api/') ||
         API_ENDPOINTS.some(endpoint => url.pathname === endpoint);
}

/**
 * Handle static file requests (Cache First)
 */
async function handleStaticFile(request) {
  try {
    // Try cache first
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      console.log('📦 Serving from cache:', request.url);
      return cachedResponse;
    }
    
    // Fallback to network
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.error('❌ Static file fetch failed:', error);
    return new Response('Offline - Static file not available', { status: 503 });
  }
}

/**
 * Handle API requests (Network First)
 */
async function handleAPIRequest(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      // Cache successful responses
      const cache = await caches.open(API_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('🌐 Network failed, trying cache for API:', request.url);
    
    // Fallback to cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline response
    return new Response(JSON.stringify({
      error: 'Offline - API not available',
      timestamp: new Date().toISOString()
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Handle dynamic requests (Stale While Revalidate)
 */
async function handleDynamicRequest(request) {
  try {
    // Try cache first for faster response
    const cachedResponse = await caches.match(request);
    
    // Start network request in background
    const networkPromise = fetch(request).then((response) => {
      if (response.ok) {
        const cache = caches.open(DYNAMIC_CACHE);
        cache.then((cache) => cache.put(request, response.clone()));
      }
      return response;
    });
    
    // Return cached response if available, otherwise wait for network
    if (cachedResponse) {
      console.log('📦 Serving dynamic content from cache:', request.url);
      return cachedResponse;
    }
    
    return networkPromise;
  } catch (error) {
    console.error('❌ Dynamic request failed:', error);
    return new Response('Offline - Content not available', { status: 503 });
  }
}

/**
 * Background sync for offline actions
 */
self.addEventListener('sync', (event) => {
  console.log('🔄 Background sync triggered:', event.tag);
  
  if (event.tag === 'background-sync') {
    event.waitUntil(processSyncQueue());
  }
});

/**
 * Process the sync queue
 */
async function processSyncQueue() {
  if (syncQueue.length === 0) {
    console.log('📭 Sync queue is empty');
    return;
  }
  
  console.log(`🔄 Processing ${syncQueue.length} queued items`);
  
  const results = await Promise.allSettled(
    syncQueue.map(async (item) => {
      try {
        const response = await fetch(item.url, item.options);
        if (response.ok) {
          console.log('✅ Synced successfully:', item.url);
          return { success: true, item };
        } else {
          console.error('❌ Sync failed:', item.url, response.status);
          return { success: false, item, error: response.status };
        }
      } catch (error) {
        console.error('❌ Sync error:', item.url, error);
        return { success: false, item, error: error.message };
      }
    })
  );
  
  // Clear successful items from queue
  const successful = results.filter(r => r.status === 'fulfilled' && r.value.success);
  const failed = results.filter(r => r.status === 'fulfilled' && !r.value.success);
  
  syncQueue = syncQueue.filter(item => 
    !successful.some(r => r.value.item.id === item.id)
  );
  
  console.log(`✅ Sync complete: ${successful.length} successful, ${failed.length} failed`);
  
  // Store updated queue
  await storeSyncQueue();
}

/**
 * Store sync queue in IndexedDB
 */
async function storeSyncQueue() {
  try {
    const db = await openDB('ukp-sync', 1, {
      upgrade(db) {
        db.createObjectStore('sync-queue');
      }
    });
    
    await db.put('sync-queue', 'queue', syncQueue);
  } catch (error) {
    console.error('❌ Failed to store sync queue:', error);
  }
}

/**
 * Load sync queue from IndexedDB
 */
async function loadSyncQueue() {
  try {
    const db = await openDB('ukp-sync', 1, {
      upgrade(db) {
        db.createObjectStore('sync-queue');
      }
    });
    
    const queue = await db.get('sync-queue', 'queue');
    syncQueue = queue || [];
    console.log(`📥 Loaded ${syncQueue.length} items from sync queue`);
  } catch (error) {
    console.error('❌ Failed to load sync queue:', error);
    syncQueue = [];
  }
}

/**
 * Add item to sync queue
 */
function addToSyncQueue(url, options = {}) {
  const item = {
    id: Date.now() + Math.random(),
    url,
    options,
    timestamp: new Date().toISOString()
  };
  
  syncQueue.push(item);
  storeSyncQueue();
  
  console.log('📝 Added to sync queue:', url);
  
  // Request background sync
  if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
    navigator.serviceWorker.ready.then((registration) => {
      registration.sync.register('background-sync');
    });
  }
}

/**
 * Push notification handling
 */
self.addEventListener('push', (event) => {
  console.log('📱 Push notification received');
  
  const options = {
    body: event.data ? event.data.text() : 'New content available',
    icon: '/static/images/icons/icon-192x192.png',
    badge: '/static/images/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View',
        icon: '/static/images/icons/action-1.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/static/images/icons/action-2.png'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('Universal Knowledge Platform', options)
  );
});

/**
 * Notification click handling
 */
self.addEventListener('notificationclick', (event) => {
  console.log('👆 Notification clicked:', event.action);
  
  event.notification.close();
  
  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

/**
 * Message handling from main thread
 */
self.addEventListener('message', (event) => {
  console.log('📨 Message received:', event.data);
  
  switch (event.data.type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'ADD_TO_SYNC_QUEUE':
      addToSyncQueue(event.data.url, event.data.options);
      break;
      
    case 'GET_SYNC_STATUS':
      event.ports[0].postMessage({
        queueLength: syncQueue.length,
        lastSync: new Date().toISOString()
      });
      break;
      
    default:
      console.log('Unknown message type:', event.data.type);
  }
});

/**
 * Initialize service worker
 */
async function initializeServiceWorker() {
  console.log('🚀 Initializing Service Worker...');
  
  // Load sync queue
  await loadSyncQueue();
  
  // Set up periodic cache cleanup
  setInterval(async () => {
    try {
      const cacheNames = await caches.keys();
      for (const cacheName of cacheNames) {
        const cache = await caches.open(cacheName);
        const requests = await cache.keys();
        
        // Remove old entries (keep last 100)
        if (requests.length > 100) {
          const toDelete = requests.slice(0, requests.length - 100);
          for (const request of toDelete) {
            await cache.delete(request);
          }
          console.log(`🧹 Cleaned up ${toDelete.length} old cache entries`);
        }
      }
    } catch (error) {
      console.error('❌ Cache cleanup failed:', error);
    }
  }, 24 * 60 * 60 * 1000); // Daily cleanup
  
  console.log('✅ Service Worker initialized');
}

// Initialize when service worker starts
initializeServiceWorker();

/**
 * Utility functions
 */

// IndexedDB helper
function openDB(name, version, upgradeCallback) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, version);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = () => {
      upgradeCallback(request.result);
    };
  });
}

console.log('🔧 Service Worker script loaded'); 