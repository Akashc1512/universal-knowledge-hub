#!/usr/bin/env node

/**
 * Frontend Health Check Script
 * 
 * This script provides a simple health check for the Next.js frontend.
 * It can be used in Docker health checks or monitoring systems.
 * 
 * Usage:
 *   node health-check.js
 *   curl http://localhost:3000/health
 */

const http = require('http');

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || 'localhost';

function checkHealth() {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: HOST,
      port: PORT,
      path: '/',
      method: 'GET',
      timeout: 5000
    };

    const req = http.request(options, (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        if (res.statusCode === 200) {
          console.log('✅ Frontend health check passed');
          resolve({
            status: 'healthy',
            statusCode: res.statusCode,
            responseTime: Date.now() - startTime
          });
        } else {
          console.log(`❌ Frontend health check failed: ${res.statusCode}`);
          reject(new Error(`HTTP ${res.statusCode}`));
        }
      });
    });

    req.on('error', (err) => {
      console.log(`❌ Frontend health check failed: ${err.message}`);
      reject(err);
    });

    req.on('timeout', () => {
      console.log('❌ Frontend health check timed out');
      req.destroy();
      reject(new Error('Timeout'));
    });

    const startTime = Date.now();
    req.end();
  });
}

// If run directly, perform health check
if (require.main === module) {
  checkHealth()
    .then((result) => {
      console.log('Health check result:', result);
      process.exit(0);
    })
    .catch((error) => {
      console.error('Health check failed:', error.message);
      process.exit(1);
    });
}

module.exports = { checkHealth }; 