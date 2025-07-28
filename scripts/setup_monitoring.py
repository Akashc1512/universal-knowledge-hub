#!/usr/bin/env python3
"""
Monitoring Setup Script for Universal Knowledge Platform
Verifies and configures Prometheus metrics, logging, and monitoring.
"""

import requests
import time
import json
import sys
from typing import Dict, Any, Optional

# Configuration
API_BASE_URL = "http://localhost:8002"
METRICS_ENDPOINT = f"{API_BASE_URL}/metrics"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
INTEGRATIONS_ENDPOINT = f"{API_BASE_URL}/integrations"

def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ API Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def check_metrics_endpoint() -> bool:
    """Check if metrics endpoint is working."""
    try:
        response = requests.get(METRICS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            content = response.text
            print("✅ Metrics endpoint is working")
            
            # Check for key metrics
            key_metrics = [
                "ukp_requests_total",
                "ukp_request_duration_seconds",
                "ukp_errors_total",
                "ukp_cache_hits_total",
                "ukp_cache_misses_total"
            ]
            
            found_metrics = []
            for metric in key_metrics:
                if metric in content:
                    found_metrics.append(metric)
            
            print(f"📊 Found {len(found_metrics)}/{len(key_metrics)} key metrics:")
            for metric in found_metrics:
                print(f"   - {metric}")
            
            return True
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot access metrics endpoint: {e}")
        return False

def check_integrations() -> bool:
    """Check integration status."""
    try:
        response = requests.get(INTEGRATIONS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            integrations = data.get('integrations', {})
            summary = data.get('summary', {})
            
            print(f"🔗 Integration Status:")
            print(f"   Total: {summary.get('total', 0)}")
            print(f"   Healthy: {summary.get('healthy', 0)}")
            print(f"   Unhealthy: {summary.get('unhealthy', 0)}")
            
            for name, status in integrations.items():
                status_icon = "✅" if status.get('status') == 'healthy' else "❌"
                print(f"   {status_icon} {name}: {status.get('status', 'unknown')}")
            
            return summary.get('healthy', 0) > 0
        else:
            print(f"❌ Integrations endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot access integrations endpoint: {e}")
        return False

def generate_test_metrics() -> bool:
    """Generate some test metrics by making API calls."""
    print("\n🧪 Generating test metrics...")
    
    test_endpoints = [
        ("GET", "/health"),
        ("GET", "/metrics"),
        ("GET", "/integrations"),
    ]
    
    for method, endpoint in test_endpoints:
        try:
            url = f"{API_BASE_URL}{endpoint}"
            response = requests.request(method, url, timeout=5)
            print(f"   {method} {endpoint} -> {response.status_code}")
        except Exception as e:
            print(f"   {method} {endpoint} -> ERROR: {e}")
    
    return True

def check_prometheus_config() -> bool:
    """Check if Prometheus configuration is valid."""
    try:
        import yaml
        with open('monitoring/prometheus-config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Prometheus configuration is valid")
        
        # Check for UKP backend job
        scrape_configs = config.get('scrape_configs', [])
        ukp_jobs = [job for job in scrape_configs if 'ukp' in job.get('job_name', '')]
        
        if ukp_jobs:
            print(f"📊 Found {len(ukp_jobs)} UKP monitoring jobs:")
            for job in ukp_jobs:
                print(f"   - {job['job_name']}: {job['static_configs'][0]['targets']}")
        else:
            print("⚠️ No UKP monitoring jobs found in Prometheus config")
        
        return True
    except Exception as e:
        print(f"❌ Prometheus configuration error: {e}")
        return False

def setup_logging_config() -> bool:
    """Verify logging configuration."""
    print("\n📝 Checking logging configuration...")
    
    # Check if structured logging is configured
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        # Test structured logging
        logger.info("Test structured log message", extra={
            "test": True,
            "component": "monitoring_setup"
        })
        
        print("✅ Logging configuration appears to be working")
        return True
    except Exception as e:
        print(f"❌ Logging configuration error: {e}")
        return False

def main():
    """Main monitoring setup function."""
    print("🔧 UKP Monitoring Setup")
    print("=" * 50)
    
    checks = []
    
    # Check API health
    print("\n1. Checking API Health...")
    checks.append(check_api_health())
    
    # Check metrics endpoint
    print("\n2. Checking Metrics Endpoint...")
    checks.append(check_metrics_endpoint())
    
    # Check integrations
    print("\n3. Checking Integrations...")
    checks.append(check_integrations())
    
    # Generate test metrics
    print("\n4. Generating Test Metrics...")
    checks.append(generate_test_metrics())
    
    # Check Prometheus config
    print("\n5. Checking Prometheus Configuration...")
    checks.append(check_prometheus_config())
    
    # Check logging
    print("\n6. Checking Logging Configuration...")
    checks.append(setup_logging_config())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Monitoring Setup Summary")
    print("=" * 50)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All monitoring checks passed!")
        print("\n📋 Next Steps:")
        print("1. Start Prometheus with the provided config")
        print("2. Set up Grafana dashboards")
        print("3. Configure alerts for critical metrics")
        print("4. Monitor the /metrics endpoint for real-time data")
    else:
        print("\n⚠️ Some checks failed. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 