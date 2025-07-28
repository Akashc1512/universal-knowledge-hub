#!/usr/bin/env python3
"""
MAANG-Level System Dashboard

Real-time operational dashboard for monitoring the Universal Knowledge Platform
with comprehensive metrics, health status, and performance indicators.

Features:
    - Real-time system monitoring
    - Performance metrics dashboard
    - Security status monitoring
    - Component health tracking
    - Alert management
    - Resource utilization
    - User activity tracking
    - Business metrics

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import structlog
import aiohttp
import psutil
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import curses
import threading
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import MAANG components
from api.integration_layer import get_system_manager, get_system_status, is_system_healthy
from api.config import get_settings
from api.monitoring import get_monitoring_manager
from api.performance import get_performance_monitor
from api.analytics_v2 import get_analytics_processor
from api.ml_integration import get_model_manager
from api.realtime import get_connection_manager
from api.cache import get_cache_manager

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

@dataclass
class SystemMetrics:
    """System metrics for dashboard display."""
    
    # System status
    system_state: str = "unknown"
    uptime: float = 0.0
    health_score: float = 0.0
    
    # Performance metrics
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: float = 0.0
    
    # Component status
    components_healthy: int = 0
    components_total: int = 0
    components_degraded: int = 0
    components_unhealthy: int = 0
    
    # Security metrics
    security_events: int = 0
    threat_detections: int = 0
    failed_logins: int = 0
    
    # Business metrics
    active_users: int = 0
    total_queries: int = 0
    cache_hit_rate: float = 0.0
    ml_predictions: int = 0
    
    # Real-time metrics
    websocket_connections: int = 0
    realtime_messages: int = 0
    collaboration_sessions: int = 0
    
    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class MAANGDashboard:
    """
    Real-time operational dashboard for MAANG-level system monitoring.
    
    Provides comprehensive monitoring of:
    - System health and performance
    - Security status and threats
    - Resource utilization
    - Business metrics
    - Component status
    - Real-time activity
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize dashboard."""
        self.base_url = base_url
        self.metrics_history: deque = deque(maxlen=100)
        self.alerts: List[Dict[str, Any]] = []
        self.running = False
        self.refresh_interval = 2.0  # seconds
        
    async def start_dashboard(self) -> None:
        """Start the dashboard monitoring."""
        logger.info("🚀 Starting MAANG-Level System Dashboard")
        self.running = True
        
        try:
            # Start monitoring loop
            while self.running:
                # Collect metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                # Display dashboard
                self._display_dashboard(metrics)
                
                # Wait for next refresh
                await asyncio.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Dashboard stopped by user")
        except Exception as e:
            logger.error("❌ Dashboard error", error=str(e))
        finally:
            self.running = False
    
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        metrics = SystemMetrics()
        
        try:
            # System status
            system_manager = get_system_manager()
            if system_manager:
                status = system_manager.get_system_status()
                metrics.system_state = status['state']
                metrics.health_score = self._calculate_health_score(status)
                
                # Component counts
                for component in status['components'].values():
                    metrics.components_total += 1
                    if component['status'] == 'healthy':
                        metrics.components_healthy += 1
                    elif component['status'] == 'degraded':
                        metrics.components_degraded += 1
                    else:
                        metrics.components_unhealthy += 1
            
            # Performance metrics
            performance_monitor = get_performance_monitor()
            if performance_monitor:
                summary = performance_monitor.get_performance_summary()
                if summary and 'endpoints' in summary:
                    for endpoint_data in summary['endpoints'].values():
                        metrics.response_time_p50 = max(metrics.response_time_p50, endpoint_data.get('p50_response_time', 0))
                        metrics.response_time_p95 = max(metrics.response_time_p95, endpoint_data.get('p95_response_time', 0))
                        metrics.response_time_p99 = max(metrics.response_time_p99, endpoint_data.get('p99_response_time', 0))
                        metrics.throughput_rps = max(metrics.throughput_rps, endpoint_data.get('throughput', 0))
                        metrics.error_rate = max(metrics.error_rate, endpoint_data.get('error_rate', 0))
            
            # Resource utilization
            metrics.cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            metrics.memory_usage = memory.percent
            disk = psutil.disk_usage('/')
            metrics.disk_usage = (disk.used / disk.total) * 100
            
            # Security metrics
            security_manager = get_security_manager()
            if security_manager:
                threat_stats = security_manager.get_threat_stats()
                metrics.security_events = threat_stats.get('total_events', 0)
                metrics.threat_detections = threat_stats.get('blocked_ips', 0)
            
            # Business metrics
            analytics_processor = get_analytics_processor()
            if analytics_processor:
                summary = await analytics_processor.get_performance_summary()
                if summary:
                    metrics.total_queries = summary.get('total_requests', 0)
            
            # Cache metrics
            cache_manager = get_cache_manager()
            if cache_manager:
                stats = await cache_manager.get_stats()
                if stats:
                    metrics.cache_hit_rate = stats.get('hit_rate', 0)
            
            # Real-time metrics
            connection_manager = get_connection_manager()
            if connection_manager:
                metrics.websocket_connections = len(connection_manager.active_connections)
            
            # ML metrics
            model_manager = get_model_manager()
            if model_manager:
                models = model_manager.get_all_models()
                metrics.ml_predictions = len(models)
            
        except Exception as e:
            logger.error("Error collecting metrics", error=str(e))
        
        return metrics
    
    def _calculate_health_score(self, status: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        if not status or 'components' not in status:
            return 0.0
        
        total_components = len(status['components'])
        if total_components == 0:
            return 0.0
        
        healthy_count = sum(
            1 for comp in status['components'].values()
            if comp['status'] == 'healthy'
        )
        
        return (healthy_count / total_components) * 100
    
    async def _check_alerts(self, metrics: SystemMetrics) -> None:
        """Check for alert conditions."""
        alerts = []
        
        # Performance alerts
        if metrics.response_time_p95 > 500:  # 500ms
            alerts.append({
                'level': 'WARNING',
                'message': f'High response time: {metrics.response_time_p95:.1f}ms',
                'timestamp': datetime.now(timezone.utc)
            })
        
        if metrics.error_rate > 5:  # 5%
            alerts.append({
                'level': 'CRITICAL',
                'message': f'High error rate: {metrics.error_rate:.1f}%',
                'timestamp': datetime.now(timezone.utc)
            })
        
        # Resource alerts
        if metrics.cpu_usage > 80:
            alerts.append({
                'level': 'WARNING',
                'message': f'High CPU usage: {metrics.cpu_usage:.1f}%',
                'timestamp': datetime.now(timezone.utc)
            })
        
        if metrics.memory_usage > 85:
            alerts.append({
                'level': 'WARNING',
                'message': f'High memory usage: {metrics.memory_usage:.1f}%',
                'timestamp': datetime.now(timezone.utc)
            })
        
        # Security alerts
        if metrics.security_events > 10:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'High security events: {metrics.security_events}',
                'timestamp': datetime.now(timezone.utc)
            })
        
        # Health alerts
        if metrics.health_score < 80:
            alerts.append({
                'level': 'WARNING',
                'message': f'Low health score: {metrics.health_score:.1f}%',
                'timestamp': datetime.now(timezone.utc)
            })
        
        # Add new alerts
        self.alerts.extend(alerts)
        
        # Keep only recent alerts (last 50)
        self.alerts = self.alerts[-50:]
    
    def _display_dashboard(self, metrics: SystemMetrics) -> None:
        """Display the dashboard."""
        # Clear screen
        print("\033[2J\033[H")
        
        # Header
        print("=" * 100)
        print("🎯 MAANG-LEVEL UNIVERSAL KNOWLEDGE PLATFORM DASHBOARD")
        print("=" * 100)
        print(f"Timestamp: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Refresh Interval: {self.refresh_interval}s")
        print("=" * 100)
        
        # System Status
        print("\n📊 SYSTEM STATUS")
        print("-" * 50)
        status_icon = "✅" if metrics.system_state == "running" else "❌"
        health_icon = "🟢" if metrics.health_score >= 90 else "🟡" if metrics.health_score >= 70 else "🔴"
        
        print(f"{status_icon} System State: {metrics.system_state.upper()}")
        print(f"{health_icon} Health Score: {metrics.health_score:.1f}%")
        print(f"⏱️  Uptime: {metrics.uptime:.1f}s")
        
        # Component Status
        print(f"\n🔧 COMPONENTS: {metrics.components_healthy}/{metrics.components_total} Healthy")
        print(f"   🟢 Healthy: {metrics.components_healthy}")
        print(f"   🟡 Degraded: {metrics.components_degraded}")
        print(f"   🔴 Unhealthy: {metrics.components_unhealthy}")
        
        # Performance Metrics
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 50)
        print(f"📈 Response Time P50: {metrics.response_time_p50:.1f}ms")
        print(f"📈 Response Time P95: {metrics.response_time_p95:.1f}ms")
        print(f"📈 Response Time P99: {metrics.response_time_p99:.1f}ms")
        print(f"🚀 Throughput: {metrics.throughput_rps:.1f} req/s")
        print(f"❌ Error Rate: {metrics.error_rate:.2f}%")
        
        # Resource Utilization
        print("\n💻 RESOURCE UTILIZATION")
        print("-" * 50)
        cpu_icon = "🟢" if metrics.cpu_usage < 70 else "🟡" if metrics.cpu_usage < 90 else "🔴"
        memory_icon = "🟢" if metrics.memory_usage < 70 else "🟡" if metrics.memory_usage < 90 else "🔴"
        disk_icon = "🟢" if metrics.disk_usage < 70 else "🟡" if metrics.disk_usage < 90 else "🔴"
        
        print(f"{cpu_icon} CPU Usage: {metrics.cpu_usage:.1f}%")
        print(f"{memory_icon} Memory Usage: {metrics.memory_usage:.1f}%")
        print(f"{disk_icon} Disk Usage: {metrics.disk_usage:.1f}%")
        print(f"🌐 Network I/O: {metrics.network_io:.1f} MB/s")
        
        # Security Metrics
        print("\n🛡️ SECURITY METRICS")
        print("-" * 50)
        security_icon = "🟢" if metrics.security_events == 0 else "🟡" if metrics.security_events < 5 else "🔴"
        print(f"{security_icon} Security Events: {metrics.security_events}")
        print(f"🚫 Threat Detections: {metrics.threat_detections}")
        print(f"🔐 Failed Logins: {metrics.failed_logins}")
        
        # Business Metrics
        print("\n📈 BUSINESS METRICS")
        print("-" * 50)
        print(f"👥 Active Users: {metrics.active_users}")
        print(f"🔍 Total Queries: {metrics.total_queries}")
        print(f"💾 Cache Hit Rate: {metrics.cache_hit_rate:.1f}%")
        print(f"🤖 ML Predictions: {metrics.ml_predictions}")
        
        # Real-time Metrics
        print("\n⚡ REAL-TIME METRICS")
        print("-" * 50)
        print(f"🔌 WebSocket Connections: {metrics.websocket_connections}")
        print(f"💬 Real-time Messages: {metrics.realtime_messages}")
        print(f"👥 Collaboration Sessions: {metrics.collaboration_sessions}")
        
        # Alerts
        if self.alerts:
            print("\n🚨 RECENT ALERTS")
            print("-" * 50)
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                level_icon = "🔴" if alert['level'] == 'CRITICAL' else "🟡" if alert['level'] == 'WARNING' else "🔵"
                timestamp = alert['timestamp'].strftime('%H:%M:%S')
                print(f"{level_icon} [{timestamp}] {alert['level']}: {alert['message']}")
        
        # Footer
        print("\n" + "=" * 100)
        print("Press Ctrl+C to stop dashboard | MAANG-Level System Monitoring")
        print("=" * 100)
    
    def export_metrics(self, filename: str) -> None:
        """Export metrics to JSON file."""
        try:
            data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'system_state': m.system_state,
                        'health_score': m.health_score,
                        'response_time_p95': m.response_time_p95,
                        'throughput_rps': m.throughput_rps,
                        'error_rate': m.error_rate,
                        'cpu_usage': m.cpu_usage,
                        'memory_usage': m.memory_usage,
                        'security_events': m.security_events,
                        'active_users': m.active_users,
                        'total_queries': m.total_queries
                    }
                    for m in self.metrics_history
                ],
                'alerts': [
                    {
                        'level': a['level'],
                        'message': a['message'],
                        'timestamp': a['timestamp'].isoformat()
                    }
                    for a in self.alerts
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")

async def main():
    """Main dashboard function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MAANG-Level System Dashboard")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the API"
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=2.0,
        help="Dashboard refresh interval in seconds"
    )
    parser.add_argument(
        "--export",
        help="Export metrics to JSON file"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no display)"
    )
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = MAANGDashboard(base_url=args.base_url)
    dashboard.refresh_interval = args.refresh_interval
    
    try:
        if args.headless:
            # Headless mode - just collect metrics
            logger.info("Running in headless mode")
            for _ in range(10):  # Collect 10 data points
                metrics = await dashboard._collect_metrics()
                dashboard.metrics_history.append(metrics)
                await asyncio.sleep(dashboard.refresh_interval)
            
            if args.export:
                dashboard.export_metrics(args.export)
        else:
            # Interactive mode
            await dashboard.start_dashboard()
            
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error("Dashboard error", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 