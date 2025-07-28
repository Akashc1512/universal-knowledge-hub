#!/usr/bin/env python3
"""
MAANG-Level Health Monitor

Comprehensive health monitoring and alerting system for the Universal Knowledge Platform
with real-time status tracking, automated alerts, and proactive issue detection.

Features:
    - Real-time health monitoring
    - Automated alerting system
    - Proactive issue detection
    - Performance degradation alerts
    - Security incident monitoring
    - Resource utilization tracking
    - SLA/SLO monitoring
    - Incident response automation

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import asyncio
import sys
import time
import json
import smtplib
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import structlog
import aiohttp
import psutil
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
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
from api.security import get_security_manager
from api.cache import get_cache_manager
from api.analytics_v2 import get_analytics_processor
from api.ml_integration import get_model_manager
from api.realtime import get_connection_manager

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

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(str, Enum):
    """Types of alerts."""
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"
    RESOURCE = "resource"
    BUSINESS = "business"
    COMPONENT = "component"

@dataclass
class HealthMetric:
    """Health metric for monitoring."""
    
    name: str
    value: float
    unit: str
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    
    def is_warning(self) -> bool:
        """Check if metric is in warning state."""
        return self.value >= self.threshold_warning
    
    def is_critical(self) -> bool:
        """Check if metric is in critical state."""
        return self.value >= self.threshold_critical

@dataclass
class Alert:
    """Alert for system issues."""
    
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class MAANGHealthMonitor:
    """
    Comprehensive health monitoring system for MAANG-level components.
    
    Features:
    - Real-time health monitoring
    - Automated alerting
    - Proactive issue detection
    - Performance degradation alerts
    - Security incident monitoring
    - Resource utilization tracking
    - SLA/SLO monitoring
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize health monitor."""
        self.base_url = base_url
        self.running = False
        self.check_interval = 30  # seconds
        self.alerts: List[Alert] = []
        self.metrics_history: deque = deque(maxlen=1000)
        self.alert_handlers: Dict[AlertType, List[Callable]] = {
            AlertType.PERFORMANCE: [self._handle_performance_alert],
            AlertType.SECURITY: [self._handle_security_alert],
            AlertType.AVAILABILITY: [self._handle_availability_alert],
            AlertType.RESOURCE: [self._handle_resource_alert],
            AlertType.BUSINESS: [self._handle_business_alert],
            AlertType.COMPONENT: [self._handle_component_alert],
        }
        self.sla_targets = {
            'response_time_p95': 500,  # ms
            'availability': 99.9,      # %
            'error_rate': 1.0,         # %
            'uptime': 99.9,            # %
        }
        
    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        logger.info("🏥 Starting MAANG-Level Health Monitor")
        self.running = True
        
        try:
            while self.running:
                # Collect health metrics
                metrics = await self._collect_health_metrics()
                self.metrics_history.append(metrics)
                
                # Check for issues
                await self._check_health_issues(metrics)
                
                # Process alerts
                await self._process_alerts()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
        except Exception as e:
            logger.error("❌ Health monitor error", error=str(e))
            raise
    
    async def _collect_health_metrics(self) -> Dict[str, HealthMetric]:
        """Collect comprehensive health metrics."""
        metrics = {}
        
        try:
            # System health
            system_manager = get_system_manager()
            if system_manager:
                status = system_manager.get_system_status()
                health_score = self._calculate_health_score(status)
                
                metrics['system_health'] = HealthMetric(
                    name="System Health Score",
                    value=health_score,
                    unit="%",
                    threshold_warning=80.0,
                    threshold_critical=60.0,
                    description="Overall system health score"
                )
                
                # Component health
                healthy_components = sum(
                    1 for comp in status['components'].values()
                    if comp['status'] == 'healthy'
                )
                total_components = len(status['components'])
                
                metrics['component_health'] = HealthMetric(
                    name="Component Health",
                    value=(healthy_components / total_components * 100) if total_components > 0 else 0,
                    unit="%",
                    threshold_warning=80.0,
                    threshold_critical=60.0,
                    description="Percentage of healthy components"
                )
            
            # Performance metrics
            performance_monitor = get_performance_monitor()
            if performance_monitor:
                summary = performance_monitor.get_performance_summary()
                if summary and 'endpoints' in summary:
                    max_response_time = max(
                        endpoint_data.get('p95_response_time', 0)
                        for endpoint_data in summary['endpoints'].values()
                    )
                    
                    metrics['response_time'] = HealthMetric(
                        name="Response Time P95",
                        value=max_response_time,
                        unit="ms",
                        threshold_warning=300.0,
                        threshold_critical=500.0,
                        description="95th percentile response time"
                    )
                    
                    max_error_rate = max(
                        endpoint_data.get('error_rate', 0)
                        for endpoint_data in summary['endpoints'].values()
                    )
                    
                    metrics['error_rate'] = HealthMetric(
                        name="Error Rate",
                        value=max_error_rate,
                        unit="%",
                        threshold_warning=1.0,
                        threshold_critical=5.0,
                        description="Error rate percentage"
                    )
            
            # Resource utilization
            cpu_usage = psutil.cpu_percent(interval=1)
            metrics['cpu_usage'] = HealthMetric(
                name="CPU Usage",
                value=cpu_usage,
                unit="%",
                threshold_warning=70.0,
                threshold_critical=90.0,
                description="CPU utilization percentage"
            )
            
            memory = psutil.virtual_memory()
            metrics['memory_usage'] = HealthMetric(
                name="Memory Usage",
                value=memory.percent,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
                description="Memory utilization percentage"
            )
            
            disk = psutil.disk_usage('/')
            metrics['disk_usage'] = HealthMetric(
                name="Disk Usage",
                value=(disk.used / disk.total) * 100,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
                description="Disk utilization percentage"
            )
            
            # Security metrics
            security_manager = get_security_manager()
            if security_manager:
                threat_stats = security_manager.get_threat_stats()
                security_events = threat_stats.get('total_events', 0)
                
                metrics['security_events'] = HealthMetric(
                    name="Security Events",
                    value=security_events,
                    unit="count",
                    threshold_warning=10.0,
                    threshold_critical=50.0,
                    description="Number of security events"
                )
            
            # Cache metrics
            cache_manager = get_cache_manager()
            if cache_manager:
                stats = await cache_manager.get_stats()
                if stats:
                    hit_rate = stats.get('hit_rate', 0)
                    metrics['cache_hit_rate'] = HealthMetric(
                        name="Cache Hit Rate",
                        value=hit_rate,
                        unit="%",
                        threshold_warning=60.0,
                        threshold_critical=40.0,
                        description="Cache hit rate percentage"
                    )
            
            # Real-time metrics
            connection_manager = get_connection_manager()
            if connection_manager:
                active_connections = len(connection_manager.active_connections)
                metrics['websocket_connections'] = HealthMetric(
                    name="WebSocket Connections",
                    value=active_connections,
                    unit="count",
                    threshold_warning=1000.0,
                    threshold_critical=5000.0,
                    description="Active WebSocket connections"
                )
            
        except Exception as e:
            logger.error("Error collecting health metrics", error=str(e))
        
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
    
    async def _check_health_issues(self, metrics: Dict[str, HealthMetric]) -> None:
        """Check for health issues and generate alerts."""
        for metric_name, metric in metrics.items():
            if metric.is_critical():
                await self._create_alert(
                    AlertType.PERFORMANCE if 'response_time' in metric_name or 'error_rate' in metric_name
                    else AlertType.RESOURCE if 'usage' in metric_name
                    else AlertType.SECURITY if 'security' in metric_name
                    else AlertType.COMPONENT,
                    AlertSeverity.CRITICAL,
                    f"Critical {metric.name}",
                    f"{metric.name} is at critical level: {metric.value}{metric.unit}",
                    metric_name
                )
            elif metric.is_warning():
                await self._create_alert(
                    AlertType.PERFORMANCE if 'response_time' in metric_name or 'error_rate' in metric_name
                    else AlertType.RESOURCE if 'usage' in metric_name
                    else AlertType.SECURITY if 'security' in metric_name
                    else AlertType.COMPONENT,
                    AlertSeverity.WARNING,
                    f"Warning {metric.name}",
                    f"{metric.name} is at warning level: {metric.value}{metric.unit}",
                    metric_name
                )
    
    async def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        component: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Create a new alert."""
        alert_id = f"{alert_type.value}_{component}_{int(time.time())}"
        
        # Check if similar alert already exists
        existing_alert = next(
            (a for a in self.alerts 
             if a.type == alert_type and a.component == component and not a.resolved),
            None
        )
        
        if existing_alert:
            # Update existing alert
            existing_alert.message = message
            existing_alert.timestamp = datetime.now(timezone.utc)
            existing_alert.metadata.update(metadata or {})
        else:
            # Create new alert
            alert = Alert(
                id=alert_id,
                type=alert_type,
                severity=severity,
                title=title,
                message=message,
                component=component,
                metadata=metadata or {}
            )
            self.alerts.append(alert)
            
            logger.warning(
                "Alert created",
                alert_id=alert_id,
                type=alert_type.value,
                severity=severity.value,
                component=component,
                message=message
            )
    
    async def _process_alerts(self) -> None:
        """Process and handle alerts."""
        for alert in self.alerts:
            if not alert.resolved and not alert.acknowledged:
                # Handle alert based on type
                handlers = self.alert_handlers.get(alert.type, [])
                for handler in handlers:
                    try:
                        await handler(alert)
                    except Exception as e:
                        logger.error(
                            "Error handling alert",
                            alert_id=alert.id,
                            error=str(e)
                        )
    
    async def _handle_performance_alert(self, alert: Alert) -> None:
        """Handle performance alerts."""
        logger.warning(
            "Performance alert",
            alert_id=alert.id,
            component=alert.component,
            message=alert.message
        )
        
        # Auto-scale if needed
        if alert.severity == AlertSeverity.CRITICAL:
            await self._trigger_auto_scaling()
    
    async def _handle_security_alert(self, alert: Alert) -> None:
        """Handle security alerts."""
        logger.critical(
            "Security alert",
            alert_id=alert.id,
            component=alert.component,
            message=alert.message
        )
        
        # Immediate response for security alerts
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            await self._trigger_security_response(alert)
    
    async def _handle_availability_alert(self, alert: Alert) -> None:
        """Handle availability alerts."""
        logger.error(
            "Availability alert",
            alert_id=alert.id,
            component=alert.component,
            message=alert.message
        )
        
        # Check if system is down
        if "system_health" in alert.component:
            await self._trigger_failover()
    
    async def _handle_resource_alert(self, alert: Alert) -> None:
        """Handle resource alerts."""
        logger.warning(
            "Resource alert",
            alert_id=alert.id,
            component=alert.component,
            message=alert.message
        )
        
        # Scale resources if needed
        if alert.severity == AlertSeverity.CRITICAL:
            await self._trigger_resource_scaling()
    
    async def _handle_business_alert(self, alert: Alert) -> None:
        """Handle business alerts."""
        logger.warning(
            "Business alert",
            alert_id=alert.id,
            component=alert.component,
            message=alert.message
        )
    
    async def _handle_component_alert(self, alert: Alert) -> None:
        """Handle component alerts."""
        logger.warning(
            "Component alert",
            alert_id=alert.id,
            component=alert.component,
            message=alert.message
        )
        
        # Restart component if needed
        if alert.severity == AlertSeverity.CRITICAL:
            await self._restart_component(alert.component)
    
    async def _trigger_auto_scaling(self) -> None:
        """Trigger auto-scaling for performance issues."""
        logger.info("Triggering auto-scaling for performance")
        # Implementation would integrate with Kubernetes HPA
        pass
    
    async def _trigger_security_response(self, alert: Alert) -> None:
        """Trigger security response for security alerts."""
        logger.critical("Triggering security response", alert_id=alert.id)
        # Implementation would include:
        # - Block suspicious IPs
        # - Increase monitoring
        # - Notify security team
        pass
    
    async def _trigger_failover(self) -> None:
        """Trigger failover for availability issues."""
        logger.critical("Triggering failover for availability")
        # Implementation would include:
        # - Switch to backup systems
        # - Notify operations team
        pass
    
    async def _trigger_resource_scaling(self) -> None:
        """Trigger resource scaling for resource issues."""
        logger.info("Triggering resource scaling")
        # Implementation would include:
        # - Scale up resources
        # - Optimize resource usage
        pass
    
    async def _restart_component(self, component: str) -> None:
        """Restart a specific component."""
        logger.info("Restarting component", component=component)
        # Implementation would include:
        # - Graceful shutdown
        # - Health check
        # - Restart
        pass
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        active_alerts = [a for a in self.alerts if not a.resolved]
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        warning_alerts = [a for a in active_alerts if a.severity == AlertSeverity.WARNING]
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_status": "healthy" if not critical_alerts else "degraded" if not warning_alerts else "warning",
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "alerts_by_type": {
                alert_type.value: len([a for a in active_alerts if a.type == alert_type])
                for alert_type in AlertType
            },
            "recent_alerts": [
                {
                    "id": a.id,
                    "type": a.type.value,
                    "severity": a.severity.value,
                    "title": a.title,
                    "component": a.component,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in active_alerts[-10:]  # Last 10 alerts
            ]
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info("Alert acknowledged", alert_id=alert_id)
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info("Alert resolved", alert_id=alert_id)
                return True
        return False

async def main():
    """Main health monitor function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MAANG-Level Health Monitor")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the API"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Health check interval in seconds"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show health summary and exit"
    )
    
    args = parser.parse_args()
    
    # Create health monitor
    monitor = MAANGHealthMonitor(base_url=args.base_url)
    monitor.check_interval = args.check_interval
    
    try:
        if args.summary:
            # Show health summary
            metrics = await monitor._collect_health_metrics()
            summary = monitor.get_health_summary()
            
            print("\n" + "=" * 80)
            print("🏥 MAANG-LEVEL HEALTH SUMMARY")
            print("=" * 80)
            print(f"System Status: {summary['system_status'].upper()}")
            print(f"Active Alerts: {summary['active_alerts']}")
            print(f"Critical Alerts: {summary['critical_alerts']}")
            print(f"Warning Alerts: {summary['warning_alerts']}")
            print("\nRecent Metrics:")
            
            for metric_name, metric in metrics.items():
                status_icon = "🟢" if not metric.is_warning() else "🟡" if not metric.is_critical() else "🔴"
                print(f"  {status_icon} {metric.name}: {metric.value}{metric.unit}")
            
            print("=" * 80)
        else:
            # Start monitoring
            await monitor.start_monitoring()
            
    except KeyboardInterrupt:
        logger.info("Health monitor stopped by user")
    except Exception as e:
        logger.error("Health monitor error", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 