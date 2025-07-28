"""
Integration Layer - MAANG Standards.

This module serves as the central integration layer that connects all
MAANG-level components into a cohesive, production-ready system.

Features:
    - Component lifecycle management
    - Dependency injection and initialization
    - Health check orchestration
    - Service discovery and registration
    - Cross-component communication
    - System-wide monitoring
    - Graceful startup and shutdown
    - Configuration validation
    - Error handling and recovery

Integration Components:
    - Security integration
    - Analytics integration
    - ML integration
    - Real-time integration
    - Monitoring integration
    - Caching integration
    - Database integration
    - API integration

Authors:
    - Universal Knowledge Platform Engineering Team
    
Version:
    2.0.0 (2024-12-28)
"""

import asyncio
import signal
import sys
from typing import (
    Optional, Dict, Any, List, Union, Callable,
    TypeVar, Protocol, Tuple, Set
)
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import structlog
from contextlib import asynccontextmanager

# Import all MAANG components
from api.config import get_settings
from api.exceptions import UniversalKnowledgeError
from api.monitoring import get_monitoring_manager
from api.security import get_security_manager
from api.cache import get_cache_manager
from api.rate_limiter import get_rate_limiter
from api.performance import get_performance_monitor
from api.analytics_v2 import get_analytics_processor
from api.ml_integration import get_model_manager, get_nlp_models
from api.realtime import (
    get_connection_manager, get_stream_processor,
    get_collaboration_manager, get_notification_manager
)
from api.user_management_v2 import get_user_manager
from api.database.models import get_database_manager
from api.versioning_v2 import get_version_manager

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar('T')

# System states
class SystemState(str, Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class ComponentStatus(str, Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

# Component information
@dataclass
class ComponentInfo:
    """Component information and status."""
    
    name: str
    version: str
    status: ComponentStatus
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    error_message: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    health_check: Optional[Callable] = None
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == ComponentStatus.HEALTHY
    
    def is_operational(self) -> bool:
        """Check if component is operational."""
        return self.status in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]

# System integration manager
class SystemIntegrationManager:
    """
    Central integration manager for all MAANG components.
    
    Features:
    - Component lifecycle management
    - Dependency resolution
    - Health monitoring
    - Error recovery
    - Graceful shutdown
    """
    
    def __init__(self):
        """Initialize system integration manager."""
        self.settings = get_settings()
        self.state = SystemState.INITIALIZING
        self.components: Dict[str, ComponentInfo] = {}
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Initialize component registry
        self._register_components()
    
    def _register_components(self) -> None:
        """Register all system components."""
        components = [
            # Core infrastructure
            ComponentInfo(
                name="config",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=[],
                health_check=self._check_config_health
            ),
            ComponentInfo(
                name="security",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config"],
                health_check=self._check_security_health
            ),
            ComponentInfo(
                name="monitoring",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config"],
                health_check=self._check_monitoring_health
            ),
            ComponentInfo(
                name="cache",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config"],
                health_check=self._check_cache_health
            ),
            ComponentInfo(
                name="database",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config"],
                health_check=self._check_database_health
            ),
            ComponentInfo(
                name="user_management",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config", "security", "database"],
                health_check=self._check_user_management_health
            ),
            ComponentInfo(
                name="rate_limiter",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config", "cache"],
                health_check=self._check_rate_limiter_health
            ),
            ComponentInfo(
                name="performance_monitor",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config", "monitoring"],
                health_check=self._check_performance_monitor_health
            ),
            ComponentInfo(
                name="analytics",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config", "cache", "monitoring"],
                health_check=self._check_analytics_health
            ),
            ComponentInfo(
                name="ml_integration",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config", "cache", "monitoring"],
                health_check=self._check_ml_integration_health
            ),
            ComponentInfo(
                name="realtime",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config", "cache", "monitoring"],
                health_check=self._check_realtime_health
            ),
            ComponentInfo(
                name="versioning",
                version="2.0.0",
                status=ComponentStatus.UNKNOWN,
                dependencies=["config"],
                health_check=self._check_versioning_health
            )
        ]
        
        for component in components:
            self.components[component.name] = component
        
        # Determine startup order based on dependencies
        self.startup_order = self._resolve_startup_order()
        self.shutdown_order = list(reversed(self.startup_order))
    
    def _resolve_startup_order(self) -> List[str]:
        """Resolve component startup order based on dependencies."""
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(component_name: str) -> None:
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected: {component_name}")
            if component_name in visited:
                return
            
            temp_visited.add(component_name)
            component = self.components[component_name]
            
            for dependency in component.dependencies:
                visit(dependency)
            
            temp_visited.remove(component_name)
            visited.add(component_name)
            order.append(component_name)
        
        for component_name in self.components:
            if component_name not in visited:
                visit(component_name)
        
        return order
    
    async def start_system(self) -> None:
        """Start the entire system with all components."""
        logger.info("Starting Universal Knowledge Platform system")
        self.state = SystemState.STARTING
        
        try:
            # Start components in dependency order
            for component_name in self.startup_order:
                await self._start_component(component_name)
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())
            
            self.state = SystemState.RUNNING
            logger.info("System started successfully", components=len(self.components))
            
        except Exception as e:
            self.state = SystemState.ERROR
            logger.error("System startup failed", error=str(e))
            await self.stop_system()
            raise
    
    async def stop_system(self) -> None:
        """Stop the entire system gracefully."""
        logger.info("Stopping Universal Knowledge Platform system")
        self.state = SystemState.STOPPING
        
        # Stop health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Stop components in reverse dependency order
        for component_name in self.shutdown_order:
            await self._stop_component(component_name)
        
        self.state = SystemState.STOPPED
        logger.info("System stopped successfully")
    
    async def _start_component(self, component_name: str) -> None:
        """Start a specific component."""
        component = self.components[component_name]
        logger.info("Starting component", component=component_name)
        
        try:
            component.start_time = datetime.now(timezone.utc)
            
            # Initialize component based on name
            if component_name == "config":
                # Config is already initialized
                pass
            elif component_name == "security":
                security_manager = get_security_manager()
                # Security is auto-initialized
            elif component_name == "monitoring":
                monitoring_manager = get_monitoring_manager()
                await monitoring_manager.start()
            elif component_name == "cache":
                cache_manager = get_cache_manager()
                await cache_manager.initialize()
            elif component_name == "database":
                database_manager = get_database_manager()
                await database_manager.initialize()
            elif component_name == "user_management":
                user_manager = get_user_manager()
                await user_manager.initialize()
            elif component_name == "rate_limiter":
                rate_limiter = get_rate_limiter()
                await rate_limiter.initialize()
            elif component_name == "performance_monitor":
                performance_monitor = get_performance_monitor()
                performance_monitor.start_monitoring()
            elif component_name == "analytics":
                analytics_processor = get_analytics_processor()
                await analytics_processor.start_processing()
            elif component_name == "ml_integration":
                model_manager = get_model_manager()
                nlp_models = get_nlp_models()
                # ML components are auto-initialized
            elif component_name == "realtime":
                from api.realtime import start_realtime_services
                await start_realtime_services()
            elif component_name == "versioning":
                version_manager = get_version_manager()
                # Versioning is auto-initialized
            
            component.status = ComponentStatus.HEALTHY
            logger.info("Component started successfully", component=component_name)
            
        except Exception as e:
            component.status = ComponentStatus.UNHEALTHY
            component.error_message = str(e)
            logger.error(
                "Component startup failed",
                component=component_name,
                error=str(e)
            )
            raise
    
    async def _stop_component(self, component_name: str) -> None:
        """Stop a specific component."""
        component = self.components[component_name]
        logger.info("Stopping component", component=component_name)
        
        try:
            component.stop_time = datetime.now(timezone.utc)
            
            # Stop component based on name
            if component_name == "monitoring":
                monitoring_manager = get_monitoring_manager()
                await monitoring_manager.stop()
            elif component_name == "cache":
                cache_manager = get_cache_manager()
                await cache_manager.shutdown()
            elif component_name == "database":
                database_manager = get_database_manager()
                await database_manager.shutdown()
            elif component_name == "user_management":
                user_manager = get_user_manager()
                await user_manager.shutdown()
            elif component_name == "rate_limiter":
                rate_limiter = get_rate_limiter()
                await rate_limiter.shutdown()
            elif component_name == "performance_monitor":
                performance_monitor = get_performance_monitor()
                performance_monitor.stop_monitoring()
            elif component_name == "analytics":
                analytics_processor = get_analytics_processor()
                await analytics_processor.stop_processing()
            elif component_name == "realtime":
                from api.realtime import stop_realtime_services
                await stop_realtime_services()
            
            component.status = ComponentStatus.STOPPED
            logger.info("Component stopped successfully", component=component_name)
            
        except Exception as e:
            logger.error(
                "Component shutdown failed",
                component=component_name,
                error=str(e)
            )
    
    async def _health_monitor(self) -> None:
        """Monitor system health continuously."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks for all components."""
        unhealthy_components = []
        
        for component_name, component in self.components.items():
            if component.health_check:
                try:
                    is_healthy = await component.health_check()
                    if is_healthy:
                        component.status = ComponentStatus.HEALTHY
                        component.error_message = None
                    else:
                        component.status = ComponentStatus.DEGRADED
                        unhealthy_components.append(component_name)
                except Exception as e:
                    component.status = ComponentStatus.UNHEALTHY
                    component.error_message = str(e)
                    unhealthy_components.append(component_name)
        
        # Update system state based on component health
        if unhealthy_components:
            self.state = SystemState.DEGRADED
            logger.warning(
                "System health degraded",
                unhealthy_components=unhealthy_components
            )
        else:
            self.state = SystemState.RUNNING
    
    # Health check methods for each component
    async def _check_config_health(self) -> bool:
        """Check configuration health."""
        try:
            settings = get_settings()
            return settings is not None
        except Exception:
            return False
    
    async def _check_security_health(self) -> bool:
        """Check security component health."""
        try:
            security_manager = get_security_manager()
            return security_manager is not None
        except Exception:
            return False
    
    async def _check_monitoring_health(self) -> bool:
        """Check monitoring component health."""
        try:
            monitoring_manager = get_monitoring_manager()
            return monitoring_manager.is_running()
        except Exception:
            return False
    
    async def _check_cache_health(self) -> bool:
        """Check cache component health."""
        try:
            cache_manager = get_cache_manager()
            return await cache_manager.is_healthy()
        except Exception:
            return False
    
    async def _check_database_health(self) -> bool:
        """Check database component health."""
        try:
            database_manager = get_database_manager()
            return await database_manager.is_healthy()
        except Exception:
            return False
    
    async def _check_user_management_health(self) -> bool:
        """Check user management component health."""
        try:
            user_manager = get_user_manager()
            return await user_manager.is_healthy()
        except Exception:
            return False
    
    async def _check_rate_limiter_health(self) -> bool:
        """Check rate limiter component health."""
        try:
            rate_limiter = get_rate_limiter()
            return await rate_limiter.is_healthy()
        except Exception:
            return False
    
    async def _check_performance_monitor_health(self) -> bool:
        """Check performance monitor component health."""
        try:
            performance_monitor = get_performance_monitor()
            return performance_monitor._monitoring
        except Exception:
            return False
    
    async def _check_analytics_health(self) -> bool:
        """Check analytics component health."""
        try:
            analytics_processor = get_analytics_processor()
            return analytics_processor._processing
        except Exception:
            return False
    
    async def _check_ml_integration_health(self) -> bool:
        """Check ML integration component health."""
        try:
            model_manager = get_model_manager()
            nlp_models = get_nlp_models()
            return model_manager is not None and nlp_models is not None
        except Exception:
            return False
    
    async def _check_realtime_health(self) -> bool:
        """Check real-time component health."""
        try:
            connection_manager = get_connection_manager()
            return connection_manager._running
        except Exception:
            return False
    
    async def _check_versioning_health(self) -> bool:
        """Check versioning component health."""
        try:
            version_manager = get_version_manager()
            return version_manager is not None
        except Exception:
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'state': self.state.value,
            'components': {
                name: {
                    'status': component.status.value,
                    'version': component.version,
                    'start_time': component.start_time.isoformat() if component.start_time else None,
                    'error_message': component.error_message,
                    'dependencies': component.dependencies
                }
                for name, component in self.components.items()
            },
            'startup_order': self.startup_order,
            'shutdown_order': self.shutdown_order,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def is_system_healthy(self) -> bool:
        """Check if system is healthy."""
        return all(component.is_healthy() for component in self.components.values())
    
    def is_system_operational(self) -> bool:
        """Check if system is operational."""
        return all(component.is_operational() for component in self.components.values())

# Global system manager
_system_manager: Optional[SystemIntegrationManager] = None

def get_system_manager() -> SystemIntegrationManager:
    """Get global system integration manager instance."""
    global _system_manager
    
    if _system_manager is None:
        _system_manager = SystemIntegrationManager()
    
    return _system_manager

# System lifecycle management
@asynccontextmanager
async def system_lifecycle():
    """Context manager for system lifecycle."""
    manager = get_system_manager()
    
    try:
        await manager.start_system()
        yield manager
    finally:
        await manager.stop_system()

# Signal handlers for graceful shutdown
def setup_signal_handlers(manager: SystemIntegrationManager) -> None:
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(signum: int, frame) -> None:
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        manager._shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# System utilities
async def start_system() -> SystemIntegrationManager:
    """Start the entire system."""
    manager = get_system_manager()
    setup_signal_handlers(manager)
    await manager.start_system()
    return manager

async def stop_system() -> None:
    """Stop the entire system."""
    manager = get_system_manager()
    await manager.stop_system()

def get_system_status() -> Dict[str, Any]:
    """Get system status."""
    manager = get_system_manager()
    return manager.get_system_status()

def is_system_healthy() -> bool:
    """Check if system is healthy."""
    manager = get_system_manager()
    return manager.is_system_healthy()

# Export public API
__all__ = [
    # Classes
    'SystemIntegrationManager',
    'ComponentInfo',
    
    # Enums
    'SystemState',
    'ComponentStatus',
    
    # Functions
    'get_system_manager',
    'system_lifecycle',
    'setup_signal_handlers',
    'start_system',
    'stop_system',
    'get_system_status',
    'is_system_healthy',
] 