"""
Observer Pattern implementation for event-driven architecture.

This module implements the Observer Pattern following SOLID principles:
- Single Responsibility: Each observer handles specific events
- Open/Closed: New observers can be added without modifying existing code
- Liskov Substitution: All observers can be used interchangeably
- Interface Segregation: Specific interfaces for different observer types
- Dependency Inversion: Depend on observer interfaces, not implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
from collections import defaultdict
import weakref

from .interfaces import Event, EventType, Observer, Subject

logger = logging.getLogger(__name__)


# ============================================================================
# EVENT IMPLEMENTATION
# ============================================================================


@dataclass
class BaseEvent:
    """
    Base event implementation.
    Single Responsibility: Encapsulate event data.
    """
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'source': self.source,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata
        }


@dataclass
class QueryEvent(BaseEvent):
    """Event for query operations."""
    query_id: str = ""
    query_text: str = ""
    user_id: Optional[str] = None
    
    def __post_init__(self):
        self.event_type = EventType.SEARCHED
        self.data.update({
            'query_id': self.query_id,
            'query_text': self.query_text,
            'user_id': self.user_id
        })


@dataclass
class ErrorEvent(BaseEvent):
    """Event for error occurrences."""
    error_type: str = ""
    error_message: str = ""
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        self.event_type = EventType.ERROR
        self.data.update({
            'error_type': self.error_type,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace
        })


@dataclass
class PerformanceEvent(BaseEvent):
    """Event for performance metrics."""
    operation: str = ""
    duration_ms: float = 0.0
    success: bool = True
    
    def __post_init__(self):
        self.event_type = EventType.PERFORMANCE
        self.data.update({
            'operation': self.operation,
            'duration_ms': self.duration_ms,
            'success': self.success
        })


# ============================================================================
# OBSERVER IMPLEMENTATIONS - Single Responsibility for each type
# ============================================================================


class BaseObserver(ABC):
    """
    Base observer with common functionality.
    Template Method Pattern for observation logic.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.events_handled = 0
        self.last_event_time = None
    
    async def update(self, event: Event) -> None:
        """
        Handle event notification.
        Template method with hooks for customization.
        """
        if not self.can_handle(event):
            return
        
        try:
            # Pre-processing
            await self._before_handle(event)
            
            # Core handling
            await self._handle_event(event)
            
            # Post-processing
            await self._after_handle(event)
            
            # Update metrics
            self.events_handled += 1
            self.last_event_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Observer {self.name} error handling event: {e}")
            await self._handle_error(event, e)
    
    @abstractmethod
    async def _handle_event(self, event: Event) -> None:
        """Handle the event - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if observer can handle this event type."""
        pass
    
    async def _before_handle(self, event: Event) -> None:
        """Hook called before handling event."""
        logger.debug(f"Observer {self.name} handling {event.event_type.value}")
    
    async def _after_handle(self, event: Event) -> None:
        """Hook called after handling event."""
        pass
    
    async def _handle_error(self, event: Event, error: Exception) -> None:
        """Hook called on error."""
        logger.error(f"Error in {self.name}: {error}")


class LoggingObserver(BaseObserver):
    """
    Observer for logging events.
    Single Responsibility: Log events to appropriate destinations.
    """
    
    def __init__(self, name: str = "LoggingObserver", log_level: str = "INFO"):
        super().__init__(name)
        self.log_level = getattr(logging, log_level.upper())
    
    async def _handle_event(self, event: Event) -> None:
        """Log event details."""
        log_message = (
            f"Event: {event.event_type.value} | "
            f"Source: {event.source} | "
            f"Data: {event.data}"
        )
        
        if event.event_type == EventType.ERROR:
            logger.error(log_message)
        elif event.event_type == EventType.PERFORMANCE:
            logger.info(log_message)
        else:
            logger.log(self.log_level, log_message)
    
    def can_handle(self, event: Event) -> bool:
        """Logging observer handles all events."""
        return True


class MetricsObserver(BaseObserver):
    """
    Observer for collecting metrics.
    Single Responsibility: Update metrics based on events.
    """
    
    def __init__(self, name: str = "MetricsObserver"):
        super().__init__(name)
        self.metrics = defaultdict(lambda: defaultdict(float))
    
    async def _handle_event(self, event: Event) -> None:
        """Update metrics based on event."""
        event_type = event.event_type.value
        
        # Count events by type
        self.metrics['event_counts'][event_type] += 1
        
        # Handle specific event types
        if event.event_type == EventType.PERFORMANCE:
            operation = event.data.get('operation', 'unknown')
            duration = event.data.get('duration_ms', 0)
            
            self.metrics['operation_times'][operation] += duration
            self.metrics['operation_counts'][operation] += 1
            
            # Calculate average
            count = self.metrics['operation_counts'][operation]
            total = self.metrics['operation_times'][operation]
            self.metrics['operation_avg'][operation] = total / count
        
        elif event.event_type == EventType.ERROR:
            error_type = event.data.get('error_type', 'unknown')
            self.metrics['error_counts'][error_type] += 1
    
    def can_handle(self, event: Event) -> bool:
        """Metrics observer handles performance and error events."""
        return event.event_type in [EventType.PERFORMANCE, EventType.ERROR]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        return dict(self.metrics)


class AlertingObserver(BaseObserver):
    """
    Observer for sending alerts on critical events.
    Single Responsibility: Send alerts for critical conditions.
    """
    
    def __init__(
        self,
        name: str = "AlertingObserver",
        alert_threshold: int = 5,
        time_window_seconds: int = 60
    ):
        super().__init__(name)
        self.alert_threshold = alert_threshold
        self.time_window_seconds = time_window_seconds
        self.error_timestamps: List[datetime] = []
        self.alerts_sent = 0
    
    async def _handle_event(self, event: Event) -> None:
        """Check if alert should be sent."""
        if event.event_type == EventType.ERROR:
            now = datetime.now()
            self.error_timestamps.append(now)
            
            # Remove old timestamps outside window
            cutoff = now.timestamp() - self.time_window_seconds
            self.error_timestamps = [
                ts for ts in self.error_timestamps
                if ts.timestamp() > cutoff
            ]
            
            # Check if threshold exceeded
            if len(self.error_timestamps) >= self.alert_threshold:
                await self._send_alert(event)
                self.error_timestamps.clear()  # Reset after alert
    
    async def _send_alert(self, event: Event) -> None:
        """Send alert notification."""
        alert_message = (
            f"ALERT: {len(self.error_timestamps)} errors "
            f"in {self.time_window_seconds} seconds\n"
            f"Latest error: {event.data.get('error_message', 'Unknown')}"
        )
        
        logger.critical(alert_message)
        self.alerts_sent += 1
        
        # In production, send to alerting service (PagerDuty, Slack, etc.)
    
    def can_handle(self, event: Event) -> bool:
        """Alerting observer handles error events."""
        return event.event_type == EventType.ERROR


class PersistenceObserver(BaseObserver):
    """
    Observer for persisting events to storage.
    Single Responsibility: Store events for audit/analysis.
    """
    
    def __init__(self, name: str = "PersistenceObserver", batch_size: int = 100):
        super().__init__(name)
        self.batch_size = batch_size
        self.event_buffer: List[Event] = []
    
    async def _handle_event(self, event: Event) -> None:
        """Buffer event for batch persistence."""
        self.event_buffer.append(event)
        
        if len(self.event_buffer) >= self.batch_size:
            await self._flush_buffer()
    
    async def _flush_buffer(self) -> None:
        """Persist buffered events to storage."""
        if not self.event_buffer:
            return
        
        try:
            # In production, save to database/file/cloud storage
            logger.info(f"Persisting {len(self.event_buffer)} events")
            
            # Simulate persistence
            await asyncio.sleep(0.1)
            
            self.event_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to persist events: {e}")
    
    def can_handle(self, event: Event) -> bool:
        """Persistence observer handles all events."""
        return True
    
    async def __del__(self):
        """Ensure buffer is flushed on cleanup."""
        await self._flush_buffer()


# ============================================================================
# SUBJECT IMPLEMENTATION - Event publisher
# ============================================================================


class EventBus(Subject):
    """
    Event bus implementation of Subject.
    Open/Closed: New event types and observers can be added without modification.
    """
    
    def __init__(self):
        # Use weak references to avoid memory leaks
        self._observers: Set[weakref.ref] = set()
        self._event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._async_mode = True
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
    
    def attach(self, observer: Observer) -> None:
        """Attach observer to event bus."""
        self._observers.add(weakref.ref(observer))
        logger.info(f"Attached observer: {observer}")
    
    def detach(self, observer: Observer) -> None:
        """Detach observer from event bus."""
        self._observers.discard(weakref.ref(observer))
        logger.info(f"Detached observer: {observer}")
    
    async def notify(self, event: Event) -> None:
        """
        Notify all observers of event.
        Implements async notification for better performance.
        """
        # Clean up dead references
        self._observers = {ref for ref in self._observers if ref() is not None}
        
        # Notify observers
        if self._async_mode:
            # Async parallel notification
            tasks = []
            for observer_ref in self._observers:
                observer = observer_ref()
                if observer and observer.can_handle(event):
                    tasks.append(observer.update(event))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential notification
            for observer_ref in self._observers:
                observer = observer_ref()
                if observer and observer.can_handle(event):
                    await observer.update(event)
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]):
        """
        Subscribe to specific event type with handler function.
        Provides alternative to observer classes.
        """
        self._event_handlers[event_type].append(handler)
    
    async def publish(self, event: Event) -> None:
        """
        Publish event to bus.
        Queues event for processing.
        """
        await self._event_queue.put(event)
        
        # Start processing if not already running
        if not self._processing:
            asyncio.create_task(self._process_events())
    
    async def _process_events(self) -> None:
        """Process queued events."""
        self._processing = True
        
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                    
                    # Notify observers
                    await self.notify(event)
                    
                    # Call type-specific handlers
                    handlers = self._event_handlers.get(event.event_type, [])
                    for handler in handlers:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")
                    
                except asyncio.TimeoutError:
                    # No events, check if should continue
                    if self._event_queue.empty():
                        break
                        
        finally:
            self._processing = False


# ============================================================================
# SPECIALIZED EVENT BUSES - Single Responsibility
# ============================================================================


class PerformanceEventBus(EventBus):
    """
    Specialized event bus for performance monitoring.
    Single Responsibility: Handle performance-related events.
    """
    
    def __init__(self):
        super().__init__()
        self.performance_thresholds = {
            'query': 1000,  # ms
            'retrieval': 500,
            'synthesis': 2000
        }
    
    async def notify(self, event: Event) -> None:
        """Enhanced notification with performance checks."""
        if event.event_type == EventType.PERFORMANCE:
            operation = event.data.get('operation')
            duration = event.data.get('duration_ms', 0)
            
            # Check against thresholds
            threshold = self.performance_thresholds.get(operation)
            if threshold and duration > threshold:
                # Create alert event
                alert_event = ErrorEvent(
                    source=event.source,
                    error_type="performance_threshold_exceeded",
                    error_message=f"{operation} took {duration}ms (threshold: {threshold}ms)"
                )
                await super().notify(alert_event)
        
        await super().notify(event)


class AuditEventBus(EventBus):
    """
    Specialized event bus for audit logging.
    Single Responsibility: Ensure audit compliance.
    """
    
    def __init__(self, audit_events: List[EventType] = None):
        super().__init__()
        self.audit_events = audit_events or [
            EventType.CREATED,
            EventType.UPDATED,
            EventType.DELETED
        ]
        
        # Always attach persistence observer for audit
        self.attach(PersistenceObserver("AuditPersistence"))
    
    async def notify(self, event: Event) -> None:
        """Enhanced notification with audit requirements."""
        if event.event_type in self.audit_events:
            # Ensure audit metadata
            if 'audit_timestamp' not in event.metadata:
                event.metadata['audit_timestamp'] = datetime.now().isoformat()
            if 'audit_user' not in event.metadata:
                event.metadata['audit_user'] = event.data.get('user_id', 'system')
        
        await super().notify(event)


# ============================================================================
# EVENT BUS MANAGER - Singleton pattern for global access
# ============================================================================


class EventBusManager:
    """
    Manager for multiple event buses.
    Singleton Pattern: Single point of access to event system.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._buses: Dict[str, EventBus] = {
            'main': EventBus(),
            'performance': PerformanceEventBus(),
            'audit': AuditEventBus()
        }
        
        # Setup default observers
        self._setup_default_observers()
        
        self._initialized = True
    
    def _setup_default_observers(self):
        """Setup default observers for all buses."""
        # Main bus gets all observers
        main_bus = self._buses['main']
        main_bus.attach(LoggingObserver())
        main_bus.attach(MetricsObserver())
        main_bus.attach(AlertingObserver())
        
        # Performance bus gets specialized observers
        perf_bus = self._buses['performance']
        perf_bus.attach(MetricsObserver("PerformanceMetrics"))
        
        # Audit bus already has persistence observer
    
    def get_bus(self, name: str = 'main') -> EventBus:
        """Get event bus by name."""
        return self._buses.get(name, self._buses['main'])
    
    async def publish(self, event: Event, bus: str = 'main') -> None:
        """Publish event to specified bus."""
        event_bus = self.get_bus(bus)
        await event_bus.publish(event)
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Awaitable[None]],
        bus: str = 'main'
    ):
        """Subscribe to events on specified bus."""
        event_bus = self.get_bus(bus)
        event_bus.subscribe(event_type, handler)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_event_bus(name: str = 'main') -> EventBus:
    """Get event bus instance."""
    manager = EventBusManager()
    return manager.get_bus(name)


async def publish_event(event: Event, bus: str = 'main') -> None:
    """Publish event to bus."""
    manager = EventBusManager()
    await manager.publish(event, bus)


def create_event(
    event_type: EventType,
    data: Dict[str, Any],
    source: str = "",
    **kwargs
) -> BaseEvent:
    """Factory function to create events."""
    return BaseEvent(
        event_type=event_type,
        data=data,
        source=source,
        **kwargs
    )


# Export public API
__all__ = [
    # Events
    'BaseEvent',
    'QueryEvent',
    'ErrorEvent',
    'PerformanceEvent',
    
    # Observers
    'BaseObserver',
    'LoggingObserver',
    'MetricsObserver',
    'AlertingObserver',
    'PersistenceObserver',
    
    # Event buses
    'EventBus',
    'PerformanceEventBus',
    'AuditEventBus',
    'EventBusManager',
    
    # Convenience functions
    'get_event_bus',
    'publish_event',
    'create_event',
] 