"""
Graceful Shutdown Handler for Universal Knowledge Platform
Handles system signals and ensures clean shutdown.
"""

import asyncio
import signal
import logging
import sys
from typing import Optional, Set, Callable
import time

logger = logging.getLogger(__name__)

# Global shutdown state
_shutdown_event: Optional[asyncio.Event] = None
_shutdown_handlers: Set[Callable] = set()
_shutdown_in_progress = False
_shutdown_timeout = 30  # Maximum time to wait for shutdown


class GracefulShutdownHandler:
    """Handles graceful shutdown of the application."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.shutdown_event = asyncio.Event()
        self.original_handlers = {}
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        if sys.platform == "win32":
            # Windows doesn't support all signals
            signals = [signal.SIGTERM, signal.SIGINT]
        else:
            # Unix-like systems
            signals = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]
        
        for sig in signals:
            # Store original handler
            self.original_handlers[sig] = signal.signal(sig, self._signal_handler)
            
        logger.info(f"Signal handlers installed for: {[s.name for s in signals]}")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        global _shutdown_in_progress
        
        signal_name = signal.Signals(signum).name
        logger.info(f"🛑 Received signal {signal_name} ({signum})")
        
        if _shutdown_in_progress:
            logger.warning("Shutdown already in progress, ignoring signal")
            return
            
        _shutdown_in_progress = True
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Create task to handle shutdown
        asyncio.create_task(self._handle_shutdown(signal_name))
        
    async def _handle_shutdown(self, signal_name: str):
        """Handle the shutdown process."""
        logger.info(f"⏳ Starting graceful shutdown due to {signal_name}")
        
        start_time = time.time()
        
        # Run all registered shutdown handlers
        await self.run_shutdown_handlers()
        
        # Wait for application to shutdown or timeout
        try:
            await asyncio.wait_for(
                self.wait_for_shutdown(),
                timeout=self.timeout
            )
            logger.info("✅ Graceful shutdown completed")
        except asyncio.TimeoutError:
            logger.error(f"❌ Shutdown timeout after {self.timeout} seconds, forcing exit")
            sys.exit(1)
            
        shutdown_duration = time.time() - start_time
        logger.info(f"📊 Shutdown completed in {shutdown_duration:.2f} seconds")
        
        # Exit the process
        sys.exit(0)
        
    async def wait_for_shutdown(self):
        """Wait for shutdown event."""
        await self.shutdown_event.wait()
        
    async def run_shutdown_handlers(self):
        """Run all registered shutdown handlers."""
        logger.info(f"Running {len(_shutdown_handlers)} shutdown handlers")
        
        # Run handlers concurrently
        tasks = [handler() for handler in _shutdown_handlers]
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Shutdown handler {i} failed: {result}")
                    
    def register_shutdown_handler(self, handler: Callable):
        """Register a shutdown handler."""
        _shutdown_handlers.add(handler)
        logger.debug(f"Registered shutdown handler: {handler.__name__}")
        
    def restore_signal_handlers(self):
        """Restore original signal handlers."""
        for sig, handler in self.original_handlers.items():
            signal.signal(sig, handler)
            

# Global instance
_shutdown_handler: Optional[GracefulShutdownHandler] = None


def get_shutdown_handler() -> GracefulShutdownHandler:
    """Get or create the global shutdown handler."""
    global _shutdown_handler
    
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdownHandler()
        
    return _shutdown_handler


def register_shutdown_handler(handler: Callable):
    """
    Register a function to be called during shutdown.
    
    Args:
        handler: Async function to call during shutdown
    """
    shutdown_handler = get_shutdown_handler()
    shutdown_handler.register_shutdown_handler(handler)


async def wait_for_shutdown():
    """Wait for shutdown signal."""
    shutdown_handler = get_shutdown_handler()
    await shutdown_handler.wait_for_shutdown()


def is_shutting_down() -> bool:
    """Check if shutdown is in progress."""
    return _shutdown_in_progress


# Decorator for shutdown handlers
def on_shutdown(func: Callable):
    """
    Decorator to register a function as a shutdown handler.
    
    Example:
        @on_shutdown
        async def cleanup():
            await close_database()
    """
    register_shutdown_handler(func)
    return func


# Context manager for graceful shutdown
class GracefulShutdown:
    """Context manager for graceful shutdown handling."""
    
    def __init__(self, timeout: int = 30):
        self.handler = GracefulShutdownHandler(timeout)
        
    async def __aenter__(self):
        """Setup signal handlers on entry."""
        self.handler.setup_signal_handlers()
        return self.handler
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Restore signal handlers on exit."""
        self.handler.restore_signal_handlers()
        
        # If exiting due to exception, run shutdown handlers
        if exc_type is not None:
            await self.handler.run_shutdown_handlers() 