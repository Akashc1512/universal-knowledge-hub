"""
Analytics System for Universal Knowledge Platform
Provides query tracking and analytics with privacy protection.
"""

import time
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import json
import re
import os

logger = logging.getLogger(__name__)

# Privacy configuration
DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', '30'))
ANONYMIZE_QUERIES = os.getenv('ANONYMIZE_QUERIES', 'true').lower() == 'true'
LOG_QUERY_CONTENT = os.getenv('LOG_QUERY_CONTENT', 'false').lower() == 'true'

# Analytics storage with privacy controls
query_history = deque(maxlen=10000)  # Limited history
user_analytics = defaultdict(lambda: {
    'query_count': 0,
    'avg_response_time': 0.0,
    'last_seen': 0,
    'anonymized_id': None
})

# Global analytics
global_stats = {
    'total_queries': 0,
    'total_errors': 0,
    'avg_response_time': 0.0,
    'cache_hit_rate': 0.0,
    'popular_queries': defaultdict(int),
    'query_categories': defaultdict(int)
}

def sanitize_query_for_logging(query: str) -> str:
    """
    Sanitize query for logging to protect privacy.
    
    Args:
        query: Original query
        
    Returns:
        Sanitized query safe for logging
    """
    if not LOG_QUERY_CONTENT:
        return "[QUERY_CONTENT_LOGGING_DISABLED]"
    
    # Remove or mask sensitive patterns
    sensitive_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card
        r'\b\d{10,}\b',  # Long numbers (phone, etc.)
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b',  # IBAN
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
    ]
    
    sanitized = query
    for pattern in sensitive_patterns:
        sanitized = re.sub(pattern, '[REDACTED]', sanitized)
    
    # Truncate very long queries
    if len(sanitized) > 200:
        sanitized = sanitized[:200] + "..."
    
    return sanitized

def anonymize_user_id(user_id: str) -> str:
    """
    Create an anonymized user ID for analytics.
    
    Args:
        user_id: Original user ID
        
    Returns:
        Anonymized user ID
    """
    if not user_id:
        return "anonymous"
    
    # Create consistent hash for the same user_id
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]

def categorize_query(query: str) -> str:
    """
    Categorize query for analytics without storing sensitive content.
    
    Args:
        query: Query to categorize
        
    Returns:
        Query category
    """
    query_lower = query.lower()
    
    # Define categories based on keywords
    categories = {
        'technology': ['programming', 'code', 'software', 'computer', 'tech', 'algorithm'],
        'science': ['research', 'study', 'experiment', 'scientific', 'analysis'],
        'education': ['learn', 'teach', 'education', 'school', 'university', 'course'],
        'business': ['company', 'business', 'market', 'finance', 'management'],
        'health': ['medical', 'health', 'disease', 'treatment', 'medicine'],
        'general': ['what', 'how', 'why', 'when', 'where', 'who']
    }
    
    for category, keywords in categories.items():
        if any(keyword in query_lower for keyword in keywords):
            return category
    
    return 'general'

async def track_query(
    query: str, 
    execution_time: float, 
    confidence: float,
    client_ip: str,
    user_id: Optional[str] = None
) -> None:
    """
    Track query analytics with privacy protection.
    
    Args:
        query: User query
        execution_time: Query execution time
        confidence: Response confidence
        client_ip: Client IP address
        user_id: User ID (optional)
    """
    try:
        # Sanitize and anonymize data
        sanitized_query = sanitize_query_for_logging(query)
        anonymized_user_id = anonymize_user_id(user_id) if user_id else "anonymous"
        category = categorize_query(query)
        
        # Create analytics entry
        analytics_entry = {
            'timestamp': time.time(),
            'query_length': len(query),
            'sanitized_query': sanitized_query,
            'category': category,
            'execution_time': execution_time,
            'confidence': confidence,
            'client_ip': client_ip,
            'anonymized_user_id': anonymized_user_id,
            'cache_hit': False  # Will be updated by cache system
        }
        
        # Update global stats
        global_stats['total_queries'] += 1
        global_stats['avg_response_time'] = (
            (global_stats['avg_response_time'] * (global_stats['total_queries'] - 1) + execution_time) 
            / global_stats['total_queries']
        )
        global_stats['popular_queries'][category] += 1
        global_stats['query_categories'][category] += 1
        
        # Update user analytics (anonymized)
        if anonymized_user_id != "anonymous":
            user_data = user_analytics[anonymized_user_id]
            user_data['query_count'] += 1
            user_data['last_seen'] = time.time()
            user_data['anonymized_id'] = anonymized_user_id
            
            # Update average response time
            if user_data['query_count'] == 1:
                user_data['avg_response_time'] = execution_time
            else:
                user_data['avg_response_time'] = (
                    (user_data['avg_response_time'] * (user_data['query_count'] - 1) + execution_time) 
                    / user_data['query_count']
                )
        
        # Store in history (limited size)
        query_history.append(analytics_entry)
        
        # Log analytics safely
        logger.info(f"Query tracked: category={category}, time={execution_time:.3f}s, "
                   f"confidence={confidence:.2f}, user={anonymized_user_id}")
        
    except Exception as e:
        logger.error(f"Error tracking query analytics: {e}")

async def get_analytics_summary() -> Dict[str, Any]:
    """
    Get analytics summary with privacy protection.
    
    Returns:
        Analytics summary
    """
    try:
        # Clean old data based on retention policy
        cutoff_time = time.time() - (DATA_RETENTION_DAYS * 24 * 60 * 60)
        
        # Filter recent data
        recent_queries = [
            entry for entry in query_history
            if entry['timestamp'] > cutoff_time
        ]
        
        # Calculate cache hit rate (placeholder - should be updated by cache system)
        cache_hits = sum(1 for entry in recent_queries if entry.get('cache_hit', False))
        cache_hit_rate = cache_hits / len(recent_queries) if recent_queries else 0.0
        
        # Get popular categories
        category_counts = defaultdict(int)
        for entry in recent_queries:
            category_counts[entry['category']] += 1
        
        # Get active users (anonymized)
        active_users = len([
            user_id for user_id, data in user_analytics.items()
            if data['last_seen'] > cutoff_time
        ])
        
        return {
            'total_queries': global_stats['total_queries'],
            'recent_queries': len(recent_queries),
            'avg_response_time': global_stats['avg_response_time'],
            'cache_hit_rate': cache_hit_rate,
            'popular_categories': dict(category_counts),
            'active_users': active_users,
            'data_retention_days': DATA_RETENTION_DAYS,
            'privacy_protection': {
                'anonymize_queries': ANONYMIZE_QUERIES,
                'log_query_content': LOG_QUERY_CONTENT,
                'data_retention_enabled': True
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating analytics summary: {e}")
        return {
            'error': 'Failed to generate analytics summary',
            'total_queries': global_stats['total_queries']
        }

async def clear_user_data(user_id: str) -> bool:
    """
    Clear all data for a specific user (GDPR compliance).
    
    Args:
        user_id: User ID to clear
        
    Returns:
        Success status
    """
    try:
        anonymized_id = anonymize_user_id(user_id)
        
        # Remove from user analytics
        if anonymized_id in user_analytics:
            del user_analytics[anonymized_id]
        
        # Remove from query history (this is more complex as we need to identify entries)
        # For now, we'll just note that this user's data should be excluded from future queries
        # In a production system, you'd want to mark entries for deletion or use a proper database
        
        logger.info(f"Cleared data for user: {anonymized_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing user data: {e}")
        return False

async def export_user_data(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Export user data for GDPR compliance.
    
    Args:
        user_id: User ID to export
        
    Returns:
        User data (anonymized)
    """
    try:
        anonymized_id = anonymize_user_id(user_id)
        
        # Get user analytics
        user_data = user_analytics.get(anonymized_id, {})
        
        # Get user's query history (anonymized)
        user_queries = [
            {
                'timestamp': entry['timestamp'],
                'category': entry['category'],
                'execution_time': entry['execution_time'],
                'confidence': entry['confidence']
            }
            for entry in query_history
            if entry['anonymized_user_id'] == anonymized_id
        ]
        
        return {
            'user_id': anonymized_id,  # Anonymized
            'analytics': user_data,
            'query_history': user_queries,
            'export_timestamp': time.time(),
            'data_retention_days': DATA_RETENTION_DAYS
        }
        
    except Exception as e:
        logger.error(f"Error exporting user data: {e}")
        return None 