"""
Advanced Security System for Universal Knowledge Platform
Provides threat detection, anomaly detection, and security monitoring.
"""

import time
import hashlib
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import ipaddress
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Global thread pool for CPU-intensive operations
_security_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="security")


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: str
    severity: str  # low, medium, high, critical
    source_ip: str
    user_id: Optional[str]
    details: Dict[str, Any]
    action_taken: str


class ThreatDetector:
    """Detects various types of security threats."""
    
    def __init__(self):
        self.suspicious_patterns = {
            'sql_injection': [
                r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
                r"(--|\b(and|or)\b\s+\d+\s*[=<>])",
                r"(\b(exec|execute|xp_|sp_)\b)",
                r"(\b(script|javascript|vbscript)\b)",
                r"(\b(iframe|object|embed)\b)"
            ],
            'xss_attack': [
                r"(<script[^>]*>.*?</script>)",
                r"(javascript:.*)",
                r"(on\w+\s*=)",
                r"(<iframe[^>]*>)",
                r"(<object[^>]*>)"
            ],
            'path_traversal': [
                r"(\.\./|\.\.\\)",
                r"(/etc/passwd|/etc/shadow)",
                r"(c:\\windows\\system32)",
                r"(/proc/|/sys/)"
            ],
            'command_injection': [
                r"(\b(cat|ls|dir|whoami|id|pwd|wget|curl)\b)",
                r"(\b(rm|del|format|fdisk)\b)",
                r"(\b(netcat|nc|telnet|ssh)\b)",
                r"(\$\(.*\)|`.*`)"
            ]
        }
        
        self.threat_scores = defaultdict(int)
        self.blocked_ips = set()
        self.suspicious_ips = defaultdict(lambda: {
            'score': 0,
            'events': deque(maxlen=100),
            'last_seen': 0
        })
    
    def _analyze_query_sync(self, query: str, source_ip: str) -> Tuple[bool, Dict[str, Any]]:
        """Synchronous version of query analysis for thread pool execution."""
        threats = []
        total_score = 0
        false_positive_indicators = []
        
        # Check for legitimate educational/technical content that might trigger false positives
        educational_patterns = [
            r'\b(what is|how to|explain|define|describe|tutorial|example|code)\b',
            r'\b(sql|javascript|html|css|python|java|c\+\+|php|ruby)\b',
            r'\b(select|insert|update|delete|create|drop|alter)\s+(statement|query|command)\b',
            r'\b(script|iframe|object|embed)\s+(tag|element|example)\b',
            r'\b(command|shell|terminal|bash|powershell)\s+(line|prompt|example)\b'
        ]
        
        # Check if query appears to be educational/technical
        is_educational = any(re.search(pattern, query, re.IGNORECASE) for pattern in educational_patterns)
        
        for threat_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    # Reduce score for educational queries
                    base_score = self._get_threat_score(threat_type)
                    adjusted_score = base_score // 2 if is_educational else base_score
                    
                    threats.append({
                        'type': threat_type,
                        'pattern': pattern,
                        'matches': matches,
                        'severity': self._get_threat_severity(threat_type),
                        'adjusted_score': adjusted_score,
                        'educational_context': is_educational
                    })
                    total_score += adjusted_score
                    
                    # Track potential false positives
                    if is_educational:
                        false_positive_indicators.append({
                            'threat_type': threat_type,
                            'pattern': pattern,
                            'context': 'educational_query'
                        })
        
        # Update IP threat score with more nuanced scoring
        if total_score > 0:
            # Reduce score for educational queries
            final_score = total_score // 2 if is_educational else total_score
            
            self.suspicious_ips[source_ip]['score'] += final_score
            self.suspicious_ips[source_ip]['last_seen'] = time.time()
            self.suspicious_ips[source_ip]['events'].append({
                'timestamp': time.time(),
                'query': query,
                'threats': threats,
                'score': final_score,
                'educational_context': is_educational,
                'false_positive_indicators': false_positive_indicators
            })
        
        # More nuanced blocking logic
        ip_score = self.suspicious_ips[source_ip]['score']
        is_blocked = False
        block_reason = None
        
        # Only block if score is very high AND not educational
        if ip_score > 100 and not is_educational:
            self.blocked_ips.add(source_ip)
            is_blocked = True
            block_reason = "High threat score from non-educational queries"
        elif ip_score > 200:  # Even educational queries can be blocked if score is very high
            self.blocked_ips.add(source_ip)
            is_blocked = True
            block_reason = "Extremely high threat score"
        
        return total_score > 0, {
            'threats': threats,
            'total_score': total_score,
            'adjusted_score': total_score // 2 if is_educational else total_score,
            'ip_score': ip_score,
            'is_blocked': is_blocked,
            'block_reason': block_reason,
            'educational_context': is_educational,
            'false_positive_indicators': false_positive_indicators
        }
    
    async def analyze_query(self, query: str, source_ip: str) -> Tuple[bool, Dict[str, Any]]:
        """Analyze query for security threats (non-blocking)."""
        # Use thread pool for CPU-intensive regex operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _security_thread_pool, 
            self._analyze_query_sync, 
            query, 
            source_ip
        )
    
    def _get_threat_severity(self, threat_type: str) -> str:
        """Get severity level for threat type."""
        severity_map = {
            'sql_injection': 'high',
            'xss_attack': 'high',
            'path_traversal': 'medium',
            'command_injection': 'critical'
        }
        return severity_map.get(threat_type, 'medium')
    
    def _get_threat_score(self, threat_type: str) -> int:
        """Get score for threat type."""
        score_map = {
            'sql_injection': 10,
            'xss_attack': 15,
            'path_traversal': 5,
            'command_injection': 20
        }
        return score_map.get(threat_type, 5)
    
    def is_ip_blocked(self, source_ip: str) -> bool:
        """Check if IP is blocked."""
        return source_ip in self.blocked_ips
    
    def get_threat_stats(self) -> Dict[str, Any]:
        """Get threat detection statistics."""
        return {
            'blocked_ips': len(self.blocked_ips),
            'suspicious_ips': len(self.suspicious_ips),
            'total_threats': sum(len(ip_data['events']) for ip_data in self.suspicious_ips.values()),
            'threat_scores': dict(self.threat_scores)
        }


class AnomalyDetector:
    """Detects anomalous user behavior patterns."""
    
    def __init__(self):
        self.user_profiles = defaultdict(lambda: {
            'query_count': 0,
            'avg_response_time': 0.0,
            'query_patterns': defaultdict(int),
            'last_seen': 0,
            'suspicious_activity': 0
        })
        
        self.global_patterns = {
            'avg_query_length': 0,
            'common_terms': defaultdict(int),
            'peak_hours': defaultdict(int)
        }
        
        self._lock = threading.Lock()
    
    def _analyze_user_behavior_sync(self, user_id: str, query: str, response_time: float) -> Dict[str, Any]:
        """Synchronous version of user behavior analysis."""
        with self._lock:
            profile = self.user_profiles[user_id]
            
            # Update basic stats
            profile['query_count'] += 1
            profile['last_seen'] = time.time()
            
            # Update average response time
            if profile['query_count'] == 1:
                profile['avg_response_time'] = response_time
            else:
                profile['avg_response_time'] = (
                    (profile['avg_response_time'] * (profile['query_count'] - 1) + response_time) 
                    / profile['query_count']
                )
            
            # Analyze query patterns
            words = query.lower().split()
            for word in words:
                profile['query_patterns'][word] += 1
            
            # Detect unusual patterns
            anomalies = []
            
            # Check for rapid-fire queries
            if profile['query_count'] > 10 and response_time < 0.1:
                anomalies.append('rapid_fire_queries')
                profile['suspicious_activity'] += 1
            
            # Check for unusual query length
            if len(query) > 1000:
                anomalies.append('unusually_long_query')
                profile['suspicious_activity'] += 1
            
            # Check for repeated patterns
            if profile['query_patterns'][words[0] if words else ''] > 5:
                anomalies.append('repeated_patterns')
                profile['suspicious_activity'] += 1
            
            return {
                'anomalies': anomalies,
                'suspicious_score': profile['suspicious_activity'],
                'profile': dict(profile)
            }
    
    async def analyze_user_behavior(self, user_id: str, query: str, response_time: float) -> Dict[str, Any]:
        """Analyze user behavior for anomalies (non-blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _security_thread_pool,
            self._analyze_user_behavior_sync,
            user_id,
            query,
            response_time
        )
    
    def _detect_unusual_pattern(self, query: str, user_data: Dict[str, Any]) -> bool:
        """Detect unusual query patterns."""
        # Simple pattern detection
        unusual_patterns = [
            r'(\w)\1{5,}',  # Repeated characters
            r'[A-Z]{10,}',  # All caps
            r'\d{20,}',     # Long numbers
        ]
        
        for pattern in unusual_patterns:
            if re.search(pattern, query):
                return True
        return False
    
    def update_global_patterns(self, global_stats: Dict[str, Any]):
        """Update global behavior patterns."""
        with self._lock:
            self.global_patterns.update(global_stats)


class SecurityMonitor:
    """Main security monitoring system."""
    
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.anomaly_detector = AnomalyDetector()
        self.security_events = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def _analyze_request_sync(self, query: str, source_ip: str, user_id: Optional[str], 
                       response_time: float) -> Dict[str, Any]:
        """Synchronous version of request analysis."""
        with self._lock:
            # Check if IP is blocked
            if self.threat_detector.is_ip_blocked(source_ip):
                return {
                    'blocked': True,
                    'reason': 'IP blocked due to suspicious activity',
                    'severity': 'high'
                }
            
            # Analyze for threats
            has_threats, threat_info = self.threat_detector._analyze_query_sync(query, source_ip)
            
            # Analyze user behavior
            user_analysis = {}
            if user_id:
                user_analysis = self.anomaly_detector._analyze_user_behavior_sync(user_id, query, response_time)
            
            # Determine overall security status
            security_status = 'safe'
            severity = 'low'
            actions = []
            
            if has_threats:
                security_status = 'threat_detected'
                severity = 'high'
                actions.append('log_threat')
            
            if user_analysis.get('anomalies'):
                security_status = 'anomaly_detected'
                severity = 'medium'
                actions.append('log_anomaly')
            
            if threat_info.get('is_blocked'):
                security_status = 'blocked'
                severity = 'critical'
                actions.append('block_request')
            
            # Record security event
            self._record_security_event_sync(
                'request_analyzed',
                severity,
                source_ip,
                user_id,
                {
                    'query_length': len(query),
                    'response_time': response_time,
                    'threats': threat_info.get('threats', []),
                    'anomalies': user_analysis.get('anomalies', [])
                },
                'monitored'
            )
            
            return {
                'status': security_status,
                'severity': severity,
                'actions': actions,
                'threat_info': threat_info,
                'user_analysis': user_analysis
            }
    
    async def analyze_request(self, query: str, source_ip: str, user_id: Optional[str], 
                       response_time: float) -> Dict[str, Any]:
        """Analyze request for security issues (non-blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _security_thread_pool,
            self._analyze_request_sync,
            query,
            source_ip,
            user_id,
            response_time
        )
    
    def _record_security_event_sync(self, event_type: str, severity: str, source_ip: str,
                              user_id: Optional[str], details: Dict[str, Any], action_taken: str):
        """Record security event (synchronous version)."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            details=details,
            action_taken=action_taken
        )
        
        self.security_events.append(event)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security monitoring statistics."""
        with self._lock:
            return {
                'total_events': len(self.security_events),
                'threat_stats': self.threat_detector.get_threat_stats(),
                'recent_events': [
                    {
                        'timestamp': event.timestamp,
                        'type': event.event_type,
                        'severity': event.severity,
                        'source_ip': event.source_ip
                    }
                    for event in list(self.security_events)[-10:]
                ]
            }


# Global security monitor instance
_security_monitor = SecurityMonitor()


async def check_security(query: str, source_ip: str, user_id: Optional[str], 
                        response_time: float) -> Dict[str, Any]:
    """Check security for a request (non-blocking)."""
    return await _security_monitor.analyze_request(query, source_ip, user_id, response_time)


def get_security_summary() -> Dict[str, Any]:
    """Get security summary."""
    return _security_monitor.get_security_stats() 