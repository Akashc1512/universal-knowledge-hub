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

logger = logging.getLogger(__name__)


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
    
    def analyze_query(self, query: str, source_ip: str) -> Tuple[bool, Dict[str, Any]]:
        """Analyze query for security threats."""
        threats = []
        total_score = 0
        
        for threat_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    threats.append({
                        'type': threat_type,
                        'pattern': pattern,
                        'matches': matches,
                        'severity': self._get_threat_severity(threat_type)
                    })
                    total_score += self._get_threat_score(threat_type)
        
        # Update IP threat score
        if total_score > 0:
            self.suspicious_ips[source_ip]['score'] += total_score
            self.suspicious_ips[source_ip]['last_seen'] = time.time()
            self.suspicious_ips[source_ip]['events'].append({
                'timestamp': time.time(),
                'query': query,
                'threats': threats,
                'score': total_score
            })
        
        # Check if IP should be blocked
        ip_data = self.suspicious_ips[source_ip]
        if ip_data['score'] > 50:  # High threat threshold
            self.blocked_ips.add(source_ip)
            logger.warning(f"IP {source_ip} blocked due to high threat score: {ip_data['score']}")
        
        return len(threats) > 0, {
            'threats_detected': threats,
            'total_score': total_score,
            'ip_threat_score': ip_data['score'],
            'is_blocked': source_ip in self.blocked_ips
        }
    
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
            'sql_injection': 20,
            'xss_attack': 15,
            'path_traversal': 10,
            'command_injection': 30
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
            'total_threat_score': sum(ip_data['score'] for ip_data in self.suspicious_ips.values()),
            'recent_events': len([ip for ip, data in self.suspicious_ips.items() 
                                if time.time() - data['last_seen'] < 3600])
        }


class AnomalyDetector:
    """Detects anomalous behavior patterns."""
    
    def __init__(self):
        self.user_patterns = defaultdict(lambda: {
            'query_count': 0,
            'avg_response_time': 0.0,
            'query_types': defaultdict(int),
            'last_activity': 0,
            'suspicious_activity': 0
        })
        
        self.global_patterns = {
            'avg_queries_per_minute': 0,
            'avg_response_time': 0.0,
            'unique_users_per_hour': 0
        }
        
        self.anomaly_thresholds = {
            'queries_per_minute': 100,
            'response_time_spike': 2.0,  # 2x normal
            'unusual_query_pattern': 0.8,  # 80% similarity
            'rapid_fire_queries': 10  # queries per second
        }
    
    def analyze_user_behavior(self, user_id: str, query: str, response_time: float) -> Dict[str, Any]:
        """Analyze user behavior for anomalies."""
        current_time = time.time()
        user_data = self.user_patterns[user_id]
        
        # Update user patterns
        user_data['query_count'] += 1
        user_data['last_activity'] = current_time
        
        # Update average response time
        if user_data['query_count'] > 1:
            old_avg = user_data['avg_response_time']
            new_avg = (old_avg * (user_data['query_count'] - 1) + response_time) / user_data['query_count']
            user_data['avg_response_time'] = new_avg
        else:
            user_data['avg_response_time'] = response_time
        
        # Detect anomalies
        anomalies = []
        
        # Check for rapid-fire queries
        if user_data['query_count'] > 1:
            time_since_last = current_time - user_data['last_activity']
            if time_since_last < 1.0 and user_data['query_count'] > self.anomaly_thresholds['rapid_fire_queries']:
                anomalies.append({
                    'type': 'rapid_fire_queries',
                    'severity': 'medium',
                    'details': f'User made {user_data["query_count"]} queries in {time_since_last:.2f}s'
                })
        
        # Check for response time anomalies
        if user_data['avg_response_time'] > self.global_patterns['avg_response_time'] * self.anomaly_thresholds['response_time_spike']:
            anomalies.append({
                'type': 'response_time_spike',
                'severity': 'low',
                'details': f'Response time {user_data["avg_response_time"]:.2f}s vs global {self.global_patterns["avg_response_time"]:.2f}s'
            })
        
        # Check for unusual query patterns
        if self._detect_unusual_pattern(query, user_data):
            anomalies.append({
                'type': 'unusual_query_pattern',
                'severity': 'medium',
                'details': f'Unusual query pattern detected'
            })
        
        if anomalies:
            user_data['suspicious_activity'] += 1
        
        return {
            'anomalies_detected': anomalies,
            'user_score': user_data['suspicious_activity'],
            'is_suspicious': user_data['suspicious_activity'] > 3
        }
    
    def _detect_unusual_pattern(self, query: str, user_data: Dict[str, Any]) -> bool:
        """Detect unusual query patterns."""
        # Simple implementation - can be enhanced with ML
        query_length = len(query)
        word_count = len(query.split())
        
        # Check for unusually long or short queries
        if query_length > 1000 or query_length < 5:
            return True
        
        # Check for repetitive patterns
        if user_data['query_count'] > 10:
            # Check if this query is very similar to previous ones
            return False  # Simplified for now
        
        return False
    
    def update_global_patterns(self, global_stats: Dict[str, Any]):
        """Update global behavior patterns."""
        self.global_patterns.update(global_stats)


class SecurityMonitor:
    """Main security monitoring system."""
    
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.anomaly_detector = AnomalyDetector()
        self.security_events = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # Rate limiting for security events
        self.event_rate_limits = {
            'high': 10,  # events per minute
            'medium': 50,
            'low': 100
        }
        self.event_counts = defaultdict(lambda: deque(maxlen=60))
    
    def analyze_request(self, query: str, source_ip: str, user_id: Optional[str], 
                       response_time: float) -> Dict[str, Any]:
        """Analyze request for security threats and anomalies."""
        with self.lock:
            # Check if IP is blocked
            if self.threat_detector.is_ip_blocked(source_ip):
                return {
                    'blocked': True,
                    'reason': 'IP blocked due to suspicious activity',
                    'action': 'reject'
                }
            
            # Threat detection
            has_threats, threat_info = self.threat_detector.analyze_query(query, source_ip)
            
            # Anomaly detection
            anomaly_info = self.anomaly_detector.analyze_user_behavior(user_id or 'anonymous', query, response_time)
            
            # Determine action
            action = 'allow'
            if has_threats and threat_info['total_score'] > 20:
                action = 'block'
            elif anomaly_info['is_suspicious']:
                action = 'monitor'
            
            # Record security event
            if has_threats or anomaly_info['anomalies_detected']:
                self._record_security_event(
                    event_type='threat_detected' if has_threats else 'anomaly_detected',
                    severity='high' if has_threats else 'medium',
                    source_ip=source_ip,
                    user_id=user_id,
                    details={
                        'threats': threat_info.get('threats_detected', []),
                        'anomalies': anomaly_info.get('anomalies_detected', []),
                        'query': query[:100]  # Truncate for security
                    },
                    action_taken=action
                )
            
            return {
                'blocked': action == 'block',
                'monitored': action == 'monitor',
                'threats': threat_info,
                'anomalies': anomaly_info,
                'action': action
            }
    
    def _record_security_event(self, event_type: str, severity: str, source_ip: str,
                              user_id: Optional[str], details: Dict[str, Any], action_taken: str):
        """Record a security event."""
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
        
        # Update rate limiting
        current_time = time.time()
        self.event_counts[severity].append(current_time)
        
        # Clean old events
        cutoff_time = current_time - 60
        while self.event_counts[severity] and self.event_counts[severity][0] < cutoff_time:
            self.event_counts[severity].popleft()
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        with self.lock:
            recent_events = [e for e in self.security_events 
                           if time.time() - e.timestamp < 3600]  # Last hour
            
            return {
                'threat_stats': self.threat_detector.get_threat_stats(),
                'recent_security_events': len(recent_events),
                'blocked_ips': len(self.threat_detector.blocked_ips),
                'suspicious_users': len([u for u, data in self.anomaly_detector.user_patterns.items() 
                                       if data['suspicious_activity'] > 0]),
                'event_rate_limits': {
                    severity: len(events) for severity, events in self.event_counts.items()
                }
            }


# Global security monitor
security_monitor = SecurityMonitor()


async def check_security(query: str, source_ip: str, user_id: Optional[str], 
                        response_time: float) -> Dict[str, Any]:
    """Check request for security threats and anomalies."""
    return security_monitor.analyze_request(query, source_ip, user_id, response_time)


def get_security_summary() -> Dict[str, Any]:
    """Get security summary."""
    return security_monitor.get_security_stats() 