# Monitoring & Observability Setup

This document describes the monitoring and observability setup for the Universal Knowledge Platform.

## Overview

The platform includes comprehensive monitoring with:
- **Prometheus** for metrics collection
- **Grafana** for visualization and dashboards
- **Structured logging** with JSON formatting
- **Health checks** and integration monitoring
- **Custom metrics** for business KPIs

## Quick Start

### 1. Start Monitoring Services

```bash
# Create the monitoring network
docker network create ukp-network

# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Start the main application
docker-compose -f docker-compose.fullstack.yml up -d
```

### 2. Verify Setup

```bash
# Run the monitoring setup script
python scripts/setup_monitoring.py
```

### 3. Access Monitoring Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin123)
- **cAdvisor**: http://localhost:8080
- **Node Exporter**: http://localhost:9100/metrics

## Metrics Endpoints

### API Metrics
- **Endpoint**: `GET /metrics`
- **Format**: Prometheus text format
- **Content-Type**: `text/plain; version=0.0.4; charset=utf-8`

### Health Checks
- **Endpoint**: `GET /health`
- **Response**: JSON with service status

### Integration Status
- **Endpoint**: `GET /integrations`
- **Response**: JSON with external service status

## Key Metrics

### Request Metrics
- `ukp_requests_total` - Total request count by method/endpoint/status
- `ukp_request_duration_seconds` - Request duration histogram
- `ukp_errors_total` - Error count by type/endpoint

### Cache Metrics
- `ukp_cache_hits_total` - Cache hit count by type
- `ukp_cache_misses_total` - Cache miss count by type
- `ukp_cache_size` - Current cache size by type

### Agent Metrics
- `ukp_agent_requests_total` - Agent request count by type/status
- `ukp_agent_duration_seconds` - Agent processing duration

### Security Metrics
- `ukp_security_threats_total` - Security threat count by type/severity
- `ukp_blocked_requests_total` - Blocked request count by reason

### System Metrics
- `ukp_system_memory_bytes` - Memory usage
- `ukp_system_cpu_percent` - CPU usage
- `ukp_active_connections` - Active connections

## Logging

### Structured Logging
All logs are in JSON format with consistent fields:
- `timestamp` - ISO 8601 timestamp
- `level` - Log level (DEBUG, INFO, WARNING, ERROR)
- `request_id` - Unique request identifier
- `user_id` - User identifier
- `component` - Component name
- `message` - Log message

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General operational messages
- **WARNING**: Warning conditions
- **ERROR**: Error conditions

### Example Log Entry
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "request_id": "req_123456",
  "user_id": "user_789",
  "component": "api.main",
  "method": "POST",
  "path": "/query",
  "status_code": 200,
  "duration": 1.234,
  "message": "📤 POST /query -> 200 (1.234s)"
}
```

## Prometheus Configuration

### Scrape Targets
- **UKP Backend**: `backend:8002/metrics` (every 10s)
- **UKP Frontend**: `frontend:3000/metrics` (every 30s)
- **Redis**: `redis:6379` (every 30s)
- **Node Exporter**: `node-exporter:9100` (every 15s)

### Retention
- **Metrics**: 200 hours (8.3 days)
- **Storage**: Persistent volume

## Grafana Dashboards

### Default Dashboards
1. **UKP Overview** - High-level system metrics
2. **Request Analytics** - API performance metrics
3. **Error Tracking** - Error rates and types
4. **Cache Performance** - Cache hit/miss ratios
5. **System Resources** - CPU, memory, disk usage

### Dashboard Setup
1. Access Grafana at http://localhost:3001
2. Login with admin/admin123
3. Add Prometheus as a data source:
   - URL: `http://prometheus:9090`
   - Access: Server (default)
4. Import dashboards from `monitoring/grafana/dashboards/`

## Alerts

### Critical Alerts
- **High Error Rate**: >5% error rate for 5 minutes
- **High Response Time**: >10s average response time
- **Service Down**: Health check failing for 2 minutes
- **High Memory Usage**: >90% memory usage
- **High CPU Usage**: >80% CPU usage

### Alert Configuration
Alerts are configured in Prometheus rules and can be sent to:
- Email
- Slack
- PagerDuty
- Webhook

## Troubleshooting

### Common Issues

#### Metrics Not Appearing
1. Check if the API is running: `curl http://localhost:8002/health`
2. Verify metrics endpoint: `curl http://localhost:8002/metrics`
3. Check Prometheus targets: http://localhost:9090/targets
4. Verify network connectivity between services

#### High Memory Usage
1. Check container memory usage: `docker stats`
2. Review application logs for memory leaks
3. Monitor cache size metrics
4. Consider increasing container memory limits

#### Slow Response Times
1. Check request duration metrics
2. Review database query performance
3. Monitor external API response times
4. Check cache hit/miss ratios

### Debug Commands

```bash
# Check API health
curl http://localhost:8002/health

# Get metrics
curl http://localhost:8002/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# View container logs
docker logs ukp-backend
docker logs ukp-prometheus
docker logs ukp-grafana

# Check network connectivity
docker exec ukp-prometheus wget -qO- http://backend:8002/metrics
```

## Performance Tuning

### Prometheus
- **Scrape Interval**: 10s for critical metrics, 30s for others
- **Retention**: 200 hours (adjust based on storage)
- **Memory**: 2-4GB recommended
- **Storage**: SSD recommended for better performance

### Grafana
- **Memory**: 1-2GB recommended
- **Refresh Rate**: 30s for dashboards
- **Query Timeout**: 30s default

### Application
- **Metrics Collection**: Asynchronous to avoid blocking
- **Log Buffering**: Use structured logging with buffering
- **Health Checks**: Lightweight and fast

## Security Considerations

### Access Control
- **Prometheus**: Internal network only
- **Grafana**: Admin authentication required
- **API Metrics**: Protected by authentication

### Data Privacy
- **Logs**: No sensitive data in logs
- **Metrics**: No PII in metric labels
- **Retention**: Configurable data retention

### Network Security
- **Internal Communication**: Docker network isolation
- **External Access**: Reverse proxy with authentication
- **Monitoring**: Separate network for monitoring services

## Next Steps

1. **Custom Dashboards**: Create business-specific dashboards
2. **Alert Rules**: Configure alerts for critical metrics
3. **Log Aggregation**: Set up centralized log collection
4. **Tracing**: Implement distributed tracing with Jaeger
5. **APM**: Add application performance monitoring
6. **SLA Monitoring**: Track service level agreements

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [FastAPI Monitoring](https://fastapi.tiangolo.com/advanced/middleware/)
- [Structured Logging Best Practices](https://12factor.net/logs) 