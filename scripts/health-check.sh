#!/bin/bash

# Universal Knowledge Hub Enterprise Platform Health Check Script
# Comprehensive health monitoring for production readiness

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="knowledge-hub"
TIMEOUT=30

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
        exit 1
    fi
}

# Check namespace exists
check_namespace() {
    log "Checking namespace: $NAMESPACE"
    
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    log "Namespace $NAMESPACE exists"
}

# Check pod health
check_pod_health() {
    log "Checking pod health..."
    
    # Get all pods in namespace
    pods=$(kubectl get pods -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')
    
    if [ -z "$pods" ]; then
        error "No pods found in namespace $NAMESPACE"
        return 1
    fi
    
    all_healthy=true
    
    for pod in $pods; do
        status=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.phase}')
        ready=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.containerStatuses[0].ready}')
        
        if [ "$status" = "Running" ] && [ "$ready" = "true" ]; then
            log "✓ Pod $pod is healthy"
        else
            error "✗ Pod $pod is not healthy (Status: $status, Ready: $ready)"
            all_healthy=false
        fi
    done
    
    if [ "$all_healthy" = true ]; then
        log "All pods are healthy"
        return 0
    else
        error "Some pods are not healthy"
        return 1
    fi
}

# Check service health
check_service_health() {
    log "Checking service health..."
    
    # Get all services in namespace
    services=$(kubectl get svc -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')
    
    if [ -z "$services" ]; then
        error "No services found in namespace $NAMESPACE"
        return 1
    fi
    
    all_healthy=true
    
    for service in $services; do
        # Check if service has endpoints
        endpoints=$(kubectl get endpoints $service -n $NAMESPACE -o jsonpath='{.subsets[*].addresses[*].ip}')
        
        if [ -n "$endpoints" ]; then
            log "✓ Service $service has endpoints"
        else
            error "✗ Service $service has no endpoints"
            all_healthy=false
        fi
    done
    
    if [ "$all_healthy" = true ]; then
        log "All services are healthy"
        return 0
    else
        error "Some services are not healthy"
        return 1
    fi
}

# Check database connectivity
check_database_health() {
    log "Checking database health..."
    
    # Get database pod
    db_pod=$(kubectl get pods -n $NAMESPACE -l app=postgresql -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$db_pod" ]; then
        error "Database pod not found"
        return 1
    fi
    
    # Check if database is ready
    if kubectl exec $db_pod -n $NAMESPACE -- pg_isready -U postgres; then
        log "✓ Database is healthy"
        return 0
    else
        error "✗ Database is not healthy"
        return 1
    fi
}

# Check Redis connectivity
check_redis_health() {
    log "Checking Redis health..."
    
    # Get Redis pod
    redis_pod=$(kubectl get pods -n $NAMESPACE -l app=redis -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$redis_pod" ]; then
        error "Redis pod not found"
        return 1
    fi
    
    # Check if Redis is ready
    if kubectl exec $redis_pod -n $NAMESPACE -- redis-cli ping; then
        log "✓ Redis is healthy"
        return 0
    else
        error "✗ Redis is not healthy"
        return 1
    fi
}

# Check Elasticsearch health
check_elasticsearch_health() {
    log "Checking Elasticsearch health..."
    
    # Get Elasticsearch service
    es_service=$(kubectl get svc -n $NAMESPACE -l app=elasticsearch -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$es_service" ]; then
        error "Elasticsearch service not found"
        return 1
    fi
    
    # Check Elasticsearch health
    if kubectl exec -n $NAMESPACE deployment/elasticsearch-master -- curl -s http://localhost:9200/_cluster/health | grep -q '"status":"green"'; then
        log "✓ Elasticsearch is healthy"
        return 0
    else
        error "✗ Elasticsearch is not healthy"
        return 1
    fi
}

# Check API health
check_api_health() {
    log "Checking API health..."
    
    # Get API service
    api_service=$(kubectl get svc -n $NAMESPACE -l app=knowledge-hub-api -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$api_service" ]; then
        error "API service not found"
        return 1
    fi
    
    # Port forward to API service
    kubectl port-forward svc/$api_service 8000:8000 -n $NAMESPACE &
    PF_PID=$!
    
    # Wait for port forward
    sleep 5
    
    # Check API health endpoint
    if curl -f http://localhost:8000/health; then
        log "✓ API is healthy"
        kill $PF_PID
        return 0
    else
        error "✗ API is not healthy"
        kill $PF_PID
        return 1
    fi
}

# Check frontend health
check_frontend_health() {
    log "Checking frontend health..."
    
    # Get frontend service
    frontend_service=$(kubectl get svc -n $NAMESPACE -l app=knowledge-hub-frontend -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$frontend_service" ]; then
        error "Frontend service not found"
        return 1
    fi
    
    # Port forward to frontend service
    kubectl port-forward svc/$frontend_service 3000:3000 -n $NAMESPACE &
    PF_PID=$!
    
    # Wait for port forward
    sleep 5
    
    # Check frontend health
    if curl -f http://localhost:3000/; then
        log "✓ Frontend is healthy"
        kill $PF_PID
        return 0
    else
        error "✗ Frontend is not healthy"
        kill $PF_PID
        return 1
    fi
}

# Check monitoring stack
check_monitoring_health() {
    log "Checking monitoring stack health..."
    
    # Check Prometheus
    if kubectl get pods -n monitoring -l app=prometheus &> /dev/null; then
        log "✓ Prometheus is running"
    else
        warning "Prometheus is not running"
    fi
    
    # Check Grafana
    if kubectl get pods -n monitoring -l app=grafana &> /dev/null; then
        log "✓ Grafana is running"
    else
        warning "Grafana is not running"
    fi
    
    # Check Alertmanager
    if kubectl get pods -n monitoring -l app=alertmanager &> /dev/null; then
        log "✓ Alertmanager is running"
    else
        warning "Alertmanager is not running"
    fi
}

# Check security components
check_security_health() {
    log "Checking security components..."
    
    # Check Vault
    if kubectl get pods -n vault-system -l app.kubernetes.io/name=vault &> /dev/null; then
        log "✓ Vault is running"
    else
        warning "Vault is not running"
    fi
    
    # Check network policies
    if kubectl get networkpolicies -n $NAMESPACE &> /dev/null; then
        log "✓ Network policies are configured"
    else
        warning "Network policies are not configured"
    fi
    
    # Check RBAC
    if kubectl get roles -n $NAMESPACE &> /dev/null; then
        log "✓ RBAC is configured"
    else
        warning "RBAC is not configured"
    fi
}

# Check AI/ML components
check_ai_ml_health() {
    log "Checking AI/ML components..."
    
    # Check MLflow
    if kubectl get pods -n $NAMESPACE -l app=mlflow &> /dev/null; then
        log "✓ MLflow is running"
    else
        warning "MLflow is not running"
    fi
    
    # Check model serving
    if kubectl get pods -n $NAMESPACE -l app=model-serving &> /dev/null; then
        log "✓ Model serving is running"
    else
        warning "Model serving is not running"
    fi
    
    # Check GPU support
    if kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' | grep -q '[1-9]'; then
        log "✓ GPU support is available"
    else
        info "GPU support is not available"
    fi
}

# Check performance metrics
check_performance_metrics() {
    log "Checking performance metrics..."
    
    # Check CPU usage
    cpu_usage=$(kubectl top pods -n $NAMESPACE --no-headers | awk '{sum+=$2} END {print sum}')
    if [ "$cpu_usage" -lt 1000 ]; then
        log "✓ CPU usage is normal: ${cpu_usage}m"
    else
        warning "High CPU usage: ${cpu_usage}m"
    fi
    
    # Check memory usage
    memory_usage=$(kubectl top pods -n $NAMESPACE --no-headers | awk '{sum+=$3} END {print sum}')
    if [ "$memory_usage" -lt 2048 ]; then
        log "✓ Memory usage is normal: ${memory_usage}Mi"
    else
        warning "High memory usage: ${memory_usage}Mi"
    fi
    
    # Check disk usage
    disk_usage=$(kubectl exec -n $NAMESPACE deployment/knowledge-hub-api -- df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 80 ]; then
        log "✓ Disk usage is normal: ${disk_usage}%"
    else
        warning "High disk usage: ${disk_usage}%"
    fi
}

# Check logs for errors
check_logs() {
    log "Checking logs for errors..."
    
    # Check for error logs in the last hour
    error_count=$(kubectl logs -n $NAMESPACE --since=1h --all-containers | grep -i error | wc -l)
    
    if [ "$error_count" -eq 0 ]; then
        log "✓ No errors found in logs"
    else
        warning "Found $error_count errors in logs"
        kubectl logs -n $NAMESPACE --since=1h --all-containers | grep -i error | tail -5
    fi
}

# Check resource quotas
check_resource_quotas() {
    log "Checking resource quotas..."
    
    quotas=$(kubectl get resourcequota -n $NAMESPACE --no-headers)
    
    if [ -n "$quotas" ]; then
        log "✓ Resource quotas are configured"
        kubectl get resourcequota -n $NAMESPACE
    else
        warning "No resource quotas configured"
    fi
}

# Check ingress/load balancer
check_ingress_health() {
    log "Checking ingress/load balancer..."
    
    # Check if ingress is configured
    if kubectl get ingress -n $NAMESPACE &> /dev/null; then
        log "✓ Ingress is configured"
        
        # Get ingress host
        host=$(kubectl get ingress -n $NAMESPACE -o jsonpath='{.items[0].spec.rules[0].host}')
        
        if [ -n "$host" ]; then
            log "✓ Ingress host: $host"
        else
            warning "No ingress host configured"
        fi
    else
        warning "No ingress configured"
    fi
}

# Main health check function
main() {
    log "Starting comprehensive health check for Universal Knowledge Hub Enterprise Platform"
    
    # Check prerequisites
    check_kubectl
    
    # Check namespace
    check_namespace
    
    # Run all health checks
    local exit_code=0
    
    check_pod_health || exit_code=1
    check_service_health || exit_code=1
    check_database_health || exit_code=1
    check_redis_health || exit_code=1
    check_elasticsearch_health || exit_code=1
    check_api_health || exit_code=1
    check_frontend_health || exit_code=1
    check_monitoring_health
    check_security_health
    check_ai_ml_health
    check_performance_metrics
    check_logs
    check_resource_quotas
    check_ingress_health
    
    if [ $exit_code -eq 0 ]; then
        log "✓ All critical health checks passed"
        log "Platform is healthy and ready for production"
    else
        error "✗ Some critical health checks failed"
        error "Please review the errors above and fix them before proceeding"
    fi
    
    return $exit_code
}

# Run main function
main "$@" 