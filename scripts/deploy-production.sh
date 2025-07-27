#!/bin/bash

# 🚀 Universal Knowledge Platform - Production Deployment Script
# Enterprise-grade deployment with blue-green strategy, health checks, and rollback

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_NAME="ukp-production"
NAMESPACE="ukp-production"
REGISTRY="ghcr.io"
IMAGE_NAME="universal-knowledge-hub"
VERSION="${1:-latest}"
ENVIRONMENT="production"
ROLLBACK_VERSION="${2:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check kubectl context
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not configured or cluster is not accessible"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Validate deployment parameters
validate_deployment() {
    log_info "Validating deployment parameters..."
    
    if [[ -z "$VERSION" ]]; then
        log_error "Version parameter is required"
        exit 1
    fi
    
    if [[ "$VERSION" == "latest" ]]; then
        log_warning "Deploying latest version - consider using specific version tag"
    fi
    
    log_success "Deployment validation passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace if it doesn't exist..."
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl create namespace "$NAMESPACE"
        log_success "Created namespace: $NAMESPACE"
    else
        log_info "Namespace $NAMESPACE already exists"
    fi
}

# Deploy secrets and configmaps
deploy_secrets() {
    log_info "Deploying secrets and configmaps..."
    
    # Create secrets from environment variables
    kubectl create secret generic ukp-database-secret \
        --from-literal=database-url="$DATABASE_URL" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic ukp-redis-secret \
        --from-literal=redis-url="$REDIS_URL" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic ukp-openai-secret \
        --from-literal=api-key="$OPENAI_API_KEY" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic ukp-anthropic-secret \
        --from-literal=api-key="$ANTHROPIC_API_KEY" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets and configmaps deployed"
}

# Deploy infrastructure components
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Deploy PostgreSQL
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm upgrade --install ukp-postgresql bitnami/postgresql \
        --namespace="$NAMESPACE" \
        --set postgresqlPassword="$POSTGRES_PASSWORD" \
        --set postgresqlDatabase="ukp" \
        --set persistence.enabled=true \
        --set persistence.size="100Gi" \
        --set resources.requests.memory="1Gi" \
        --set resources.requests.cpu="500m" \
        --set resources.limits.memory="4Gi" \
        --set resources.limits.cpu="2000m"
    
    # Deploy Redis
    helm upgrade --install ukp-redis bitnami/redis \
        --namespace="$NAMESPACE" \
        --set auth.password="$REDIS_PASSWORD" \
        --set persistence.enabled=true \
        --set persistence.size="50Gi" \
        --set resources.requests.memory="512Mi" \
        --set resources.requests.cpu="250m" \
        --set resources.limits.memory="2Gi" \
        --set resources.limits.cpu="1000m"
    
    # Deploy Elasticsearch
    helm repo add elastic https://helm.elastic.co
    helm upgrade --install ukp-elasticsearch elastic/elasticsearch \
        --namespace="$NAMESPACE" \
        --set replicas=3 \
        --set resources.requests.memory="2Gi" \
        --set resources.requests.cpu="1000m" \
        --set resources.limits.memory="4Gi" \
        --set resources.limits.cpu="2000m"
    
    log_success "Infrastructure components deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Deploy Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm upgrade --install ukp-prometheus prometheus-community/kube-prometheus-stack \
        --namespace="$NAMESPACE" \
        --set grafana.enabled=true \
        --set grafana.adminPassword="$GRAFANA_PASSWORD" \
        --set prometheus.prometheusSpec.retention="30d" \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage="100Gi"
    
    # Deploy Jaeger for distributed tracing
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    helm upgrade --install ukp-jaeger jaegertracing/jaeger \
        --namespace="$NAMESPACE" \
        --set storage.type=elasticsearch \
        --set storage.options.es.server-urls=http://ukp-elasticsearch-master:9200
    
    log_success "Monitoring stack deployed"
}

# Blue-green deployment
deploy_blue_green() {
    log_info "Starting blue-green deployment..."
    
    # Get current deployment color
    CURRENT_COLOR=$(kubectl get deployment -n "$NAMESPACE" ukp-api -o jsonpath='{.metadata.labels.color}' 2>/dev/null || echo "blue")
    
    # Determine new color
    if [[ "$CURRENT_COLOR" == "blue" ]]; then
        NEW_COLOR="green"
    else
        NEW_COLOR="blue"
    fi
    
    log_info "Current color: $CURRENT_COLOR, New color: $NEW_COLOR"
    
    # Deploy new version
    kubectl apply -f "$PROJECT_ROOT/infrastructure/kubernetes/production/api-deployment.yaml" \
        --namespace="$NAMESPACE"
    
    # Update deployment with new image and color
    kubectl set image deployment/ukp-api ukp-api="$REGISTRY/$IMAGE_NAME/api:$VERSION" \
        --namespace="$NAMESPACE"
    kubectl patch deployment ukp-api -p "{\"metadata\":{\"labels\":{\"color\":\"$NEW_COLOR\"}}}" \
        --namespace="$NAMESPACE"
    
    # Wait for new deployment to be ready
    log_info "Waiting for new deployment to be ready..."
    kubectl rollout status deployment/ukp-api --namespace="$NAMESPACE" --timeout=600s
    
    # Health check
    if perform_health_check; then
        log_success "New deployment is healthy"
        
        # Switch traffic to new deployment
        kubectl patch service ukp-api-service -p "{\"spec\":{\"selector\":{\"color\":\"$NEW_COLOR\"}}}" \
            --namespace="$NAMESPACE"
        
        log_success "Traffic switched to $NEW_COLOR deployment"
        
        # Scale down old deployment
        kubectl scale deployment ukp-api --replicas=0 --namespace="$NAMESPACE"
        
        log_success "Blue-green deployment completed successfully"
    else
        log_error "Health check failed - rolling back"
        rollback_deployment
        exit 1
    fi
}

# Health check function
perform_health_check() {
    log_info "Performing health checks..."
    
    # Wait for service to be ready
    kubectl wait --for=condition=ready pod -l app=ukp-api --namespace="$NAMESPACE" --timeout=300s
    
    # Get service URL
    SERVICE_URL=$(kubectl get service ukp-api-service --namespace="$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -z "$SERVICE_URL" ]]; then
        SERVICE_URL="localhost:8002"
    fi
    
    # Health check endpoints
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        # Check health endpoint
        if curl -f "http://$SERVICE_URL/health" &> /dev/null; then
            log_success "Health endpoint is responding"
            
            # Check metrics endpoint
            if curl -f "http://$SERVICE_URL/metrics" &> /dev/null; then
                log_success "Metrics endpoint is responding"
                
                # Check agents endpoint
                if curl -f "http://$SERVICE_URL/agents" &> /dev/null; then
                    log_success "Agents endpoint is responding"
                    return 0
                fi
            fi
        fi
        
        log_warning "Health check failed, retrying in 10 seconds..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Rollback function
rollback_deployment() {
    log_warning "Initiating rollback..."
    
    if [[ -n "$ROLLBACK_VERSION" ]]; then
        log_info "Rolling back to version: $ROLLBACK_VERSION"
        kubectl set image deployment/ukp-api ukp-api="$REGISTRY/$IMAGE_NAME/api:$ROLLBACK_VERSION" \
            --namespace="$NAMESPACE"
    else
        log_info "Rolling back to previous version"
        kubectl rollout undo deployment/ukp-api --namespace="$NAMESPACE"
    fi
    
    kubectl rollout status deployment/ukp-api --namespace="$NAMESPACE" --timeout=300s
    
    if perform_health_check; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed - manual intervention required"
        exit 1
    fi
}

# Performance testing
run_performance_tests() {
    log_info "Running performance tests..."
    
    # Install k6 if not available
    if ! command -v k6 &> /dev/null; then
        log_info "Installing k6..."
        curl -L https://github.com/grafana/k6/releases/download/v0.45.0/k6-v0.45.0-linux-amd64.tar.gz | tar xz
        sudo mv k6-v0.45.0-linux-amd64/k6 /usr/local/bin/
    fi
    
    # Create performance test script
    cat > /tmp/performance-test.js << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 10 }, // Ramp up
    { duration: '5m', target: 10 }, // Stay at 10 users
    { duration: '2m', target: 50 }, // Ramp up to 50
    { duration: '5m', target: 50 }, // Stay at 50 users
    { duration: '2m', target: 0 },  // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
    http_req_failed: ['rate<0.01'],   // Error rate must be below 1%
  },
};

export default function () {
  const response = http.get('http://localhost:8002/health');
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });
  
  sleep(1);
}
EOF
    
    # Run performance test
    k6 run /tmp/performance-test.js
    
    log_success "Performance tests completed"
}

# Security scanning
run_security_scan() {
    log_info "Running security scan..."
    
    # Install Trivy if not available
    if ! command -v trivy &> /dev/null; then
        log_info "Installing Trivy..."
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    fi
    
    # Scan container image
    trivy image "$REGISTRY/$IMAGE_NAME/api:$VERSION" --severity HIGH,CRITICAL --exit-code 1
    
    log_success "Security scan completed"
}

# Main deployment function
main() {
    log_info "Starting Universal Knowledge Platform production deployment"
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    
    # Check prerequisites
    check_prerequisites
    
    # Validate deployment
    validate_deployment
    
    # Create namespace
    create_namespace
    
    # Deploy secrets
    deploy_secrets
    
    # Deploy infrastructure
    deploy_infrastructure
    
    # Deploy monitoring
    deploy_monitoring
    
    # Run security scan
    run_security_scan
    
    # Deploy application
    deploy_blue_green
    
    # Run performance tests
    run_performance_tests
    
    # Final health check
    if perform_health_check; then
        log_success "🎉 Production deployment completed successfully!"
        log_info "Application is available at: https://api.ukp.example.com"
        log_info "Monitoring dashboard: https://grafana.ukp.example.com"
        log_info "MLflow UI: https://mlflow.ukp.example.com"
    else
        log_error "❌ Deployment failed - rolling back"
        rollback_deployment
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    "rollback")
        if [[ -z "$ROLLBACK_VERSION" ]]; then
            log_error "Rollback version is required"
            exit 1
        fi
        rollback_deployment
        ;;
    "health-check")
        perform_health_check
        ;;
    "performance-test")
        run_performance_tests
        ;;
    "security-scan")
        run_security_scan
        ;;
    *)
        main
        ;;
esac 