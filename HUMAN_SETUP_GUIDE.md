# 🚀 Universal Knowledge Platform - Human Setup Guide

This guide provides step-by-step instructions for all tasks that require human intervention to deploy and operate the Universal Knowledge Platform (SarvanOM).

## 📋 Table of Contents

1. [Prerequisites & Initial Setup](#1-prerequisites--initial-setup)
2. [Account Creation & API Keys](#2-account-creation--api-keys)
3. [Domain & SSL Setup](#3-domain--ssl-setup)
4. [AWS Infrastructure Setup](#4-aws-infrastructure-setup)
5. [External Services Configuration](#5-external-services-configuration)
6. [Security & Secrets Management](#6-security--secrets-management)
7. [Production Deployment](#7-production-deployment)
8. [Monitoring & Alerting](#8-monitoring--alerting)
9. [Team Setup & Access Management](#9-team-setup--access-management)
10. [Ongoing Operations](#10-ongoing-operations)

---

## 1. Prerequisites & Initial Setup

### 1.1 Required Tools Installation
```bash
# Install required tools on your local machine
# AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Helm
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

### 1.2 Initial Project Setup
- [ ] Clone the repository to your local machine
- [ ] Review the project structure and documentation
- [ ] Create a project management board (Jira, GitHub Projects, etc.)
- [ ] Set up version control branch protection rules

---

## 2. Account Creation & API Keys

### 2.1 AI Service Providers

#### OpenAI Account Setup
- [ ] **Visit**: https://platform.openai.com/
- [ ] **Sign up** for an OpenAI account
- [ ] **Add payment method** (required for API access)
- [ ] **Set spending limits**: Recommended $100/month initially
- [ ] **Generate API key**:
  1. Go to API Keys section
  2. Create new secret key
  3. Copy and store securely
  4. Note: Key shows only once!
- [ ] **Set usage alerts** at 50% and 80% of spending limit

#### Anthropic (Claude) Account Setup
- [ ] **Visit**: https://console.anthropic.com/
- [ ] **Apply for access** (may require approval)
- [ ] **Add payment information**
- [ ] **Generate API key** once approved
- [ ] **Configure rate limits** as needed

### 2.2 Vector Database Services

#### Pinecone Setup
- [ ] **Visit**: https://www.pinecone.io/
- [ ] **Create account** (free tier available)
- [ ] **Create new index**:
  - Name: `ukp-knowledge-base`
  - Dimensions: 1536 (for OpenAI embeddings)
  - Metric: cosine
  - Environment: Choose closest region
- [ ] **Generate API key**
- [ ] **Note environment name** for configuration

#### Qdrant Setup (Alternative/Additional)
- [ ] **Choose deployment**: Cloud vs Self-hosted
- [ ] **If cloud**: Sign up at https://cloud.qdrant.io/
- [ ] **Create cluster** with appropriate size
- [ ] **Generate API key**
- [ ] **Note cluster URL**

### 2.3 Cloud Provider Setup

#### AWS Account Setup
- [ ] **Create AWS account** at https://aws.amazon.com/
- [ ] **Verify email and phone number**
- [ ] **Add payment method**
- [ ] **Enable billing alerts**:
  ```bash
  aws budgets create-budget --account-id YOUR_ACCOUNT_ID --budget '{
    "BudgetName": "UKP-Monthly-Budget",
    "BudgetLimit": {"Amount": "200", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'
  ```
- [ ] **Set up root account MFA**
- [ ] **Create IAM admin user** (never use root for daily operations)

### 2.4 Container Registry
- [ ] **Choose option**:
  - GitHub Container Registry (free with GitHub)
  - Docker Hub (free tier available)
  - AWS ECR (pay per use)
- [ ] **Create account** if using external registry
- [ ] **Generate access tokens** for CI/CD

---

## 3. Domain & SSL Setup

### 3.1 Domain Registration
- [ ] **Choose domain registrar** (GoDaddy, Namecheap, Route53, etc.)
- [ ] **Register domain**: `your-domain.com`
- [ ] **Configure DNS settings**:
  ```
  Type: A Record
  Name: api
  Value: [Will be filled after AWS setup]
  TTL: 300
  
  Type: CNAME
  Name: www
  Value: your-domain.com
  TTL: 300
  ```

### 3.2 SSL Certificate Setup

#### Option A: AWS Certificate Manager (Recommended)
- [ ] **Log into AWS Console**
- [ ] **Navigate to Certificate Manager**
- [ ] **Request public certificate**:
  - Domain: `*.your-domain.com`
  - Validation: DNS validation
- [ ] **Add CNAME records** to your DNS provider for validation
- [ ] **Wait for validation** (can take up to 24 hours)
- [ ] **Copy certificate ARN** for Terraform configuration

#### Option B: Let's Encrypt (Free)
- [ ] **Install cert-manager** in Kubernetes cluster (post-deployment)
- [ ] **Configure ClusterIssuer** for automatic certificate management

---

## 4. AWS Infrastructure Setup

### 4.1 IAM Setup
- [ ] **Create IAM user** for Terraform:
  ```bash
  aws iam create-user --user-name terraform-user
  aws iam attach-user-policy --user-name terraform-user --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
  aws iam create-access-key --user-name terraform-user
  ```
- [ ] **Save access key and secret** securely
- [ ] **Configure AWS CLI**:
  ```bash
  aws configure
  # Enter access key, secret, region (us-east-1), output format (json)
  ```

### 4.2 S3 Bucket for Terraform State
- [ ] **Create S3 bucket** for Terraform state:
  ```bash
  aws s3 mb s3://ukp-terraform-state-YOUR_RANDOM_ID
  aws s3api put-bucket-versioning --bucket ukp-terraform-state-YOUR_RANDOM_ID --versioning-configuration Status=Enabled
  aws s3api put-bucket-encryption --bucket ukp-terraform-state-YOUR_RANDOM_ID --server-side-encryption-configuration '{
    "Rules": [
      {
        "ApplyServerSideEncryptionByDefault": {
          "SSEAlgorithm": "AES256"
        }
      }
    ]
  }'
  ```

### 4.3 Service Limits Check
- [ ] **Check current limits**:
  ```bash
  aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A
  aws service-quotas get-service-quota --service-code eks --quota-code L-1194D53C
  ```
- [ ] **Request increases if needed**:
  - EKS clusters per region: Default 100
  - EC2 instances: Check current limits
  - VPC per region: Default 5

---

## 5. External Services Configuration

### 5.1 Elasticsearch Setup

#### Option A: AWS OpenSearch
- [ ] **Navigate to OpenSearch** in AWS Console
- [ ] **Create domain**:
  - Domain name: `ukp-opensearch`
  - Instance type: t3.small.search (for dev)
  - Number of nodes: 1 (for dev), 3 (for prod)
  - Storage: 20GB EBS GP2
- [ ] **Configure access policy** to allow EKS access
- [ ] **Note endpoint URL**

#### Option B: Self-hosted
- [ ] **Deploy using Helm**:
  ```bash
  helm repo add elastic https://helm.elastic.co
  helm install elasticsearch elastic/elasticsearch
  ```

### 5.2 Neo4j Setup (Graph Database)

#### Option A: Neo4j AuraDB (Cloud)
- [ ] **Visit**: https://neo4j.com/cloud/aura/
- [ ] **Create account**
- [ ] **Create database instance**
- [ ] **Note connection URI and credentials**

#### Option B: Self-hosted
- [ ] **Deploy using Docker**:
  ```bash
  docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your-password \
    neo4j:latest
  ```

### 5.3 Redis Setup

#### Option A: AWS ElastiCache
- [ ] **Create Redis cluster** in AWS Console
- [ ] **Configure security groups** for EKS access
- [ ] **Note cluster endpoint**

#### Option B: Self-hosted
- [ ] **Deploy using Helm**:
  ```bash
  helm repo add bitnami https://charts.bitnami.com/bitnami
  helm install redis bitnami/redis
  ```

---

## 6. Security & Secrets Management

### 6.1 Generate Secure Secrets
- [ ] **Generate strong secrets**:
  ```bash
  # Generate random secrets
  openssl rand -base64 32  # For SECRET_KEY
  openssl rand -base64 32  # For API_KEY_SECRET
  openssl rand -base64 32  # For ENCRYPTION_KEY
  
  # Generate API keys for different access levels
  python -c "import secrets; print('admin-' + secrets.token_urlsafe(32))"
  python -c "import secrets; print('user-' + secrets.token_urlsafe(32))"
  python -c "import secrets; print('readonly-' + secrets.token_urlsafe(32))"
  ```

### 6.2 AWS Secrets Manager Setup
- [ ] **Create secrets in AWS Secrets Manager**:
  ```bash
  # OpenAI API Key
  aws secretsmanager create-secret \
    --name ukp/openai-api-key \
    --secret-string "your-openai-key"
  
  # Anthropic API Key
  aws secretsmanager create-secret \
    --name ukp/anthropic-api-key \
    --secret-string "your-anthropic-key"
  
  # Database password
  aws secretsmanager create-secret \
    --name ukp/database-password \
    --secret-string "your-secure-db-password"
  
  # Application secrets
  aws secretsmanager create-secret \
    --name ukp/app-secrets \
    --secret-string '{
      "SECRET_KEY": "your-secret-key",
      "API_KEY_SECRET": "your-api-key-secret",
      "ENCRYPTION_KEY": "your-encryption-key"
    }'
  ```

### 6.3 Environment File Setup
- [ ] **Copy template**:
  ```bash
  cp env.template .env
  ```
- [ ] **Fill in all values** in `.env` file
- [ ] **Never commit `.env`** to version control
- [ ] **Create separate `.env` files** for each environment (dev, staging, prod)

---

## 7. Production Deployment

### 7.1 Terraform Deployment
- [ ] **Configure Terraform variables**:
  ```bash
  cd infrastructure/terraform
  cp terraform.tfvars.example terraform.tfvars
  # Edit terraform.tfvars with your values
  ```
- [ ] **Update backend configuration** in `main.tf`:
  ```hcl
  backend "s3" {
    bucket = "your-terraform-state-bucket"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
  ```
- [ ] **Initialize and deploy**:
  ```bash
  terraform init
  terraform plan
  terraform apply
  ```

### 7.2 Kubernetes Deployment
- [ ] **Configure kubectl**:
  ```bash
  aws eks update-kubeconfig --region us-east-1 --name ukp-cluster
  ```
- [ ] **Create namespaces**:
  ```bash
  kubectl create namespace ukp-production
  kubectl create namespace ukp-monitoring
  ```
- [ ] **Deploy application**:
  ```bash
  kubectl apply -f infrastructure/kubernetes/production/
  ```

### 7.3 DNS Configuration
- [ ] **Get Load Balancer DNS**:
  ```bash
  kubectl get ingress -n ukp-production
  ```
- [ ] **Update DNS records** to point to load balancer
- [ ] **Verify SSL certificate** is working

---

## 8. Monitoring & Alerting

### 8.1 Prometheus & Grafana Setup
- [ ] **Install monitoring stack**:
  ```bash
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
  helm install prometheus prometheus-community/kube-prometheus-stack -n ukp-monitoring
  ```
- [ ] **Access Grafana**:
  ```bash
  kubectl port-forward -n ukp-monitoring svc/prometheus-grafana 3000:80
  # Default: admin/prom-operator
  ```
- [ ] **Import dashboards** for application metrics

### 8.2 Alerting Configuration

#### PagerDuty Setup
- [ ] **Create PagerDuty account**
- [ ] **Create service** for UKP alerts
- [ ] **Generate integration key**
- [ ] **Configure escalation policies**

#### Slack Integration
- [ ] **Create Slack workspace** or use existing
- [ ] **Create #alerts channel**
- [ ] **Add webhook integration**
- [ ] **Configure alertmanager** to send notifications

### 8.3 Log Management
- [ ] **Configure log aggregation**:
  ```bash
  helm repo add elastic https://helm.elastic.co
  helm install elasticsearch elastic/elasticsearch -n ukp-monitoring
  helm install kibana elastic/kibana -n ukp-monitoring
  helm install filebeat elastic/filebeat -n ukp-monitoring
  ```

---

## 9. Team Setup & Access Management

### 9.1 AWS IAM for Team
- [ ] **Create IAM group** for developers:
  ```bash
  aws iam create-group --group-name ukp-developers
  aws iam attach-group-policy --group-name ukp-developers --policy-arn arn:aws:iam::aws:policy/ReadOnlyAccess
  ```
- [ ] **Create developer policies** for specific access needs
- [ ] **Add team members** to appropriate groups

### 9.2 Kubernetes RBAC
- [ ] **Create service accounts** for different roles
- [ ] **Configure role bindings**:
  ```yaml
  apiVersion: rbac.authorization.k8s.io/v1
  kind: ClusterRoleBinding
  metadata:
    name: ukp-developers
  subjects:
  - kind: User
    name: developer@company.com
    apiGroup: rbac.authorization.k8s.io
  roleRef:
    kind: ClusterRole
    name: view
    apiGroup: rbac.authorization.k8s.io
  ```

### 9.3 VPN Setup (if required)
- [ ] **Choose VPN solution** (AWS Client VPN, OpenVPN, etc.)
- [ ] **Configure VPN server**
- [ ] **Generate client certificates**
- [ ] **Distribute access credentials** to team

---

## 10. Ongoing Operations

### 10.1 Backup Strategy
- [ ] **Configure automated backups**:
  - RDS automated backups (7-30 days)
  - EBS snapshot schedules
  - Application data backup procedures
- [ ] **Test backup restoration** procedures
- [ ] **Document recovery processes**

### 10.2 Security Auditing
- [ ] **Enable AWS CloudTrail** for audit logging
- [ ] **Configure AWS Config** for compliance monitoring
- [ ] **Set up security scanning**:
  ```bash
  # Example: Trivy for container scanning
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image your-app:latest
  ```
- [ ] **Schedule regular security reviews**

### 10.3 Cost Optimization
- [ ] **Set up cost allocation tags**
- [ ] **Configure billing alerts**:
  ```bash
  aws budgets create-budget --account-id YOUR_ACCOUNT_ID --budget '{
    "BudgetName": "UKP-Cost-Alert",
    "BudgetLimit": {"Amount": "500", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80
    },
    "Subscribers": [{
      "SubscriptionType": "EMAIL",
      "Address": "admin@your-domain.com"
    }]
  }]'
  ```
- [ ] **Review and optimize** monthly

### 10.4 Performance Monitoring
- [ ] **Set up performance baselines**
- [ ] **Configure auto-scaling policies**
- [ ] **Monitor and tune** application performance
- [ ] **Regular load testing** schedule

### 10.5 Disaster Recovery
- [ ] **Document disaster recovery procedures**
- [ ] **Create runbooks** for common incidents
- [ ] **Test disaster recovery** quarterly
- [ ] **Maintain off-site backups**

---

## 📊 Checklist Summary

### Phase 1: Foundation (Week 1)
- [ ] All accounts created and configured
- [ ] Domain and SSL certificates ready
- [ ] AWS infrastructure provisioned
- [ ] Basic monitoring in place

### Phase 2: Security & Production (Week 2)
- [ ] All secrets properly managed
- [ ] Production deployment successful
- [ ] Monitoring and alerting configured
- [ ] Team access established

### Phase 3: Operations (Week 3-4)
- [ ] Backup and recovery tested
- [ ] Performance monitoring active
- [ ] Cost optimization implemented
- [ ] Documentation complete

---

## 🆘 Emergency Contacts

**AWS Support**: Create support case in AWS Console
**OpenAI Support**: help@openai.com
**Anthropic Support**: support@anthropic.com
**Domain Registrar**: [Your registrar's support]
**Team Lead**: [Primary contact]
**DevOps Engineer**: [Secondary contact]

---

## 📚 Additional Resources

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/)
- [Terraform Best Practices](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FastAPI Production Guide](https://fastapi.tiangolo.com/deployment/)

---

**Note**: This guide assumes you're deploying to AWS. Adjust cloud-specific steps if using Azure or GCP. Always test in a development environment before applying to production. 