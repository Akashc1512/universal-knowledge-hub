# Universal Knowledge Platform - Infrastructure as Code

This directory contains Terraform configurations for deploying the Universal Knowledge Platform on AWS.

## Architecture Overview

The infrastructure includes:

- **VPC** with public and private subnets across 3 availability zones
- **EKS Cluster** with managed node groups for scalability
- **RDS PostgreSQL** for persistent data storage
- **ElastiCache Redis** for caching and session storage
- **S3 Bucket** for file storage with versioning and encryption
- **DynamoDB** for session management
- **Application Load Balancer** for traffic distribution
- **Security Groups** with least-privilege access

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.0 installed
3. **kubectl** for Kubernetes management
4. **AWS S3 bucket** for Terraform state (create manually)

### Create S3 Bucket for Terraform State

```bash
aws s3 mb s3://universal-knowledge-hub-terraform-state
aws s3api put-bucket-versioning --bucket universal-knowledge-hub-terraform-state --versioning-configuration Status=Enabled
```

## Quick Start

1. **Clone and navigate to the terraform directory:**
   ```bash
   cd infra/terraform
   ```

2. **Copy the example variables file:**
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

3. **Edit terraform.tfvars with your configuration:**
   ```bash
   # Modify the variables as needed
   vim terraform.tfvars
   ```

4. **Initialize Terraform:**
   ```bash
   terraform init
   ```

5. **Plan the deployment:**
   ```bash
   terraform plan
   ```

6. **Apply the configuration:**
   ```bash
   terraform apply
   ```

7. **Configure kubectl for the new cluster:**
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name universal-knowledge-hub-cluster
   ```

## Environment Management

### Development Environment

```bash
# Create dev environment
terraform workspace new dev
terraform plan -var-file=dev.tfvars
terraform apply -var-file=dev.tfvars
```

### Staging Environment

```bash
# Create staging environment
terraform workspace new staging
terraform plan -var-file=staging.tfvars
terraform apply -var-file=staging.tfvars
```

### Production Environment

```bash
# Create production environment
terraform workspace new prod
terraform plan -var-file=prod.tfvars
terraform apply -var-file=prod.tfvars
```

## Configuration Files

### terraform.tfvars

Main configuration file with all variables. Example:

```hcl
aws_region = "us-east-1"
environment = "dev"
project_name = "universal-knowledge-hub"
cluster_name = "universal-knowledge-hub-cluster"
```

### Environment-specific files

Create environment-specific variable files:

- `dev.tfvars` - Development environment
- `staging.tfvars` - Staging environment  
- `prod.tfvars` - Production environment

## Security Considerations

- All resources are deployed in private subnets where possible
- Security groups follow least-privilege principle
- RDS and Redis are not publicly accessible
- S3 bucket has public access blocked
- EKS cluster uses AWS IAM for authentication

## Monitoring and Logging

The infrastructure includes:

- **CloudWatch Logs** for centralized logging
- **VPC Flow Logs** for network monitoring
- **EKS CloudWatch integration** for cluster monitoring

## Cost Optimization

- Use Spot instances for non-critical workloads
- Enable RDS storage autoscaling
- Use S3 lifecycle policies for cost management
- Monitor and optimize EKS node group sizing

## Troubleshooting

### Common Issues

1. **Terraform state lock issues:**
   ```bash
   terraform force-unlock <lock-id>
   ```

2. **EKS cluster access issues:**
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name universal-knowledge-hub-cluster
   ```

3. **RDS connection issues:**
   - Verify security group rules
   - Check VPC and subnet configuration

### Useful Commands

```bash
# View current state
terraform show

# List resources
terraform state list

# Import existing resources
terraform import aws_s3_bucket.example example-bucket

# Destroy specific resources
terraform destroy -target=aws_rds_cluster.example
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning:** This will permanently delete all resources. Make sure you have backups of important data.

## Next Steps

After infrastructure deployment:

1. Deploy the application using Helm charts
2. Configure monitoring and alerting
3. Set up CI/CD pipelines
4. Configure backup and disaster recovery
5. Implement security hardening

## Support

For issues and questions:

1. Check the [main project README](../../README.md)
2. Review Terraform documentation
3. Check AWS service documentation
4. Open an issue in the project repository 