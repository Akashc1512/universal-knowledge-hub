# SarvanOM Production Deployment Guide

This guide covers deploying the Universal Knowledge Platform to production using the sarvanom.com domain.

## 🚀 Quick Deployment

### Prerequisites

- VPS or cloud server (Ubuntu 20.04+ recommended)
- Domain: sarvanom.com (already acquired)
- Docker and Docker Compose installed
- Basic Linux administration knowledge

### 1. Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
```

### 2. Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/universal-knowledge-hub.git
cd universal-knowledge-hub

# Copy production environment file
cp production.env.example .env.production
```

### 3. Configure Environment

Edit `.env.production` with your secure values:

```bash
# Generate secure API keys
openssl rand -hex 32

# Edit the environment file
nano .env.production
```

**Critical Security Steps:**
- Change all default API keys
- Use strong passwords for databases
- Enable rate limiting and security features
- Set up proper SSL certificates

### 4. DNS Configuration

Configure your domain registrar:

```
A     sarvanom.com     → Your server IP
CNAME www.sarvanom.com → sarvanom.com
A     api.sarvanom.com → Your server IP
```

### 5. SSL Certificate Setup

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot certonly --standalone -d sarvanom.com -d www.sarvanom.com -d api.sarvanom.com

# Set up auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 6. Deploy Application

```bash
# Build and start services
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### 7. Verify Deployment

```bash
# Test health endpoints
curl https://sarvanom.com
curl https://api.sarvanom.com/health

# Check SSL certificate
curl -I https://sarvanom.com
```

## 🔧 Advanced Configuration

### Nginx Configuration

Create `nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server frontend:3000;
    }

    upstream backend {
        server backend:8002;
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name sarvanom.com www.sarvanom.com;
        return 301 https://$server_name$request_uri;
    }

    # Frontend (sarvanom.com)
    server {
        listen 443 ssl http2;
        server_name sarvanom.com www.sarvanom.com;

        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;

        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

    # API (api.sarvanom.com)
    server {
        listen 443 ssl http2;
        server_name api.sarvanom.com;

        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;

        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### Monitoring Setup

```bash
# Access Grafana
# URL: https://sarvanom.com:3001
# Username: admin
# Password: (from GRAFANA_PASSWORD in .env.production)

# Access Prometheus
# URL: https://sarvanom.com:9090
```

### Backup Strategy

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/sarvanom_$DATE"

mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker exec sarvanom-postgres-prod pg_dump -U sarvanom_user sarvanom_prod > $BACKUP_DIR/database.sql

# Backup application data
tar -czf $BACKUP_DIR/app_data.tar.gz data/ logs/ cache/

# Backup configuration
cp .env.production $BACKUP_DIR/
cp docker-compose.prod.yml $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh

# Set up automated backups
echo "0 2 * * * /path/to/backup.sh" | crontab -
```

## 🔒 Security Hardening

### Firewall Configuration

```bash
# Install UFW
sudo apt install ufw

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

### Security Headers

Add to nginx configuration:

```nginx
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header X-Content-Type-Options "nosniff" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
```

### Rate Limiting

```nginx
# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=web:10m rate=30r/s;

location /api/ {
    limit_req zone=api burst=20 nodelay;
    proxy_pass http://backend;
}
```

## 📊 Monitoring and Alerts

### Health Checks

```bash
# Create health check script
cat > health_check.sh << 'EOF'
#!/bin/bash

# Check frontend
if ! curl -f https://sarvanom.com > /dev/null 2>&1; then
    echo "Frontend is down!"
    # Send alert
fi

# Check backend
if ! curl -f https://api.sarvanom.com/health > /dev/null 2>&1; then
    echo "Backend is down!"
    # Send alert
fi

# Check disk space
if [ $(df / | awk 'NR==2 {print $5}' | sed 's/%//') -gt 80 ]; then
    echo "Disk space is low!"
    # Send alert
fi
EOF

chmod +x health_check.sh

# Add to crontab
echo "*/5 * * * * /path/to/health_check.sh" | crontab -
```

### Log Monitoring

```bash
# Set up log rotation
sudo nano /etc/logrotate.d/sarvanom

# Add:
/var/log/sarvanom/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 root root
}
```

## 🚀 Scaling Considerations

### Horizontal Scaling

```yaml
# docker-compose.prod.yml
services:
  backend:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### Load Balancer

```nginx
# Multiple backend instances
upstream backend {
    server backend1:8002;
    server backend2:8002;
    server backend3:8002;
}
```

## 🔄 Updates and Maintenance

### Application Updates

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.prod.yml --env-file .env.production down
docker-compose -f docker-compose.prod.yml --env-file .env.production build --no-cache
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
```

### Database Migrations

```bash
# Run migrations
docker exec sarvanom-backend-prod python -m alembic upgrade head
```

### SSL Certificate Renewal

```bash
# Certbot auto-renewal is already configured
# Manual renewal if needed:
sudo certbot renew
docker-compose -f docker-compose.prod.yml restart nginx
```

## 🆘 Troubleshooting

### Common Issues

1. **SSL Certificate Issues**:
   ```bash
   # Check certificate
   sudo certbot certificates
   
   # Renew manually
   sudo certbot renew --force-renewal
   ```

2. **Database Connection Issues**:
   ```bash
   # Check database logs
   docker logs sarvanom-postgres-prod
   
   # Test connection
   docker exec sarvanom-postgres-prod psql -U sarvanom_user -d sarvanom_prod
   ```

3. **Memory Issues**:
   ```bash
   # Check memory usage
   docker stats
   
   # Increase swap if needed
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Performance Optimization

```bash
# Monitor performance
docker stats

# Optimize PostgreSQL
docker exec sarvanom-postgres-prod psql -U sarvanom_user -d sarvanom_prod -c "
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
"
```

## 📞 Support

- **Documentation**: [https://sarvanom.com/docs](https://sarvanom.com/docs)
- **Issues**: [GitHub Issues](https://github.com/your-org/universal-knowledge-hub/issues)
- **Security**: security@sarvanom.com
- **Support**: support@sarvanom.com

---

**Domain**: [https://sarvanom.com](https://sarvanom.com)

This deployment guide ensures a secure, scalable, and maintainable production environment for SarvanOM. 