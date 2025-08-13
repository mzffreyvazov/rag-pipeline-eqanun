#!/bin/bash

# Production deployment script for Digital Ocean or similar VPS

echo "🚀 Starting production deployment of Azerbaijani Legal RAG Pipeline..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Install nginx for reverse proxy
if ! command -v nginx &> /dev/null; then
    echo "🌐 Installing Nginx..."
    sudo apt install nginx -y
fi

# Create application directory
APP_DIR="/opt/rag-pipeline"
echo "📁 Creating application directory: $APP_DIR"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Clone or copy application files
if [ -d ".git" ]; then
    echo "📋 Copying application files..."
    cp -r . $APP_DIR/
else
    echo "📋 Application files should be in current directory"
    echo "Please ensure all files are present and run this script from the project root"
fi

cd $APP_DIR

# Set up environment file
if [ ! -f ".env" ]; then
    echo "📝 Setting up environment file..."
    cp .env.example .env
    echo "⚠️ Please edit $APP_DIR/.env with your actual credentials"
    echo "Required variables: GOOGLE_API_KEY, PROJECT_ID, REGION"
fi

# Create necessary directories with proper permissions
echo "📁 Creating data directories..."
sudo mkdir -p /var/lib/rag-pipeline/{chroma_data,uploads,logs}
sudo chown -R $USER:$USER /var/lib/rag-pipeline

# Create systemd service file
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/rag-pipeline.service > /dev/null <<EOF
[Unit]
Description=Azerbaijani Legal RAG Pipeline
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=$USER
Group=$USER

[Install]
WantedBy=multi-user.target
EOF

# Create nginx configuration
echo "🌐 Configuring Nginx..."
sudo tee /etc/nginx/sites-available/rag-pipeline > /dev/null <<'EOF'
server {
    listen 80;
    server_name _;
    
    client_max_body_size 100M;
    client_body_timeout 300s;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Handle large file uploads
        proxy_request_buffering off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
    }
    
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/rag-pipeline /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t

# Configure firewall
echo "🔥 Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

# Set up log rotation
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/rag-pipeline > /dev/null <<EOF
/var/lib/rag-pipeline/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    postrotate
        docker-compose -f $APP_DIR/docker-compose.yml restart rag-pipeline
    endscript
}
EOF

# Build and start services
echo "🏗️ Building and starting services..."
docker-compose build
docker-compose up -d

# Enable services
sudo systemctl daemon-reload
sudo systemctl enable rag-pipeline
sudo systemctl restart nginx

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Application is running successfully!"
else
    echo "❌ Application health check failed"
    echo "Check logs with: docker-compose logs"
fi

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit $APP_DIR/.env with your Google Cloud credentials"
echo "2. Restart the service: sudo systemctl restart rag-pipeline"
echo "3. Check status: sudo systemctl status rag-pipeline"
echo "4. View logs: docker-compose -f $APP_DIR/docker-compose.yml logs"
echo ""
echo "🌐 Your API is available at:"
echo "   - HTTP: http://$(curl -s ifconfig.me || hostname -I | awk '{print $1}')"
echo "   - Documentation: http://$(curl -s ifconfig.me || hostname -I | awk '{print $1}')/docs"
echo ""
echo "🔧 Useful commands:"
echo "   - Restart: sudo systemctl restart rag-pipeline"
echo "   - Stop: sudo systemctl stop rag-pipeline"
echo "   - Logs: docker-compose -f $APP_DIR/docker-compose.yml logs -f"
echo "   - Update: cd $APP_DIR && git pull && docker-compose build && sudo systemctl restart rag-pipeline"
