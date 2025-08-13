#!/bin/bash

# Azerbaijani Legal RAG Pipeline - Deployment Script

echo "🚀 Starting Azerbaijani Legal RAG Pipeline deployment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found. Please copy .env.example to .env and fill in your credentials."
    cp .env.example .env
    echo "📝 Created .env file from template. Please edit it with your actual credentials."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p chroma_data
mkdir -p uploads
mkdir -p logs

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit the .env file with your Google Cloud credentials"
echo "2. Run the application with: python app/main.py"
echo "   Or with uvicorn: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "🌐 The API will be available at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"
