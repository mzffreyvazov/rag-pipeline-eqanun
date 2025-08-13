# Azerbaijani Legal RAG Pipeline

A FastAPI-based conversational RAG (Retrieval-Augmented Generation) system for Azerbaijani legal documents. This application allows users to upload PDF documents and chat with them using advanced AI capabilities.

## Features

- 📄 **PDF Document Upload**: Support for multiple PDF file uploads
- 🔍 **Intelligent Retrieval**: ChromaDB-powered vector search
- 💬 **Conversational AI**: Session-based chat with memory
- 🌐 **REST API**: Full FastAPI implementation with automatic documentation
- 🔄 **Persistent Storage**: ChromaDB for lasting document storage
- 🚀 **Production Ready**: Docker support and deployment scripts

## Technology Stack

- **FastAPI**: Modern web framework for building APIs
- **ChromaDB**: Vector database for document storage and retrieval
- **LangChain**: Framework for LLM applications
- **Google Gemini**: Large language model for chat responses
- **Vertex AI**: Embedding generation
- **LangGraph**: Conversation flow management

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Cloud Platform account with Vertex AI enabled
- Google API key for Gemini

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-pipeline-eqanun
   ```

2. **Run setup script**:
   
   **Linux/macOS**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   
   **Windows**:
   ```batch
   setup.bat
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

4. **Start the application**:
   ```bash
   # Activate virtual environment
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   
   # Run the application
   python app/main.py
   ```

### Using Docker

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**:
   ```bash
   docker build -t rag-pipeline .
   docker run -p 8000:8000 -v $(pwd)/chroma_data:/app/chroma_data rag-pipeline
   ```

## API Documentation

Once the application is running, visit:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Document Management

- `POST /upload` - Upload PDF documents
- `DELETE /documents` - Clear all documents
- `GET /status` - Get system status

### Chat Interface

- `POST /chat` - Send a message to the AI assistant
- `GET /health` - Health check endpoint

### Example Usage

#### Upload Documents
```python
import requests

files = [('files', open('document.pdf', 'rb'))]
response = requests.post('http://localhost:8000/upload', files=files)
print(response.json())
```

#### Chat with Documents
```python
import requests

chat_data = {
    "message": "Əmək məcəlləsinə əsasən Müəssisə anlayışı nədir?",
    "session_id": "user123"
}
response = requests.post('http://localhost:8000/chat', json=chat_data)
print(response.json())
```

## Environment Variables

Create a `.env` file with the following variables:

```env
# Google Cloud credentials
GOOGLE_API_KEY=your_google_api_key_here
PROJECT_ID=your_gcp_project_id
REGION=your_gcp_region

# LangSmith (optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

## Project Structure

```
rag-pipeline-eqanun/
├── app/
│   └── main.py              # Main FastAPI application
├── assets/                  # Sample PDF documents
├── chroma_data/            # ChromaDB persistent storage
├── uploads/                # Temporary file uploads
├── logs/                   # Application logs
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── nginx.conf             # Nginx reverse proxy config
├── setup.sh               # Linux/macOS setup script
├── setup.bat              # Windows setup script
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Deployment

### Local Development
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Digital Ocean Droplet Deployment

1. **Create a droplet** and connect via SSH
2. **Install Docker and Docker Compose**:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   sudo apt install docker-compose
   ```

3. **Clone and deploy**:
   ```bash
   git clone <your-repository>
   cd rag-pipeline-eqanun
   cp .env.example .env
   # Edit .env with your credentials
   sudo docker-compose up -d
   ```

4. **Set up reverse proxy** (optional):
   ```bash
   sudo docker-compose --profile with-proxy up -d
   ```

## Features Details

### Document Processing
- Supports multiple PDF files simultaneously
- Automatic text extraction and chunking
- Metadata preservation for source tracking
- Persistent storage in ChromaDB

### Conversational AI
- Session-based conversations with memory
- Context-aware responses using retrieved documents
- Azerbaijani language support
- Intelligent query routing (tool use vs. direct response)

### Vector Search
- ChromaDB for efficient similarity search
- Vertex AI embeddings for high-quality vector representations
- Configurable retrieval parameters
- Metadata filtering capabilities

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **ChromaDB permission errors**:
   ```bash
   sudo chown -R $USER:$USER chroma_data/
   ```

3. **Google Cloud authentication**:
   - Ensure your API key is valid
   - Check that Vertex AI is enabled in your GCP project
   - Verify the PROJECT_ID and REGION are correct

### Logs
Check application logs for detailed error information:
```bash
# If running with Docker
docker-compose logs rag-pipeline

# If running locally
tail -f logs/app.log
```

## Performance Optimization

- **Batch Processing**: Documents are processed in batches for efficiency
- **Connection Pooling**: Optimized database connections
- **Caching**: Intelligent caching of embeddings and responses
- **Memory Management**: Proper cleanup of temporary files

## Security Considerations

- **File Validation**: Only PDF files are accepted
- **Input Sanitization**: All user inputs are validated
- **CORS Configuration**: Configure CORS for production
- **Rate Limiting**: Consider adding rate limiting for production use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs`
