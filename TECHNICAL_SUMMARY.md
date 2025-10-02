# Technical Summary - Azerbaijani Legal RAG Pipeline

## 📋 Overview

This is a production-ready **Retrieval-Augmented Generation (RAG)** system built for Azerbaijani legal documents. The system enables users to upload PDF documents and interact with them through an AI-powered conversational interface using advanced natural language processing.

**Primary Use Case:** Legal document Q&A system for Azerbaijani law (Mecelleler - legal codes)

---

## 🏗️ Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Application                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Auth       │  │   Document   │  │   Chat       │  │
│  │   Layer      │  │   Processing │  │   Interface  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
          ↓                    ↓                    ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  JWT/Supabase   │  │   ChromaDB      │  │  Google Gemini  │
│  Authentication │  │  Vector Store   │  │  LLM + Vertex   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI | High-performance async API server |
| **Vector Database** | ChromaDB | Document embeddings and semantic search |
| **Embeddings** | Google Vertex AI | `gemini-embedding-001` (1024 dimensions) |
| **LLM** | Google Gemini | `gemini-2.5-flash` for chat responses |
| **Document Processing** | LangChain | Document loaders, text splitting, RAG chain |
| **Conversation Flow** | LangGraph | Stateful conversation management with memory |
| **Authentication** | JWT / Supabase Auth | Dual-mode authentication system |
| **Security** | python-jose, bcrypt | Token signing, password hashing |
| **Deployment** | Docker, Docker Compose | Containerization with Nginx proxy |

---

## 🔑 Core Features

### 1. **Hierarchical Document Chunking**
- **Implementation:** `hierarchical_chunker.py`
- **Strategy:** Parent-child chunking for legal documents
- **Parent Chunks:** 2000 chars (broader context)
- **Child Chunks:** 500 chars (precise matching)
- **Structure Preservation:** Maintains Document → Section → Chapter → Article hierarchy
- **Metadata Enrichment:** Extracts clause numbers, article headers, legal references

### 2. **Dual Authentication System**

#### Option A: Local JWT Authentication
- Built-in JWT token management
- In-memory user store (demo - replace with database)
- Demo credentials: `demo` / `demo1234`
- Access tokens: 30 minutes
- Refresh tokens: 7 days
- Endpoints: `/auth/login`, `/auth/register`, `/auth/refresh`, `/auth/me`

#### Option B: Supabase Authentication (Recommended for Production)
- Integration with Supabase Auth service
- JWKS-based token verification (public key validation)
- No backend database required
- Email verification, social login support
- Token blacklist support with Redis (optional)
- Endpoints: `/auth/supabase/login`, `/auth/supabase/register`, `/auth/supabase/refresh`

**Configuration:** Set `AUTH_PROVIDER=local` or `AUTH_PROVIDER=supabase` in `.env`

### 3. **Admin Authorization**
- Restricted endpoints (`/upload`, `/documents DELETE`)
- Admin list configured via environment variables
- Supports both email and UUID-based authorization
- Config: `ADMIN_EMAILS`, `ADMIN_USER_IDS`

### 4. **Advanced RAG Pipeline**
- **Retrieval:** Hierarchical semantic search using ChromaDB
- **Generation:** Context-aware responses with source attribution
- **Conversation Memory:** Session-based chat history via LangGraph
- **Tool-based Architecture:** Automatic retrieval for every query
- **Source Tracking:** Documents returned with metadata for citation

---

## 📁 Project Structure

```
rag-pipeline-eqanun/
├── app/
│   ├── main.py                 # Main FastAPI application
│   ├── config/
│   │   ├── security.py         # Security settings (Pydantic)
│   ├── security/
│   │   ├── auth.py             # Legacy API key auth
│   │   ├── auth_routes.py      # Local JWT auth endpoints
│   │   ├── jwt_auth.py         # JWT token creation/verification
│   │   ├── supabase_auth.py    # Supabase JWT verification
│   │   ├── supabase_auth_routes.py  # Supabase auth endpoints
│   │   └── uploads.py          # File validation utilities
│
├── hierarchical_chunker.py     # Advanced document chunking
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── Dockerfile                  # Container image definition
├── docker-compose.yml          # Multi-container orchestration
├── nginx.conf                  # Reverse proxy config
├── setup.sh / setup.bat        # Development setup scripts
├── deploy.sh                   # Production deployment script
│
├── chroma_data/                # Persistent vector storage
├── uploads/                    # Temporary file uploads
├── retrieval_logs/             # Query/response logging
└── assets/mecelleler/          # Legal document sources
```

---

## ⚙️ Configuration Files

### 1. **Environment Variables (.env)**

**Required for All Deployments:**
```bash
# Google Cloud
GOOGLE_API_KEY=<your-api-key>
PROJECT_ID=<gcp-project-id>
REGION=<gcp-region>

# Embeddings
EMBEDDING_PROVIDER=vertexai          # "vertexai" or "google-genai"
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIM=1024

# Authentication Provider
AUTH_PROVIDER=local                  # "local" or "supabase"
```

**For Local JWT Auth:**
```bash
JWT_SECRET_KEY=<generate-secure-key>  # Required!
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

**For Supabase Auth:**
```bash
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=<anon-key>
SUPABASE_JWT_SECRET=<jwt-secret>      # Optional
SUPABASE_JWT_AUDIENCE=authenticated
```

**Admin Configuration:**
```bash
ADMIN_EMAILS=admin@example.com,another@example.com
ADMIN_USER_IDS=uuid-1,uuid-2
```

**Security Features:**
```bash
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
UPLOAD_MAX_MEGABYTES=50
ENABLE_JWT_AUTH=true
ENABLE_FILE_VALIDATION=true
```

### 2. **Dockerfile**

**Base Image:** `python:3.11-slim`

**Key Features:**
- Multi-stage build optimization
- System dependencies: `build-essential`, `curl`
- Application port: `8000`
- Health check endpoint: `/health`
- Persistent volumes: `chroma_data`, `uploads`, `logs`

**Startup Command:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. **docker-compose.yml**

**Services:**

1. **rag-pipeline** (main application)
   - Port: `8000:8000`
   - Volumes: ChromaDB data, uploads, logs, `.env`
   - Health check: 30s interval
   - Restart policy: `unless-stopped`

2. **nginx** (optional reverse proxy)
   - Ports: `80:80`, `443:443`
   - Profile: `with-proxy` (opt-in)
   - Max upload size: 100MB
   - Timeout: 300s

**Usage:**
```bash
# Start main service
docker-compose up -d

# Start with Nginx proxy
docker-compose --profile with-proxy up -d
```

### 4. **nginx.conf**

**Configuration Highlights:**
- Upstream: `rag-pipeline:8000`
- Max body size: `100M` (large PDFs)
- Proxy buffering: `off` (streaming uploads)
- Timeouts: 300s (long-running operations)
- Health check: `/health` endpoint (no logs)

### 5. **requirements.txt**

**Core Dependencies:**
```
fastapi                      # Web framework
uvicorn[standard]            # ASGI server
chromadb                     # Vector database
langchain                    # RAG framework
langchain-google-genai       # Gemini integration
langchain-google-vertexai    # Vertex AI embeddings
langgraph                    # Conversation management
```

**Security:**
```
python-jose[cryptography]    # JWT signing
bcrypt                       # Password hashing
redis                        # Token blacklist (optional)
```

**Document Processing:**
```
pypdf                        # PDF parsing
docling                      # Document formatting
langchain-text-splitters     # Text chunking
```

### 6. **deploy.sh** (Production Deployment)

**Automated Deployment for VPS/Digital Ocean:**

**Installation Steps:**
1. System updates (`apt update && upgrade`)
2. Docker & Docker Compose installation
3. Nginx installation for reverse proxy
4. Application directory setup (`/opt/rag-pipeline`)
5. systemd service creation
6. UFW firewall configuration (ports 22, 80, 443)
7. Log rotation setup

**systemd Service:**
- Service name: `rag-pipeline.service`
- Auto-start: Enabled
- Restart: Automatic
- Requires: `docker.service`

**Post-Deployment:**
- Health check validation
- API availability confirmation
- Documentation links

---

## 🔐 Security Architecture

### Authentication Flow

```
┌──────────┐                 ┌──────────┐                 ┌──────────┐
│  Client  │───Login────────>│  FastAPI │──Validate──────>│ Auth     │
│          │                 │          │                 │ Provider │
└──────────┘                 └──────────┘                 └──────────┘
     │                             │                             │
     │<───Access Token─────────────┤                             │
     │                             │                             │
     │───API Request + Token──────>│                             │
     │                             │──Verify Token──────────────>│
     │                             │<──User Data─────────────────│
     │<───API Response─────────────┤                             │
```

### Security Layers

1. **CORS Protection:** Configurable allowed origins
2. **JWT Validation:** Token signature and expiration checks
3. **Admin Authorization:** Role-based access control
4. **File Validation:** MIME type and extension checking
5. **Size Limits:** Configurable upload size restrictions
6. **Security Headers:** CSP, HSTS, X-Frame-Options, etc.
7. **API Rate Limiting:** (Recommended - not yet implemented)

### Middleware Stack

```python
SecurityHeadersMiddleware      # HTTP security headers
CORSMiddleware                 # Cross-origin requests
HTTPBearer (Optional)          # Bearer token extraction
```

---

## 🚀 API Endpoints

### Public Endpoints
- `GET /` - API status
- `GET /health` - Health check
- `GET /docs` - OpenAPI documentation

### Authentication Endpoints

**Local JWT:**
- `POST /auth/login` - Login (returns access + refresh tokens)
- `POST /auth/register` - User registration
- `POST /auth/refresh` - Refresh access token
- `GET /auth/me` - Current user info

**Supabase:**
- `POST /auth/supabase/login` - Supabase login
- `POST /auth/supabase/register` - Supabase registration
- `POST /auth/supabase/refresh` - Refresh tokens
- `POST /auth/supabase/reset-password` - Password reset

### Protected Endpoints (Require Authentication)

**Document Management:**
- `POST /upload` - Upload PDF documents (admin only)
- `POST /upload/start` - Async upload job (admin only)
- `GET /upload/status/{job_id}` - Check upload progress
- `GET /documents` - List processed documents
- `DELETE /documents` - Clear all documents (admin only)

**Chat & Retrieval:**
- `POST /chat` - Send message to AI assistant
- `POST /retrieve` - Retrieve relevant document chunks
- `GET /status` - System status

---

## 🔄 RAG Pipeline Flow

```
User Query
    ↓
[1] Query Preprocessing
    ↓
[2] Hierarchical Retrieval
    ├── Search child chunks (precision)
    ├── Retrieve parent chunks (context)
    └── Return adaptive results
    ↓
[3] Context Assembly
    ├── Format retrieved documents
    ├── Add source metadata
    └── Create RAG prompt
    ↓
[4] LLM Generation (Gemini)
    ├── System prompt (legal expert)
    ├── Retrieved context
    └── User query
    ↓
[5] Response Formation
    ├── AI answer
    ├── Source citations
    └── Metadata
    ↓
User Response
```

### Conversation State Management (LangGraph)

```python
MessagesState
├── messages: List[BaseMessage]
├── session_id: str
└── conversation_history: List

StateGraph
├── query_or_respond (retrieval trigger)
├── retrieve (tool execution)
└── generate (LLM response)
```

---

## 📊 Data Models

### Key Pydantic Models

```python
# Chat
ChatRequest(message, session_id)
ChatResponse(answer, sources, session_id, metadata)

# Documents
UploadResponse(message, documents_processed, chunks_created)
DocumentInfo(id, filename, source_document, chunk_type, content_preview)

# Authentication
TokenPair(access_token, refresh_token, expires_in)
UserResponse(user_id, username, email, full_name)

# Retrieval
RetrieveResponse(documents, query, total_results)
SourceInfo(document_name, chunk_id, relevance_score)
```

---

## 🛠️ Development Setup

### Quick Start (Windows)

```cmd
# Run setup script
setup.bat

# Activate virtual environment
venv\Scripts\activate.bat

# Edit .env file
copy .env.example .env
notepad .env

# Run application
uvicorn app.main:app --reload
```

### Quick Start (Linux/Mac)

```bash
# Run setup script
bash setup.sh

# Activate virtual environment
source venv/bin/activate

# Edit .env file
cp .env.example .env
nano .env

# Run application
uvicorn app.main:app --reload
```

### Docker Development

```bash
# Build image
docker build -t rag-pipeline .

# Run container
docker run -p 8000:8000 --env-file .env rag-pipeline

# Or use docker-compose
docker-compose up -d
```

---

## 📦 Production Deployment

### Option 1: VPS Deployment (Digital Ocean, AWS EC2, etc.)

```bash
# Upload code to server
scp -r . user@your-server:/opt/rag-pipeline

# SSH into server
ssh user@your-server

# Run deployment script
cd /opt/rag-pipeline
sudo bash deploy.sh

# Edit environment variables
sudo nano /opt/rag-pipeline/.env

# Restart service
sudo systemctl restart rag-pipeline
```

### Option 2: Docker Deployment

```bash
# On server with Docker installed
git clone <your-repo>
cd rag-pipeline-eqanun

# Configure environment
cp .env.example .env
nano .env

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Service Management (systemd)

```bash
# Status
sudo systemctl status rag-pipeline

# Restart
sudo systemctl restart rag-pipeline

# Enable auto-start
sudo systemctl enable rag-pipeline

# View logs
journalctl -u rag-pipeline -f
```

---

## 🔍 Monitoring & Debugging

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status
```

### Logs

**Application Logs:**
- Location: `logs/` directory
- Format: JSON structured logs
- Rotation: Daily (via logrotate)

**Retrieval Logs:**
- Location: `retrieval_logs/`
- Files: `YYYYMMDD_HHMMSS_<query>.json|.txt`
- Content: Query, retrieved docs, LLM response

**Docker Logs:**
```bash
docker-compose logs -f rag-pipeline
docker-compose logs -f nginx
```

---

## 🔧 Customization Points

### 1. **Embedding Provider**

Change in `.env`:
```bash
# Option 1: Vertex AI (default, supports dimension control)
EMBEDDING_PROVIDER=vertexai
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIM=1024

# Option 2: Google Generative AI (simpler, fixed dimensions)
EMBEDDING_PROVIDER=google-genai
EMBEDDING_MODEL=gemini-embedding-001
```

### 2. **Chunking Strategy**

Edit `hierarchical_chunker.py`:
```python
HierarchicalLegalChunker(
    parent_chunk_size=2000,  # Adjust for context size
    child_chunk_size=500,    # Adjust for precision
    overlap=100              # Adjust for continuity
)
```

### 3. **LLM Model**

Edit `app/main.py`:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-pro", "gemini-1.5-pro"
    temperature=0.7,
    max_output_tokens=2048
)
```

### 4. **Authentication Provider**

Switch in `.env`:
```bash
# Use local JWT
AUTH_PROVIDER=local

# Use Supabase
AUTH_PROVIDER=supabase
```

---

## 📈 Performance Considerations

### Embedding Generation
- **Batch Processing:** Texts are embedded in batches
- **Caching:** ChromaDB caches embeddings
- **Dimension Trade-off:** 1024 dims (accuracy) vs 768/512 (speed)

### Vector Search
- **HNSW Index:** ChromaDB uses efficient approximate search
- **Result Limit:** Default 10 results, configurable
- **Parent-Child Strategy:** Child search → parent retrieval (balanced)

### LLM Generation
- **Model:** `gemini-2.5-flash` (fast, cost-effective)
- **Context Window:** 32K tokens (sufficient for legal docs)
- **Streaming:** Not implemented (can be added)

---

## 🚨 Known Limitations

1. **User Storage:** Demo uses in-memory store (needs database)
2. **Token Blacklist:** Logout requires Redis (optional feature)
3. **Rate Limiting:** Not implemented (recommended for production)
4. **File Types:** Currently supports PDF and Markdown only
5. **Async Uploads:** Job tracking is in-memory (consider Redis/DB)
6. **Monitoring:** No built-in metrics/observability (add Prometheus/Grafana)

---

## 🔮 Future Enhancements

- [ ] PostgreSQL user database integration
- [ ] Redis-based session management
- [ ] Rate limiting middleware
- [ ] Prometheus metrics endpoint
- [ ] Streaming chat responses (SSE)
- [ ] Multi-language support
- [ ] Advanced admin dashboard
- [ ] Automated document preprocessing pipeline
- [ ] Citation accuracy scoring
- [ ] Query intent classification

---

## 📚 Additional Documentation

- **JWT Authentication:** See `JWT_AUTHENTICATION.md`
- **Supabase Integration:** See `SUPABASE_INTEGRATION_GUIDE.md`
- **Security Guide:** See `SECURITY.md`
- **API Documentation:** See `notebooks/API_DOCUMENTATION.md`
- **Frontend Integration:** See `notebooks/FRONTEND_INTEGRATION.md`

---

## 🤝 Integration Guide

### Next.js Frontend Integration

```typescript
// 1. Login
const response = await fetch('http://localhost:8000/auth/supabase/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email, password })
});
const { access_token } = await response.json();

// 2. Chat with documents
const chatResponse = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'What are the labor law requirements?',
    session_id: 'user-session-123'
  })
});
```

---

## 📝 License & Credits

**Built with:**
- FastAPI (web framework)
- LangChain (RAG orchestration)
- ChromaDB (vector storage)
- Google Gemini & Vertex AI (embeddings & LLM)
- Supabase (authentication - optional)

**Use Case:** Azerbaijani legal document Q&A system

---

**Last Updated:** October 2, 2025
