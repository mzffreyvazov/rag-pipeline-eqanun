# RAG Pipeline API Documentation

## Overview

The **Azerbaijani Legal RAG Pipeline API** is a FastAPI-based conversational AI system that provides question-answering capabilities over uploaded legal documents. It uses ChromaDB for vector storage, Google Gemini for language modeling, and LangGraph for conversation flow management.

**Base URL**: `http://localhost:8000` (development)

**Technology Stack**:
- FastAPI for REST API
- ChromaDB with all-MiniLM-L6-v2 embeddings
- Google Gemini 2.0 Flash for LLM
- LangGraph for conversation management
- PyPDFLoader for document processing

---

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

---

## Endpoints Reference

### 1. Health & Status Endpoints

#### `GET /` - Root Status
**Description**: Get basic API status and document count.

**Response Model**: `StatusResponse`
```json
{
  "status": "healthy",
  "collection_exists": true,
  "total_documents": 150,
  "message": "Azerbaijani Legal RAG Pipeline API is running"
}
```

**Example Request**:
```bash
curl -X GET "http://localhost:8000/"
```

---

#### `GET /health` - Detailed Health Check
**Description**: Get comprehensive system health information including component status.

**Response**:
```json
{
  "status": "healthy",
  "components": {
    "chromadb": "connected",
    "embedding_function": "initialized",
    "llm": "initialized",
    "graph": "initialized"
  },
  "documents_count": 150
}
```

**Possible Status Values**:
- `healthy`: All components working
- `degraded`: Some components not initialized

**Example Request**:
```bash
curl -X GET "http://localhost:8000/health"
```

---

#### `GET /status` - Operational Status
**Description**: Get detailed system operational status.

**Response Model**: `StatusResponse`
```json
{
  "status": "operational",
  "collection_exists": true,
  "total_documents": 150,
  "message": "System operational with 150 documents indexed"
}
```

---

### 2. Document Management

#### `POST /upload` - Upload PDF Documents
**Description**: Upload and process PDF documents for indexing in the vector database.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: One or more PDF files

**Response Model**: `UploadResponse`
```json
{
  "message": "Successfully processed 2 files",
  "files_processed": [
    "legal-document-1.pdf",
    "legal-document-2.pdf"
  ],
  "total_documents": 150
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

**Processing Details**:
- Supported formats: PDF only
- Text chunking: 1536 characters with 305 overlap
- Automatic metadata addition (filename, document type, upload ID)
- Batch processing in groups of 100 chunks

**Error Responses**:
- `400`: No files provided or non-PDF files
- `500`: Processing error

---

#### `DELETE /documents` - Clear All Documents
**Description**: Remove all documents from the vector store.

**Response**:
```json
{
  "message": "All documents cleared successfully",
  "status": "success"
}
```

**Example Request**:
```bash
curl -X DELETE "http://localhost:8000/documents"
```

**⚠️ Warning**: This action is irreversible and will delete all indexed documents.

---

### 3. Chat & Conversation

#### `POST /chat` - Chat with RAG System
**Description**: Send a message to the conversational RAG system and get an AI-powered response based on indexed documents.

**Request Model**: `ChatRequest`
```json
{
  "message": "What are the key provisions in the employment law?",
  "session_id": "optional-session-uuid"
}
```

**Response Model**: `ChatResponse`
```json
{
  "response": "Based on the indexed legal documents, the key provisions in employment law include...",
  "session_id": "session-uuid-123"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the definition of employment contract?",
    "session_id": "my-session-123"
  }'
```

**Conversation Features**:
- **Session Management**: Maintains conversation context using `session_id`
- **Smart Retrieval**: Automatically determines when to search documents vs. direct response
- **Source-Grounded**: Responses include relevant document excerpts
- **Azerbaijani Legal Focus**: Specialized for legal document analysis

**Session Behavior**:
- If `session_id` is provided: Continues existing conversation
- If `session_id` is omitted: Creates new session with auto-generated UUID
- Each session maintains separate conversation history

---

## Data Models

### StatusResponse
```typescript
interface StatusResponse {
  status: string;           // "healthy", "error", "operational"
  collection_exists: boolean;
  total_documents: number;
  message: string;
}
```

### ChatRequest
```typescript
interface ChatRequest {
  message: string;          // User's question/message
  session_id?: string;      // Optional session UUID
}
```

### ChatResponse
```typescript
interface ChatResponse {
  response: string;         // AI-generated response
  session_id: string;       // Session UUID (auto-generated if not provided)
}
```

### UploadResponse
```typescript
interface UploadResponse {
  message: string;          // Success message
  files_processed: string[]; // List of processed filenames
  total_documents: number;   // Total documents in database
}
```

### DocumentInfo
```typescript
interface DocumentInfo {
  filename: string;         // Original filename
  total_pages: number;      // Number of pages in PDF
  chunks_created: number;   // Number of text chunks created
}
```

---

## Integration Examples

### React/Next.js Integration

#### 1. Upload Documents Component
```typescript
import { useState } from 'react';

interface UploadComponentProps {}

const DocumentUpload: React.FC<UploadComponentProps> = () => {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleUpload = async (files: FileList) => {
    setUploading(true);
    const formData = new FormData();
    
    Array.from(files).forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      
      const result = await response.json();
      setResult(result);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <input 
        type="file" 
        multiple 
        accept=".pdf"
        onChange={(e) => e.target.files && handleUpload(e.target.files)}
        disabled={uploading}
      />
      {uploading && <p>Uploading...</p>}
      {result && (
        <div>
          <p>{result.message}</p>
          <p>Total documents: {result.total_documents}</p>
        </div>
      )}
    </div>
  );
};
```

#### 2. Chat Component
```typescript
import { useState, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const ChatComponent: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          session_id: sessionId || undefined,
        }),
      });

      const result = await response.json();
      
      // Update session ID if new
      if (!sessionId) {
        setSessionId(result.session_id);
      }

      const assistantMessage: Message = { 
        role: 'assistant', 
        content: result.response 
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <strong>{msg.role}:</strong> {msg.content}
          </div>
        ))}
        {loading && <div className="loading">AI is thinking...</div>}
      </div>
      
      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask about legal documents..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
};
```

#### 3. System Status Hook
```typescript
import { useState, useEffect } from 'react';

interface SystemStatus {
  status: string;
  collection_exists: boolean;
  total_documents: number;
  message: string;
}

export const useSystemStatus = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/health');
      
      if (!response.ok) {
        throw new Error('Failed to fetch status');
      }
      
      const data = await response.json();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  return { status, loading, error, refetch: fetchStatus };
};
```

---

## Error Handling

### Common HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid input, wrong file types)
- `404`: Not Found (invalid endpoint)
- `500`: Internal Server Error (system errors, component failures)

### Error Response Format
```json
{
  "detail": "Error description here"
}
```

### Typical Error Scenarios

1. **Upload Errors**:
   - Non-PDF files: `400` - "Only PDF files are supported"
   - No files: `400` - "No files provided"
   - Processing failure: `500` - "Error processing files: [details]"

2. **Chat Errors**:
   - System not ready: `500` - "RAG system not initialized"
   - Processing error: `500` - "Error in chat: [details]"

3. **Document Management Errors**:
   - Database connection: `500` - "Error accessing collection: [details]"
   - Clear operation: `500` - "Error clearing documents: [details]"

---

## Best Practices for Frontend Integration

### 1. Connection Management
```typescript
// Check system health before using other endpoints
const checkSystemHealth = async () => {
  const response = await fetch('/health');
  const health = await response.json();
  return health.status === 'healthy';
};
```

### 2. Session Management
```typescript
// Persist session ID in localStorage or session storage
const getSessionId = () => {
  return localStorage.getItem('chat_session_id') || null;
};

const setSessionId = (sessionId: string) => {
  localStorage.setItem('chat_session_id', sessionId);
};
```

### 3. File Upload Progress
```typescript
// Use FormData and track upload progress
const uploadWithProgress = async (files: FileList, onProgress: (percent: number) => void) => {
  const formData = new FormData();
  Array.from(files).forEach(file => formData.append('files', file));

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const percent = (e.loaded / e.total) * 100;
        onProgress(percent);
      }
    });

    xhr.onload = () => {
      if (xhr.status === 200) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Upload failed: ${xhr.statusText}`));
      }
    };

    xhr.open('POST', 'http://localhost:8000/upload');
    xhr.send(formData);
  });
};
```

### 4. Real-time Status Updates
```typescript
// Poll system status for real-time updates
const useRealTimeStatus = (interval = 5000) => {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await fetch('/status');
        const data = await response.json();
        setStatus(data);
      } catch (error) {
        console.error('Status poll failed:', error);
      }
    };

    const intervalId = setInterval(pollStatus, interval);
    pollStatus(); // Initial call

    return () => clearInterval(intervalId);
  }, [interval]);

  return status;
};
```

---

## OpenAPI Documentation

The API automatically generates interactive documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

These provide interactive interfaces for testing endpoints directly in the browser.

---

## Development Notes

### Environment Setup
The API requires these environment variables:
```env
PROJECT_ID=your-google-cloud-project
REGION=your-preferred-region
GOOGLE_API_KEY=your-google-api-key
```

### CORS Configuration
The API is configured with permissive CORS for development:
```python
allow_origins=["*"]  # Configure for production
```

For production, update CORS settings to restrict origins to your frontend domain.

### Data Persistence
- **ChromaDB**: Data stored in `./chroma_data` directory
- **Session Memory**: Conversation history maintained in memory (resets on restart)
- **File Processing**: Temporary files cleaned up automatically

---

This documentation provides everything needed to integrate with the RAG Pipeline API in your React/Next.js application. The API is designed to be simple, reliable, and easy to integrate with modern frontend frameworks.
