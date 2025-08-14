# Frontend Integration Guide

## Quick Start Guide for React/Next.js

This guide provides ready-to-use code examples for integrating the RAG Pipeline API with your React or Next.js application.

## Table of Contents
1. [API Client Setup](#api-client-setup)
2. [TypeScript Interfaces](#typescript-interfaces)
3. [React Hooks](#react-hooks)
4. [Complete Components](#complete-components)
5. [Next.js Integration](#nextjs-integration)
6. [Error Handling](#error-handling)
7. [Styling Examples](#styling-examples)

---

## API Client Setup

### Basic API Client
```typescript
// lib/api.ts
const API_BASE_URL = 'http://localhost:8000';

export class RAGApiClient {
  private baseURL: string;

  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async uploadFiles(files: FileList): Promise<UploadResponse> {
    const formData = new FormData();
    Array.from(files).forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${this.baseURL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  async chat(message: string, sessionId?: string): Promise<ChatResponse> {
    return this.request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify({
        message,
        session_id: sessionId,
      }),
    });
  }

  async getStatus(): Promise<StatusResponse> {
    return this.request<StatusResponse>('/status');
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async clearDocuments(): Promise<{ message: string; status: string }> {
    return this.request('/documents', {
      method: 'DELETE',
    });
  }
}

export const apiClient = new RAGApiClient();
```

---

## TypeScript Interfaces

```typescript
// types/api.ts
export interface StatusResponse {
  status: string;
  collection_exists: boolean;
  total_documents: number;
  message: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
}

export interface UploadResponse {
  message: string;
  files_processed: string[];
  total_documents: number;
}

export interface DocumentInfo {
  filename: string;
  total_pages: number;
  chunks_created: number;
}

export interface HealthResponse {
  status: string;
  components: {
    chromadb: string;
    embedding_function: string;
    llm: string;
    graph: string;
  };
  documents_count: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface ChatSession {
  id: string;
  messages: Message[];
  created_at: Date;
  updated_at: Date;
}
```

---

## React Hooks

### useRAGApi Hook
```typescript
// hooks/useRAGApi.ts
import { useState, useCallback } from 'react';
import { apiClient } from '../lib/api';
import type { StatusResponse, ChatResponse, UploadResponse, HealthResponse } from '../types/api';

export const useRAGApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeAsync = useCallback(async <T>(
    operation: () => Promise<T>
  ): Promise<T | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await operation();
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const uploadFiles = useCallback(async (files: FileList): Promise<UploadResponse | null> => {
    return executeAsync(() => apiClient.uploadFiles(files));
  }, [executeAsync]);

  const sendMessage = useCallback(async (message: string, sessionId?: string): Promise<ChatResponse | null> => {
    return executeAsync(() => apiClient.chat(message, sessionId));
  }, [executeAsync]);

  const getStatus = useCallback(async (): Promise<StatusResponse | null> => {
    return executeAsync(() => apiClient.getStatus());
  }, [executeAsync]);

  const getHealth = useCallback(async (): Promise<HealthResponse | null> => {
    return executeAsync(() => apiClient.getHealth());
  }, [executeAsync]);

  const clearDocuments = useCallback(async () => {
    return executeAsync(() => apiClient.clearDocuments());
  }, [executeAsync]);

  return {
    loading,
    error,
    uploadFiles,
    sendMessage,
    getStatus,
    getHealth,
    clearDocuments,
  };
};
```

### useSystemStatus Hook
```typescript
// hooks/useSystemStatus.ts
import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../lib/api';
import type { HealthResponse } from '../types/api';

export const useSystemStatus = (pollInterval = 30000) => {
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const health = await apiClient.getHealth();
      setStatus(health);
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch status';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    
    const interval = setInterval(fetchStatus, pollInterval);
    return () => clearInterval(interval);
  }, [fetchStatus, pollInterval]);

  const isHealthy = status?.status === 'healthy';
  const documentCount = status?.documents_count || 0;

  return {
    status,
    loading,
    error,
    isHealthy,
    documentCount,
    refetch: fetchStatus,
  };
};
```

### useChat Hook
```typescript
// hooks/useChat.ts
import { useState, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useRAGApi } from './useRAGApi';
import type { Message, ChatSession } from '../types/api';

export const useChat = (sessionId?: string) => {
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const { sendMessage: apiSendMessage, loading } = useRAGApi();

  // Initialize session
  useEffect(() => {
    if (sessionId) {
      // Load existing session or create new one
      const existingSession = sessions.find(s => s.id === sessionId);
      if (existingSession) {
        setCurrentSession(existingSession);
      } else {
        createNewSession(sessionId);
      }
    } else {
      createNewSession();
    }
  }, [sessionId]);

  const createNewSession = useCallback((id?: string) => {
    const newSession: ChatSession = {
      id: id || uuidv4(),
      messages: [],
      created_at: new Date(),
      updated_at: new Date(),
    };
    
    setCurrentSession(newSession);
    setSessions(prev => [newSession, ...prev]);
    
    // Persist to localStorage
    localStorage.setItem('rag_current_session', newSession.id);
    localStorage.setItem('rag_sessions', JSON.stringify([newSession, ...sessions]));
  }, [sessions]);

  const sendMessage = useCallback(async (content: string) => {
    if (!currentSession || !content.trim()) return;

    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    };

    // Add user message immediately
    const updatedSession = {
      ...currentSession,
      messages: [...currentSession.messages, userMessage],
      updated_at: new Date(),
    };
    
    setCurrentSession(updatedSession);

    try {
      // Send to API
      const response = await apiSendMessage(content, currentSession.id);
      
      if (response) {
        const assistantMessage: Message = {
          id: uuidv4(),
          role: 'assistant',
          content: response.response,
          timestamp: new Date(),
        };

        const finalSession = {
          ...updatedSession,
          messages: [...updatedSession.messages, assistantMessage],
          updated_at: new Date(),
        };

        setCurrentSession(finalSession);
        
        // Update sessions list
        setSessions(prev => 
          prev.map(s => s.id === finalSession.id ? finalSession : s)
        );

        // Persist to localStorage
        localStorage.setItem('rag_sessions', 
          JSON.stringify(sessions.map(s => s.id === finalSession.id ? finalSession : s))
        );
      }
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  }, [currentSession, apiSendMessage, sessions]);

  const clearCurrentSession = useCallback(() => {
    if (currentSession) {
      const clearedSession = {
        ...currentSession,
        messages: [],
        updated_at: new Date(),
      };
      setCurrentSession(clearedSession);
    }
  }, [currentSession]);

  return {
    currentSession,
    sessions,
    sendMessage,
    loading,
    createNewSession,
    clearCurrentSession,
  };
};
```

---

## Complete Components

### Document Upload Component
```tsx
// components/DocumentUpload.tsx
import React, { useState, useRef } from 'react';
import { useRAGApi } from '../hooks/useRAGApi';

interface DocumentUploadProps {
  onUploadComplete?: (result: any) => void;
  className?: string;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUploadComplete,
  className = '',
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { uploadFiles, loading, error } = useRAGApi();

  const handleFiles = async (files: FileList) => {
    if (files.length === 0) return;

    // Validate PDF files
    const pdfFiles = Array.from(files).filter(file => 
      file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
    );

    if (pdfFiles.length === 0) {
      alert('Please select PDF files only.');
      return;
    }

    const fileList = new DataTransfer();
    pdfFiles.forEach(file => fileList.items.add(file));

    const result = await uploadFiles(fileList.files);
    if (result && onUploadComplete) {
      onUploadComplete(result);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className={`document-upload ${className}`}>
      <div
        className={`upload-area ${dragActive ? 'drag-active' : ''} ${loading ? 'uploading' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={openFileDialog}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf"
          onChange={handleChange}
          style={{ display: 'none' }}
          disabled={loading}
        />
        
        <div className="upload-content">
          {loading ? (
            <div className="upload-loading">
              <div className="spinner" />
              <p>Processing documents...</p>
            </div>
          ) : (
            <>
              <div className="upload-icon">📄</div>
              <h3>Upload Legal Documents</h3>
              <p>
                Drag and drop PDF files here, or{' '}
                <span className="upload-link">click to browse</span>
              </p>
              <p className="upload-note">
                Only PDF files are supported
              </p>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="error-message">
          <p>❌ Upload failed: {error}</p>
        </div>
      )}
    </div>
  );
};
```

### Chat Interface Component
```tsx
// components/ChatInterface.tsx
import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import type { Message } from '../types/api';

interface ChatInterfaceProps {
  className?: string;
  sessionId?: string;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  className = '',
  sessionId,
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { currentSession, sendMessage, loading } = useChat(sessionId);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentSession?.messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !loading) {
      await sendMessage(input);
      setInput('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const formatMessage = (content: string) => {
    // Simple formatting for better readability
    return content
      .split('\n')
      .map((line, index) => (
        <p key={index} className={line.trim() === '' ? 'empty-line' : ''}>
          {line}
        </p>
      ));
  };

  return (
    <div className={`chat-interface ${className}`}>
      <div className="chat-header">
        <h3>Legal Document Assistant</h3>
        {currentSession && (
          <p className="session-info">
            Session: {currentSession.id.slice(0, 8)}...
          </p>
        )}
      </div>

      <div className="messages-container">
        {currentSession?.messages.length === 0 ? (
          <div className="welcome-message">
            <div className="welcome-icon">⚖️</div>
            <h4>Welcome to the Legal Document Assistant</h4>
            <p>
              I can help you find information in your uploaded legal documents.
              Ask me questions about laws, regulations, or specific provisions.
            </p>
            <div className="example-questions">
              <h5>Example questions:</h5>
              <ul>
                <li>"What are the requirements for employment contracts?"</li>
                <li>"Explain the termination procedures"</li>
                <li>"What are the employee rights?"</li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="messages">
            {currentSession?.messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.role}`}
              >
                <div className="message-header">
                  <span className="message-role">
                    {message.role === 'user' ? '👤 You' : '🤖 Assistant'}
                  </span>
                  <span className="message-time">
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                </div>
                <div className="message-content">
                  {formatMessage(message.content)}
                </div>
              </div>
            ))}
          </div>
        )}

        {loading && (
          <div className="message assistant loading">
            <div className="message-header">
              <span className="message-role">🤖 Assistant</span>
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input-form">
        <div className="input-container">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about legal documents..."
            disabled={loading}
            rows={1}
            className="chat-input"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="send-button"
          >
            {loading ? '⏳' : '➤'}
          </button>
        </div>
      </form>
    </div>
  );
};
```

### System Status Component
```tsx
// components/SystemStatus.tsx
import React from 'react';
import { useSystemStatus } from '../hooks/useSystemStatus';

interface SystemStatusProps {
  className?: string;
  showDetails?: boolean;
}

export const SystemStatus: React.FC<SystemStatusProps> = ({
  className = '',
  showDetails = false,
}) => {
  const { status, loading, error, isHealthy, documentCount, refetch } = useSystemStatus();

  if (loading) {
    return (
      <div className={`system-status loading ${className}`}>
        <div className="status-indicator">🔄</div>
        <span>Checking system status...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`system-status error ${className}`}>
        <div className="status-indicator">❌</div>
        <span>System unavailable</span>
        <button onClick={refetch} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={`system-status ${isHealthy ? 'healthy' : 'degraded'} ${className}`}>
      <div className="status-indicator">
        {isHealthy ? '✅' : '⚠️'}
      </div>
      
      <div className="status-info">
        <div className="status-main">
          <span className="status-text">
            {isHealthy ? 'System Operational' : 'System Degraded'}
          </span>
          <span className="document-count">
            {documentCount} documents indexed
          </span>
        </div>

        {showDetails && status && (
          <div className="status-details">
            <div className="component-status">
              <span className={`component ${status.components.chromadb === 'connected' ? 'ok' : 'error'}`}>
                Database: {status.components.chromadb}
              </span>
              <span className={`component ${status.components.llm === 'initialized' ? 'ok' : 'error'}`}>
                AI Model: {status.components.llm}
              </span>
              <span className={`component ${status.components.graph === 'initialized' ? 'ok' : 'error'}`}>
                Chat System: {status.components.graph}
              </span>
            </div>
          </div>
        )}
      </div>

      <button onClick={refetch} className="refresh-button" title="Refresh status">
        🔄
      </button>
    </div>
  );
};
```

---

## Next.js Integration

### API Routes (App Router)
```typescript
// app/api/proxy/[...slug]/route.ts
import { NextRequest, NextResponse } from 'next/server';

const RAG_API_BASE = 'http://localhost:8000';

export async function GET(
  request: NextRequest,
  { params }: { params: { slug: string[] } }
) {
  const endpoint = params.slug.join('/');
  const url = new URL(request.url);
  const queryString = url.search;

  try {
    const response = await fetch(`${RAG_API_BASE}/${endpoint}${queryString}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to proxy request' },
      { status: 500 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { slug: string[] } }
) {
  const endpoint = params.slug.join('/');
  const body = await request.text();

  try {
    const response = await fetch(`${RAG_API_BASE}/${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': request.headers.get('content-type') || 'application/json',
      },
      body,
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to proxy request' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { slug: string[] } }
) {
  const endpoint = params.slug.join('/');

  try {
    const response = await fetch(`${RAG_API_BASE}/${endpoint}`, {
      method: 'DELETE',
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to proxy request' },
      { status: 500 }
    );
  }
}
```

### Main Page Component
```tsx
// app/page.tsx
'use client';

import React, { useState } from 'react';
import { DocumentUpload } from '../components/DocumentUpload';
import { ChatInterface } from '../components/ChatInterface';
import { SystemStatus } from '../components/SystemStatus';

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'chat' | 'upload' | 'status'>('chat');

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Legal Document Assistant
              </h1>
              <p className="text-gray-600">
                AI-powered legal document analysis and Q&A
              </p>
            </div>
            <SystemStatus />
          </div>
        </div>
      </header>

      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'chat', label: 'Chat', icon: '💬' },
              { id: 'upload', label: 'Upload Documents', icon: '📄' },
              { id: 'status', label: 'System Status', icon: '⚡' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'chat' && (
          <div className="bg-white rounded-lg shadow h-[600px]">
            <ChatInterface className="h-full" />
          </div>
        )}

        {activeTab === 'upload' && (
          <div className="space-y-6">
            <DocumentUpload
              onUploadComplete={(result) => {
                console.log('Upload completed:', result);
                // Optionally switch to chat tab
                setActiveTab('chat');
              }}
            />
          </div>
        )}

        {activeTab === 'status' && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">System Status</h2>
            <SystemStatus showDetails={true} />
          </div>
        )}
      </main>
    </div>
  );
}
```

---

## Styling Examples

### Tailwind CSS Classes
```css
/* components/chat.module.css */
.chatInterface {
  @apply flex flex-col h-full;
}

.chatHeader {
  @apply border-b p-4 bg-gray-50;
}

.messagesContainer {
  @apply flex-1 overflow-y-auto p-4 space-y-4;
}

.message {
  @apply max-w-3xl;
}

.message.user {
  @apply ml-auto;
}

.message.assistant {
  @apply mr-auto;
}

.messageHeader {
  @apply flex justify-between items-center mb-2 text-sm text-gray-600;
}

.messageContent {
  @apply bg-white rounded-lg p-4 shadow-sm border;
}

.message.user .messageContent {
  @apply bg-blue-500 text-white;
}

.chatInputForm {
  @apply border-t p-4 bg-gray-50;
}

.inputContainer {
  @apply flex space-x-2;
}

.chatInput {
  @apply flex-1 border rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none;
}

.sendButton {
  @apply bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed;
}

/* Upload component styles */
.uploadArea {
  @apply border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer transition-colors hover:border-gray-400;
}

.uploadArea.dragActive {
  @apply border-blue-500 bg-blue-50;
}

.uploadArea.uploading {
  @apply pointer-events-none opacity-75;
}

.uploadContent {
  @apply space-y-4;
}

.uploadIcon {
  @apply text-6xl;
}

.uploadLink {
  @apply text-blue-500 hover:underline;
}

.uploadNote {
  @apply text-sm text-gray-500;
}

/* Status component styles */
.systemStatus {
  @apply flex items-center space-x-2 px-3 py-2 rounded-lg;
}

.systemStatus.healthy {
  @apply bg-green-100 text-green-800;
}

.systemStatus.degraded {
  @apply bg-yellow-100 text-yellow-800;
}

.systemStatus.error {
  @apply bg-red-100 text-red-800;
}

.statusIndicator {
  @apply text-lg;
}

.componentStatus {
  @apply flex space-x-4 text-xs;
}

.component.ok {
  @apply text-green-600;
}

.component.error {
  @apply text-red-600;
}
```

This comprehensive integration guide provides everything you need to build a modern React/Next.js frontend for your RAG Pipeline API. The components are modular, typed with TypeScript, and ready for production use with proper error handling and loading states.
