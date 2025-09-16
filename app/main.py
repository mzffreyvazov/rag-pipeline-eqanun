import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading
from contextlib import asynccontextmanager
import shutil
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import vertexai
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import hierarchical chunking components
from hierarchical_chunker import HierarchicalLegalChunker, HierarchicalRetriever

# Load environment variables
load_dotenv()

# Initialize Vertex AI
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Embedding configuration (can be overridden via environment)
# Per request: use Vertex AI with gemini-embedding-001 at 1024 dims by default
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "vertexai").strip().lower()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
try:
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
except ValueError:
    EMBEDDING_DIM = 1024

if PROJECT_ID and REGION:
    vertexai.init(project=PROJECT_ID, location=REGION)

# Global variables
chroma_client = None
embedding_function = None
llm = None
vector_store = None
graph = None
collection_name = "legal_documents"

# In-memory job tracking for upload progress (consider Redis for production)
jobs_lock = threading.Lock()
jobs_store: Dict[str, Dict[str, Any]] = {}

def _build_embedding_function():
    """Build and return an embedding function compatible with Chroma.

    Supports providers:
      - google-genai (default): uses Chroma's GoogleGenerativeAiEmbeddingFunction
      - vertexai: wraps VertexAIEmbeddings from langchain
    """
    provider = EMBEDDING_PROVIDER
    model = EMBEDDING_MODEL

    if provider == "google-genai":
        # Default to Google's latest text-embedding model if not provided
        model_name = model or "gemini-embedding-001"
        ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=GOOGLE_API_KEY,
            model_name=model_name,
        )
        return ef, provider, model_name, None

    elif provider == "vertexai":
        # Use Vertex AI embeddings via Vertex SDK to control dimensionality
        v_model = model or "gemini-embedding-001"

        class _VertexEmbeddingWrapper:
            def __init__(self, model_name: str):
                # Lazy import to avoid heavy deps unless needed
                # Prefer direct Vertex AI SDK to control output dimensionality
                from vertexai.language_models import TextEmbeddingModel
                self._model = TextEmbeddingModel.from_pretrained(model_name)
                self._dim = EMBEDDING_DIM
                # Provide a public name attribute for Chroma compatibility
                self.name = f"vertexai::{model_name}::{self._dim}d"

            def __call__(self, input):
                # Chroma calls the embedding function with a list[str] named 'input'
                # Ensure list input
                if isinstance(input, str):
                    texts = [input]
                else:
                    texts = list(input)

                # Call Vertex AI with desired dimensionality
                embeddings = self._model.get_embeddings(texts, output_dimensionality=self._dim)
                # Each item has `.values`
                return [e.values for e in embeddings]

        return _VertexEmbeddingWrapper(v_model), provider, v_model, EMBEDDING_DIM

    else:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER '{provider}'. Use 'google-genai' or 'vertexai'."
        )

def initialize_components():
    """Initialize all AI components"""
    global chroma_client, embedding_function, llm, vector_store
    
    try:
        # Initialize ChromaDB with persistent storage
        chroma_client = chromadb.PersistentClient(
            path="./chroma_data",
            settings=Settings(allow_reset=True)
        )
        # Build embedding function per configuration
        embedding_function, provider, model_name, model_dim = _build_embedding_function()

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        print(
            f"✅ Components initialized | Embeddings: provider='{provider}', model='{model_name}', dim='{model_dim or 'n/a'}'"
        )
        return True
    
    except Exception as e:
        print(f"❌ Error initializing components: {str(e)}")
        return False

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Helper to embed a list of texts using the configured embedding function.

    Returns a list of vector lists (one per input text).
    """
    global embedding_function
    if embedding_function is None:
        raise RuntimeError("Embedding function is not initialized")
    # Ensure list of strings
    inputs = [str(t) for t in texts]
    vectors = embedding_function(inputs)
    # Some EF implementations may return numpy arrays; coerce to plain lists
    return [list(v) for v in vectors]

# Global variable to store last retrieved documents for source display
last_retrieved_docs: List[Dict[str, Any]] = []

def create_rag_graph():
    """Create the RAG conversation graph"""
    global llm
    
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query using hierarchical retrieval."""
        global last_retrieved_docs
        try:
            collection = get_existing_collection()
            
            # Initialize hierarchical retriever with embed function
            retriever = HierarchicalRetriever(collection, embed_texts)
            
            # Use adaptive hierarchical retrieval
            retrieved_docs = retriever.retrieve_with_hierarchy(query, n_results=10)
            
            # Store retrieved docs globally for source display
            last_retrieved_docs = retrieved_docs.copy()
            
            # Debug: Print what documents are being retrieved
            print(f"\n================ HIERARCHICAL RETRIEVAL DEBUG ================")
            print(f"🔎 Query: {query}")
            print(f"🔢 Retrieved: {len(retrieved_docs)} results")

            # Try to print embedding config if available
            try:
                provider = os.environ.get("EMBEDDING_PROVIDER", "vertexai")
                model = os.environ.get("EMBEDDING_MODEL") or "gemini-embedding-001"
                dim = int(os.environ.get("EMBEDDING_DIM", "1024"))
                print(f"🔧 Embedding config -> provider='{provider}', model='{model}', dim='{dim}'")
            except Exception:
                pass

            # Print retrieved results with hierarchy information
            for rank, doc in enumerate(retrieved_docs, start=1):
                metadata = doc.get('metadata', {})
                content = doc.get('content', '')
                source = metadata.get('source_document', 'Unknown')
                chunk_type = metadata.get('chunk_type', 'unknown')
                hierarchy_level = metadata.get('hierarchy_level', 'unknown')
                preview = content[:200].replace('\n', ' ')
                
                print(f"{rank:02d}. type={chunk_type} | level={hierarchy_level} | source={source}")
                print(f"    {preview}...")
                
                # Show parent context if available
                if metadata.get('parent_id'):
                    print(f"    ↳ Parent: {metadata.get('parent_id')}")
            
            if retrieved_docs:
                sources = set(doc['metadata'].get('source_document', 'Unknown') for doc in retrieved_docs)
                print(f"📄 Sources found: {', '.join(sources)}")
                # Print per-source counts
                counts = {}
                chunk_type_counts = {}
                for doc in retrieved_docs:
                    meta = doc.get('metadata', {})
                    s = meta.get('source_document', 'Unknown')
                    chunk_type = meta.get('chunk_type', 'unknown')
                    counts[s] = counts.get(s, 0) + 1
                    chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
                
                print("📊 Per-source counts:")
                for s, c in counts.items():
                    print(f"   - {s}: {c}")
                    
                print("🏗️ Chunk type distribution:")
                for chunk_type, c in chunk_type_counts.items():
                    print(f"   - {chunk_type}: {c}")
            else:
                print("❌ No documents retrieved")
            print("================ END HIERARCHICAL RETRIEVAL DEBUG ================\n")
            
            serialized = "\n\n".join(
                (f"[Source: {doc['metadata'].get('source_document', 'Unknown')}]\n{doc['content']}")
                for doc in retrieved_docs
            )
            
            # Save retrieved content to file for analysis
            save_retrieved_content_to_file(query, retrieved_docs, serialized)
            
            return serialized, retrieved_docs
        except Exception as e:
            return f"Error retrieving documents: {str(e)}", []

    def query_or_respond(state: MessagesState):
        """Always retrieve documents for every query."""
        # Get the user's last message
        last_message = state["messages"][-1]
        user_query = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Always call retrieve tool for every query
        from langchain_core.messages import ToolMessage, AIMessage
        
        # Create a tool call message
        tool_call_id = str(uuid.uuid4())
        ai_message = AIMessage(
            content="I'll search the legal documents to answer your question.",
            tool_calls=[{
                "name": "retrieve",
                "args": {"query": user_query},
                "id": tool_call_id
            }]
        )
        
        return {"messages": [ai_message]}

    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        
        # Debug: Print what content is being sent to LLM
        print(f"\n🤖 CONTENT BEING SENT TO LLM:")
        print(f"Content length: {len(docs_content)} characters")
        if docs_content:
            print(f"Content preview: {docs_content[:500]}...")
        else:
            print("❌ NO CONTENT RETRIEVED!")
        
        system_message_content = (
            "You are an assistant for Azerbaijani legal question-answering tasks. "
            "Use the following pieces of retrieved context from legal documents to answer "
            "the question. The context may come from multiple legal documents. "
            "If you find relevant information, cite the source document name. "
            "If you don't know the answer or can't find relevant information in the provided context, "
            "say that you don't have sufficient information in the uploaded documents. "
            "Provide a comprehensive but concise answer."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}

    # Create the graph
    graph_builder = StateGraph(MessagesState)
    tools = ToolNode([retrieve])

    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", generate)

    graph_builder.set_entry_point("query_or_respond")

    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
    )

    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Declare global variables at the beginning
    global chroma_client, embedding_function, llm, graph
    
    # Startup
    print("🚀 Starting up RAG Pipeline...")
    
    if not initialize_components():
        raise RuntimeError("Failed to initialize AI components")
    
    # Initialize the graph
    graph = create_rag_graph()
    print("🚀 RAG Pipeline API is ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down RAG Pipeline...")
    # Clean up resources if needed
    chroma_client = None
    embedding_function = None
    llm = None
    graph = None
    print("✅ Cleanup completed")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Azerbaijani Legal RAG Pipeline",
    description="A conversational RAG system for Azerbaijani legal documents (PDF and Markdown)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (moved after app initialization)
collection_name = "legal_documents"

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class SourceInfo(BaseModel):
    document_name: str
    retrieved_content: str
    page_number: Optional[int] = None
    chunk_type: Optional[str] = None
    hierarchy_level: Optional[int] = None
    clause_number: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: Optional[List[SourceInfo]] = None

class UploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_documents: int

class DocumentInfo(BaseModel):
    filename: str
    total_pages: int
    chunks_created: int
    parent_chunks: Optional[int] = None
    child_chunks: Optional[int] = None

class StatusResponse(BaseModel):
    status: str
    collection_exists: bool
    total_documents: int
    message: str

# Upload job progress models
class FileProgress(BaseModel):
    filename: str
    pages_total: Optional[int] = None
    chunks_total: Optional[int] = None
    chunks_done: int = 0
    percent: float = 0.0
    status: str = "pending"  # pending|running|completed|failed
    error: Optional[str] = None

class UploadStartResponse(BaseModel):
    job_id: str
    files: List[str]

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # pending|running|completed|failed
    files: Dict[str, FileProgress]
    overall: Dict[str, Any]
    error: Optional[str] = None

# Document retrieval models
class RetrieveRequest(BaseModel):
    query: str
    n_results: Optional[int] = 10

class RetrievedChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None

class RetrieveResponse(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    total_results: int

def get_or_create_collection():
    """Get or create the ChromaDB collection with Gemini embeddings"""
    global chroma_client, collection_name, embedding_function

    try:
        # Try to get existing collection
        provider = EMBEDDING_PROVIDER
        # Reflect the currently configured model (fall back to defaults used in _build_embedding_function)
        model = EMBEDDING_MODEL or "gemini-embedding-001"
        dim = EMBEDDING_DIM

        try:
            # Fetch existing collection without binding embedding function
            existing = chroma_client.get_collection(
                name=collection_name,
            )
            meta = existing.metadata or {}
            cur_provider = meta.get("embedding_provider")
            cur_model = meta.get("embedding_model")
            cur_dim = meta.get("embedding_dim")

            if (
                cur_provider == provider
                and (not cur_model or cur_model == model)
                and (not cur_dim or int(cur_dim) == int(dim))
            ):
                print(
                    f"✅ Using existing collection '{collection_name}' | provider='{cur_provider}', model='{cur_model or 'unknown'}', dim='{cur_dim or 'unknown'}'"
                )
                return existing
            else:
                print(
                    f"🔄 Recreating collection '{collection_name}' to align embeddings |"
                    f" current(provider='{cur_provider}', model='{cur_model}', dim='{cur_dim}'),"
                    f" desired(provider='{provider}', model='{model}', dim='{dim}')"
                )
                chroma_client.delete_collection(name=collection_name)
        except Exception:
            # Not found or unable to get — proceed to create
            pass

        # Create new collection with embedding metadata
        try:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Azerbaijani legal documents collection with embeddings",
                    "embedding_provider": provider,
                    "embedding_model": model,
                    "embedding_dim": dim,
                },
            )
        except Exception as ce:
            # If it already exists (race or concurrent call), return the existing collection
            if "already exists" in str(ce).lower():
                return chroma_client.get_collection(name=collection_name)
            raise
        print(
            f"✅ Created collection '{collection_name}' | provider='{provider}', model='{model}', dim='{dim}'"
        )
        return collection

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing collection: {str(e)}")

def get_existing_collection():
    """Get existing collection for queries (assumes it already exists with correct embedding function)"""
    global chroma_client, collection_name, embedding_function
    
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
        )
        return collection
    except Exception as e:
        # If collection doesn't exist or has wrong embedding function, create it
        return get_or_create_collection()

def save_retrieved_content_to_file(query: str, retrieved_docs: List[Dict[str, Any]], serialized_content: str):
    """
    Save retrieved content to a text file for analysis.
    Creates a timestamped file with query, retrieved chunks, and final serialized content.
    """
    try:
        # Create retrieval_logs directory if it doesn't exist
        logs_dir = Path("retrieval_logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Generate timestamp and safe filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in query[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_query = safe_query.replace(' ', '_')
        filename = f"{timestamp}_{safe_query}.txt"
        
        filepath = logs_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HIERARCHICAL RETRIEVAL ANALYSIS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Retrieved chunks: {len(retrieved_docs)}\n")
            f.write("=" * 80 + "\n\n")
            
            # Individual chunk analysis
            f.write("INDIVIDUAL RETRIEVED CHUNKS:\n")
            f.write("-" * 50 + "\n\n")
            
            for i, doc in enumerate(retrieved_docs, 1):
                metadata = doc.get('metadata', {})
                content = doc.get('content', '')
                
                f.write(f"CHUNK #{i}:\n")
                f.write(f"  Chunk Type: {metadata.get('chunk_type', 'unknown')}\n")
                f.write(f"  Source Document: {metadata.get('source_document', 'unknown')}\n")
                f.write(f"  Chunk ID: {metadata.get('chunk_id', 'unknown')}\n")
                f.write(f"  Parent ID: {metadata.get('parent_chunk_id', 'none')}\n")
                f.write(f"  Hierarchical Path: {metadata.get('hierarchical_path', 'unknown')}\n")
                f.write(f"  Content Length: {metadata.get('content_length', len(content))}\n")
                f.write(f"  Distance: {doc.get('distance', 'unknown')}\n")
                
                f.write(f"\n  CONTENT:\n")
                f.write(f"  {'-' * 40}\n")
                f.write(f"  {content}\n")
                f.write(f"  {'-' * 40}\n\n")
            
            # Final serialized content sent to LLM
            f.write("\n" + "=" * 80 + "\n")
            f.write("FINAL SERIALIZED CONTENT SENT TO LLM:\n")
            f.write("=" * 80 + "\n")
            f.write(serialized_content)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("END OF LOG\n")
            f.write("=" * 80 + "\n")
        
        print(f"📝 Retrieved content saved to: {filepath}")
        
    except Exception as e:
        print(f"⚠️ Error saving retrieved content: {str(e)}")

def sanitize_metadata_for_chromadb(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata to ensure ChromaDB compatibility.
    ChromaDB only accepts: str, int, float, bool (no None values).
    """
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            # Skip None values completely
            continue
        elif isinstance(value, str):
            # Keep non-empty strings, convert empty strings to a default
            sanitized[key] = value if value.strip() else "unknown"
        elif isinstance(value, (int, float, bool)):
            # Keep valid numeric and boolean values
            sanitized[key] = value
        elif isinstance(value, (list, dict)):
            # Convert complex types to string representation
            sanitized[key] = str(value)
        else:
            # Convert other types to string
            sanitized[key] = str(value) if value is not None else "unknown"
    
    return sanitized

async def process_uploaded_files(files: List[UploadFile]) -> List[DocumentInfo]:
    """Process uploaded PDF files and add to vector store with hierarchical chunking"""
    collection = get_existing_collection()
    document_infos = []
    
    # Initialize hierarchical chunker
    chunker = HierarchicalLegalChunker(
        parent_chunk_size=2000,  # Larger chunks for context (full articles)
        child_chunk_size=500,    # Smaller chunks for precision (individual clauses)
        overlap=100
    )
    
    for file in files:
        if not (file.filename.lower().endswith('.pdf') or 
                file.filename.lower().endswith('.md') or 
                file.filename.lower().endswith('.markdown')):
            continue
            
        # Create temporary file
        if file.filename.lower().endswith('.pdf'):
            suffix = '.pdf'
        elif file.filename.lower().endswith('.md'):
            suffix = '.md'
        elif file.filename.lower().endswith('.markdown'):
            suffix = '.markdown'
        else:
            suffix = '.txt'  # fallback
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load document based on file type
            if file.filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif file.filename.lower().endswith(('.md', '.markdown')):
                loader = UnstructuredMarkdownLoader(tmp_file_path)
            else:
                print(f"❌ Unsupported file type: {file.filename}")
                continue
                
            docs = loader.load()
            
            # Add metadata to identify source
            for doc in docs:
                # Extract document title from first header if available
                document_title = "unknown"
                if doc.page_content:
                    # Clean and normalize the text for better matching
                    content_lines = doc.page_content.split('\n')
                    for line in content_lines[:10]:  # Check first 10 lines
                        line = line.strip()
                        # Look for document title patterns
                        if ('MƏCƏLLƏ' in line.upper() or 'MECELLə' in line.upper() or 
                            'AZƏRBAYCAN' in line.upper() and len(line) > 20 and len(line) < 200):
                            document_title = line
                            break
                        elif line.startswith('# ') and len(line) > 5:
                            document_title = line[2:].strip()
                            break
                
                doc.metadata["source_document"] = document_title if document_title != "unknown" else file.filename
                doc.metadata["document_filename"] = file.filename
                doc.metadata["document_type"] = Path(file.filename).stem
                doc.metadata["upload_id"] = str(uuid.uuid4())
            
            # Use hierarchical chunking instead of basic splitting
            parent_chunks, child_chunks = chunker.chunk_documents(docs)
            print(f"📄 Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks from {file.filename}")
            
            # Combine both parent and child chunks for storage
            all_chunks = parent_chunks + child_chunks
            
            # Prepare data for ChromaDB
            if all_chunks:
                # Generate embeddings and add to collection
                batch_size = 100
                total_added = 0
                
                print(f"🔄 Adding {len(all_chunks)} chunks to collection in batches of {batch_size}")
                
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i + batch_size]
                    batch_ids = [chunk.metadata["chunk_id"] for chunk in batch]
                    batch_documents = [chunk.page_content for chunk in batch]
                    # Sanitize metadata to prevent ChromaDB validation errors
                    batch_metadatas = [sanitize_metadata_for_chromadb(chunk.metadata) for chunk in batch]
                    
                    print(f"🔄 Adding batch {i//batch_size + 1}: {len(batch)} documents")
                    
                    # Precompute embeddings to avoid relying on collection EF
                    try:
                        batch_embeddings = embed_texts(batch_documents)
                    except Exception as ee:
                        raise RuntimeError(f"Embedding batch failed: {str(ee)}")

                    # Add to collection with explicit embeddings
                    collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                    )
                    total_added += len(batch)
                    print(f"✅ Added batch {i//batch_size + 1}: {len(batch)} documents (total: {total_added})")
                
                print(f"✅ Successfully added {total_added} chunks from {file.filename}")
            else:
                print(f"❌ No chunks created from {file.filename}")
            
            document_infos.append(DocumentInfo(
                filename=file.filename,
                total_pages=len(docs),
                chunks_created=len(all_chunks),
                parent_chunks=len(parent_chunks),
                child_chunks=len(child_chunks)
            ))
            
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            continue
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    return document_infos

def _init_job(job_id: str, filenames: List[str]):
    with jobs_lock:
        jobs_store[job_id] = {
            "status": "running",
            "error": None,
            "files": {name: FileProgress(filename=name).model_dump() for name in filenames},
        }

def _update_file_progress(job_id: str, filename: str, **kwargs):
    with jobs_lock:
        job = jobs_store.get(job_id)
        if not job:
            return
        fp = job["files"].get(filename)
        if not fp:
            fp = FileProgress(filename=filename).model_dump()
            job["files"][filename] = fp
        fp.update(kwargs)

def _finish_job(job_id: str, status: str = "completed", error: Optional[str] = None):
    with jobs_lock:
        job = jobs_store.get(job_id)
        if not job:
            return
        job["status"] = status
        job["error"] = error

def _compute_overall(job_id: str) -> Dict[str, Any]:
    with jobs_lock:
        job = jobs_store.get(job_id)
        if not job:
            return {"percent": 0.0, "chunks_total": 0, "chunks_done": 0}
        files = job["files"].values()
        total = sum((f.get("chunks_total") or 0) for f in files)
        done = sum(f.get("chunks_done") or 0 for f in files)
        percent = (done / total * 100.0) if total else 0.0
        return {"percent": percent, "chunks_total": total, "chunks_done": done}

def _process_uploaded_file_paths(file_entries: List[Dict[str, str]], job_id: str):
    """Background processor: takes temp-saved files and updates job progress while indexing."""
    try:
        collection = get_existing_collection()
        
        # Initialize hierarchical chunker
        chunker = HierarchicalLegalChunker(
            parent_chunk_size=2000,  # Larger chunks for context
            child_chunk_size=500,    # Smaller chunks for precision  
            overlap=100
        )

        for entry in file_entries:
            original_name = entry["filename"]
            tmp_path = entry["path"]
            _update_file_progress(job_id, original_name, status="running")
            try:
                # Load document based on file type
                if original_name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                elif original_name.lower().endswith(('.md', '.markdown')):
                    loader = UnstructuredMarkdownLoader(tmp_path)
                else:
                    _update_file_progress(job_id, original_name, status="failed", error=f"Unsupported file type: {original_name}")
                    continue
                    
                docs = loader.load()

                # Update pages_total early
                _update_file_progress(job_id, original_name, pages_total=len(docs))

                # Attach base metadata to each page
                for doc in docs:
                    # Extract document title from first header if available
                    document_title = "unknown"
                    if doc.page_content:
                        # Clean and normalize the text for better matching
                        content_lines = doc.page_content.split('\n')
                        for line in content_lines[:10]:  # Check first 10 lines
                            line = line.strip()
                            # Look for document title patterns
                            if ('MƏCƏLLƏ' in line.upper() or 'MECELLə' in line.upper() or 
                                'AZƏRBAYCAN' in line.upper() and len(line) > 20 and len(line) < 200):
                                document_title = line
                                break
                            elif line.startswith('# ') and len(line) > 5:
                                document_title = line[2:].strip()
                                break
                    
                    doc.metadata["source_document"] = document_title if document_title != "unknown" else original_name
                    doc.metadata["document_filename"] = original_name
                    doc.metadata["document_type"] = Path(original_name).stem
                    doc.metadata["job_id"] = job_id

                # Use hierarchical chunking
                parent_chunks, child_chunks = chunker.chunk_documents(docs)
                all_chunks = parent_chunks + child_chunks
                chunks_total = len(all_chunks)
                _update_file_progress(job_id, original_name, chunks_total=chunks_total, chunks_done=0, percent=0.0)

                if all_chunks:
                    batch_size = 100
                    for i in range(0, chunks_total, batch_size):
                        batch = all_chunks[i:i + batch_size]
                        batch_ids = [chunk.metadata["chunk_id"] for chunk in batch]
                        batch_documents = [chunk.page_content for chunk in batch]
                        # Sanitize metadata to prevent ChromaDB validation errors
                        batch_metadatas = [sanitize_metadata_for_chromadb(chunk.metadata) for chunk in batch]

                        # Precompute embeddings
                        batch_embeddings = embed_texts(batch_documents)

                        # Add to Chroma
                        collection.add(
                            ids=batch_ids,
                            embeddings=batch_embeddings,
                            documents=batch_documents,
                            metadatas=batch_metadatas,
                        )

                        # Progress update
                        with jobs_lock:
                            job = jobs_store.get(job_id)
                            if job and original_name in job["files"]:
                                fp = job["files"][original_name]
                                fp["chunks_done"] = (fp.get("chunks_done") or 0) + len(batch)
                                total = fp.get("chunks_total") or 0
                                fp["percent"] = (fp["chunks_done"] / total * 100.0) if total else 0.0

                _update_file_progress(job_id, original_name, status="completed")

            except Exception as fe:
                _update_file_progress(job_id, original_name, status="failed", error=str(fe))
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        _finish_job(job_id, status="completed")

    except Exception as e:
        _finish_job(job_id, status="failed", error=str(e))

@app.get("/", response_model=StatusResponse)
async def root():
    """Get API status"""
    try:
        collection = get_existing_collection()
        doc_count = collection.count()
        
        return StatusResponse(
            status="healthy",
            collection_exists=True,
            total_documents=doc_count,
            message="Azerbaijani Legal RAG Pipeline API is running"
        )
    except Exception as e:
        return StatusResponse(
            status="error",
            collection_exists=False,
            total_documents=0,
            message=f"Error: {str(e)}"
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and process PDF and Markdown documents"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    supported_files = [f for f in files if (
        f.filename.lower().endswith('.pdf') or 
        f.filename.lower().endswith('.md') or 
        f.filename.lower().endswith('.markdown')
    )]
    if not supported_files:
        raise HTTPException(status_code=400, detail="Only PDF and Markdown files are supported")
    
    try:
        document_infos = await process_uploaded_files(supported_files)
        
        collection = get_existing_collection()
        total_docs = collection.count()
        
        print(f"📊 Collection count after processing: {total_docs}")
        print(f"📋 Processed files: {[info.filename for info in document_infos]}")
        print(f"📋 Chunks per file: {[info.chunks_created for info in document_infos]}")
        
        return UploadResponse(
            message=f"Successfully processed {len(document_infos)} files",
            files_processed=[info.filename for info in document_infos],
            total_documents=total_docs
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/upload/start", response_model=UploadStartResponse)
async def upload_start(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Start an async upload/index job and return a job_id for frontend polling."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    supported_files = [f for f in files if (
        f.filename.lower().endswith('.pdf') or 
        f.filename.lower().endswith('.md') or 
        f.filename.lower().endswith('.markdown')
    )]
    if not supported_files:
        raise HTTPException(status_code=400, detail="Only PDF and Markdown files are supported")

    # Save files to temp paths to allow background processing after response returns
    temp_entries: List[Dict[str, str]] = []
    filenames: List[str] = []
    for f in supported_files:
        # Determine file extension for temp file
        if f.filename.lower().endswith('.pdf'):
            suffix = '.pdf'
        elif f.filename.lower().endswith('.md'):
            suffix = '.md'
        elif f.filename.lower().endswith('.markdown'):
            suffix = '.markdown'
        else:
            suffix = '.txt'  # fallback
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await f.read()
            tmp.write(content)
            temp_entries.append({"filename": f.filename, "path": tmp.name})
            filenames.append(f.filename)

    # Initialize job and dispatch background task
    job_id = str(uuid.uuid4())
    _init_job(job_id, filenames)
    background_tasks.add_task(_process_uploaded_file_paths, temp_entries, job_id)

    return UploadStartResponse(job_id=job_id, files=filenames)

@app.get("/upload/status/{job_id}", response_model=JobStatusResponse)
async def upload_status(job_id: str):
    with jobs_lock:
        job = jobs_store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        # Compute overall on the fly for freshness
    overall = _compute_overall(job_id)
    with jobs_lock:
        job = jobs_store.get(job_id)
        return JobStatusResponse(
            job_id=job_id,
            status=job.get("status", "pending"),
            files={k: FileProgress(**v) for k, v in job.get("files", {}).items()},
            overall=overall,
            error=job.get("error"),
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG system"""
    global graph
    
    if not graph:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # Use provided session_id or generate new one
    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Run the conversation graph
        response_content = ""
        for step in graph.stream(
            {"messages": [{"role": "user", "content": request.message}]},
            stream_mode="values",
            config=config
        ):
            if step["messages"]:
                last_message = step["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
        
        # Use the exact same retrieved documents that were sent to the LLM
        global last_retrieved_docs
        top_sources: List[SourceInfo] = []
        
        for doc in last_retrieved_docs:  # Show ALL retrieved documents, not just top 5
            metadata = doc.get('metadata', {})
            content = doc.get('content', '')
            
            # Extract meaningful source name - prioritize document_title over source_document
            source_name = (
                metadata.get('document_title', 'unknown') if metadata.get('document_title', 'unknown') != 'unknown' 
                else metadata.get('source_document', 'unknown')
            )
            
            if source_name == 'unknown' or '.md' in source_name:
                # Fallback to chunk ID or filename
                source_name = metadata.get('chunk_id', metadata.get('document_filename', 'unknown'))
            
            top_sources.append(SourceInfo(
                document_name=source_name,
                retrieved_content=content,  # Show FULL content, not truncated
                page_number=metadata.get('page', None),
                chunk_type=metadata.get('chunk_type', 'unknown'),
                hierarchy_level=metadata.get('structure_level', 0),
                clause_number=metadata.get('clause_number', None)
            ))

        return ChatResponse(
            response=response_content or "Sorry, I couldn't generate a response.",
            session_id=session_id,
            sources=top_sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """Retrieve relevant document chunks for a query without generating LLM response using hierarchical retrieval"""
    try:
        collection = get_existing_collection()
        
        # Initialize hierarchical retriever with embed function
        retriever = HierarchicalRetriever(collection, embed_texts)
        
        # Use adaptive hierarchical retrieval
        retrieved_docs = retriever.retrieve_with_hierarchy(request.query, n_results=request.n_results)
        
        # Create serialized content for logging (same format as chat endpoint)
        serialized_content = "\n\n".join(
            (f"[Source: {doc['metadata'].get('source_document', 'Unknown')}]\n{doc['content']}")
            for doc in retrieved_docs
        )
        
        # Save retrieved content to file for analysis (same as chat endpoint)
        save_retrieved_content_to_file(request.query, retrieved_docs, serialized_content)
        
        # Prepare response with enhanced metadata
        retrieved_chunks = []
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            
            # Extract meaningful source name - prioritize document_title over source_document
            source_name = (
                metadata.get('document_title', 'unknown') if metadata.get('document_title', 'unknown') != 'unknown' 
                else metadata.get('source_document', 'unknown')
            )
            
            if source_name == 'unknown' or '.md' in source_name:
                source_name = metadata.get('chunk_id', metadata.get('document_filename', 'unknown'))
            
            retrieved_chunks.append(RetrievedChunk(
                content=doc.get('content', ''),
                metadata={
                    **metadata,
                    'display_source_name': source_name  # Add clean source name for display
                },
                distance=doc.get('distance', None)
            ))
        
        return RetrieveResponse(
            query=request.query,
            chunks=retrieved_chunks,
            total_results=len(retrieved_chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get detailed system status"""
    try:
        collection = get_existing_collection()
        doc_count = collection.count()
        
        return StatusResponse(
            status="operational",
            collection_exists=True,
            total_documents=doc_count,
            message=f"System operational with {doc_count} documents indexed"
        )
    except Exception as e:
        return StatusResponse(
            status="error",
            collection_exists=False,
            total_documents=0,
            message=f"System error: {str(e)}"
        )

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store"""
    try:
        # Fetch collection directly to avoid any recreate logic during deletion
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            # If collection doesn't exist, nothing to delete
            return {
                "message": "No collection found; nothing to delete",
                "status": "success",
                "deleted_count": 0,
            }
        
        # Get all document IDs first
        all_docs = collection.get()
        
        if all_docs['ids']:
            # Delete all documents by their IDs
            collection.delete(ids=all_docs['ids'])
            deleted_count = len(all_docs['ids'])
            return {
                "message": f"Successfully cleared {deleted_count} documents", 
                "status": "success",
                "deleted_count": deleted_count
            }
        else:
            return {
                "message": "No documents found to delete", 
                "status": "success",
                "deleted_count": 0
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check ChromaDB connection
        collection = get_existing_collection()
        doc_count = collection.count()
        
        # Check if components are initialized
        components_ok = all([chroma_client, embedding_function, llm, graph])
        
        if components_ok:
            return {
                "status": "healthy",
                "components": {
                    "chromadb": "connected",
                    "embedding_function": "initialized",
                    "llm": "initialized",
                    "graph": "initialized"
                },
                "documents_count": doc_count
            }
        else:
            return {
                "status": "degraded",
                "components": {
                    "chromadb": "connected" if chroma_client else "not_initialized",
                    "embedding_function": "initialized" if embedding_function else "not_initialized",
                    "llm": "initialized" if llm else "not_initialized",
                    "graph": "initialized" if graph else "not_initialized"
                },
                "documents_count": doc_count
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
