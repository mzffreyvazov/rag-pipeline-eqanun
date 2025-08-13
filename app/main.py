import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import shutil

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import vertexai
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Initialize Vertex AI
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if PROJECT_ID and REGION:
    vertexai.init(project=PROJECT_ID, location=REGION)

# Global variables
chroma_client = None
embedding_function = None
llm = None
vector_store = None
graph = None
collection_name = "legal_documents"

def initialize_components():
    """Initialize all AI components"""
    global chroma_client, embedding_function, llm, vector_store
    
    try:
        # Initialize ChromaDB with persistent storage
        chroma_client = chromadb.PersistentClient(
            path="./chroma_data",
            settings=Settings(allow_reset=True)
        )
        
        # Initialize embedding function
        embedding_function = VertexAIEmbeddings(model_name="gemini-embedding-001")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        print("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing components: {str(e)}")
        return False

def create_rag_graph():
    """Create the RAG conversation graph"""
    global llm
    
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        try:
            collection = get_or_create_collection()
            
            # Use ChromaDB's query method directly
            results = collection.query(
                query_texts=[query],
                n_results=5
            )
            
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    retrieved_docs.append({
                        'content': doc,
                        'metadata': metadata
                    })
            
            serialized = "\n\n".join(
                (f"Source: {doc['metadata']}\nContent: {doc['content']}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        except Exception as e:
            return f"Error retrieving documents: {str(e)}", []

    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        system = SystemMessage(
            "You are a helpful Azerbaijani legal assistant with access to a 'retrieve' tool that searches the indexed PDFs. "
            "Call the 'retrieve' tool when the user's question likely requires grounding in the documents (facts, definitions, citations, articles). "
            "If the request is a greeting or meta-chat that doesn't need grounding, answer directly. "
            "If you are uncertain or the question appears document-specific, prefer calling the tool."
        )
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

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
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
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
    description="A conversational RAG system for Azerbaijani legal documents",
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

class ChatResponse(BaseModel):
    response: str
    session_id: str

class UploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_documents: int

class DocumentInfo(BaseModel):
    filename: str
    total_pages: int
    chunks_created: int

class StatusResponse(BaseModel):
    status: str
    collection_exists: bool
    total_documents: int
    message: str

def get_or_create_collection():
    """Get or create the ChromaDB collection"""
    global chroma_client, collection_name
    
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Azerbaijani legal documents collection"}
        )
        return collection
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing collection: {str(e)}")

def initialize_components():
    """Initialize all AI components"""
    global chroma_client, embedding_function, llm, vector_store
    
    try:
        # Initialize ChromaDB with persistent storage
        chroma_client = chromadb.PersistentClient(
            path="./chroma_data",
            settings=Settings(allow_reset=True)
        )
        
        # Initialize embedding function
        embedding_function = VertexAIEmbeddings(model_name="gemini-embedding-001")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        print("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing components: {str(e)}")
        return False

def create_rag_graph():
    """Create the RAG conversation graph"""
    global llm
    
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        try:
            collection = get_or_create_collection()
            
            # Use ChromaDB's query method directly
            results = collection.query(
                query_texts=[query],
                n_results=5
            )
            
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    retrieved_docs.append({
                        'content': doc,
                        'metadata': metadata
                    })
            
            serialized = "\n\n".join(
                (f"Source: {doc['metadata']}\nContent: {doc['content']}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        except Exception as e:
            return f"Error retrieving documents: {str(e)}", []

    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        system = SystemMessage(
            "You are a helpful Azerbaijani legal assistant with access to a 'retrieve' tool that searches the indexed PDFs. "
            "Call the 'retrieve' tool when the user's question likely requires grounding in the documents (facts, definitions, citations, articles). "
            "If the request is a greeting or meta-chat that doesn't need grounding, answer directly. "
            "If you are uncertain or the question appears document-specific, prefer calling the tool."
        )
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

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
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
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

async def process_uploaded_files(files: List[UploadFile]) -> List[DocumentInfo]:
    """Process uploaded PDF files and add to vector store"""
    collection = get_or_create_collection()
    document_infos = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1536, 
        chunk_overlap=305
    )
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue
            
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            # Add metadata to identify source
            for doc in docs:
                doc.metadata["source_document"] = file.filename
                doc.metadata["document_type"] = Path(file.filename).stem
                doc.metadata["upload_id"] = str(uuid.uuid4())
            
            # Split documents
            all_splits = text_splitter.split_documents(docs)
            
            # Prepare data for ChromaDB
            if all_splits:
                # Generate embeddings and add to collection
                batch_size = 100
                total_added = 0
                
                for i in range(0, len(all_splits), batch_size):
                    batch = all_splits[i:i + batch_size]
                    batch_ids = [f"{file.filename}_{j}" for j in range(i, i + len(batch))]
                    batch_documents = [doc.page_content for doc in batch]
                    batch_metadatas = [doc.metadata for doc in batch]
                    
                    # Add to collection
                    collection.add(
                        ids=batch_ids,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    total_added += len(batch)
            
            document_infos.append(DocumentInfo(
                filename=file.filename,
                total_pages=len(docs),
                chunks_created=len(all_splits)
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

@app.get("/", response_model=StatusResponse)
async def root():
    """Get API status"""
    try:
        collection = get_or_create_collection()
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
    """Upload and process PDF documents"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    pdf_files = [f for f in files if f.filename.lower().endswith('.pdf')]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        document_infos = await process_uploaded_files(pdf_files)
        
        collection = get_or_create_collection()
        total_docs = collection.count()
        
        return UploadResponse(
            message=f"Successfully processed {len(document_infos)} files",
            files_processed=[info.filename for info in document_infos],
            total_documents=total_docs
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

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
        
        return ChatResponse(
            response=response_content or "Sorry, I couldn't generate a response.",
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get detailed system status"""
    try:
        collection = get_or_create_collection()
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
        collection = get_or_create_collection()
        
        # Delete all documents
        collection.delete()
        
        return {"message": "All documents cleared successfully", "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check ChromaDB connection
        collection = get_or_create_collection()
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
