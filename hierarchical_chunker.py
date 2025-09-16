"""
Hierarchical chunking implementation for structured legal documents.
This module provides improved chunking strategies for legal documents with clear hierarchical structure.
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class HierarchicalLegalChunker:
    """
    Advanced chunker for legal documents with hierarchical structure.
    Implements parent-child chunking strategy with metadata preservation.
    """
    
    def __init__(self, 
                 parent_chunk_size: int = 2000,
                 child_chunk_size: int = 500,
                 overlap: int = 100):
        """
        Initialize the hierarchical chunker.
        
        Args:
            parent_chunk_size: Size of parent chunks (broader context)
            child_chunk_size: Size of child chunks (precise matching)  
            overlap: Overlap between chunks
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.overlap = overlap
        
        # Markdown header splitter for structure preservation
        # Based on actual document structure: Document Title -> Section -> Chapter -> Article
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Document_Title"),     # e.g., "AZƏRBAYCAN RESPUBLİKASININ AİLƏ MƏCƏLLƏSİ"
                ("##", "Section"),           # e.g., "Birinci bölmə. Ümumi müddəalar"
                ("###", "Chapter"),          # e.g., "1-ci fəsil. Ailə qanunvericiliği"
                ("####", "Article_Header")   # e.g., "Maddə 1. Azərbaycan Respublikasının ailə qanunvericiliyi"
            ]
        )
        
        # Text splitters for parent and child chunks
        # Parent chunks: larger sections that preserve context (articles with their sub-clauses)
        # Child chunks: individual clauses and sub-clauses for precise matching
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, 
            chunk_overlap=overlap//2,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_clause_number(self, text: str) -> Optional[str]:
        """Extract clause numbers like '1.1', '1.1.1', 'Maddə 1' etc."""
        patterns = [
            r'(\d+\.\d+\.\d+)',  # 1.1.1 format (sub-sub clauses)
            r'(\d+\.\d+)',       # 1.1 format (sub clauses)
            r'(Maddə\s+\d+)',    # Maddə 1 format (main articles)
            r'(Article\s+\d+)',  # Article 1 format (English)
            r'(\d+\s*-\s*ci\s+fəsil)',  # "1-ci fəsil" format (chapters)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def build_hierarchical_path(self, headers: Dict[str, str]) -> str:
        """Build a hierarchical path from headers like 'Document > Section > Chapter > Article'"""
        path_parts = []
        for level in ["Document_Title", "Section", "Chapter", "Article_Header"]:
            if headers.get(level):
                path_parts.append(headers[level])
        return " > ".join(path_parts)

    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
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

    def enhance_metadata(self, 
                        chunk: Document, 
                        headers: Dict[str, str],
                        chunk_type: str,
                        parent_id: Optional[str] = None,
                        chunk_index: int = 0) -> Dict[str, Any]:
        """Enhanced metadata with essential hierarchical information only"""
        
        hierarchical_path = self.build_hierarchical_path(headers)
        
        # Get document title from headers first, then fallback to original metadata
        document_title = headers.get("Document_Title") or chunk.metadata.get("source_document", "unknown")
        
        metadata = {
            # Essential hierarchical structure only
            "document_title": document_title,
            "section": headers.get("Section") or "unknown", 
            "chapter": headers.get("Chapter") or "unknown",
            "article_header": headers.get("Article_Header") or "unknown",
            
            # Essential chunk information only
            "chunk_type": chunk_type,  # "parent" or "child"
            "chunk_index": chunk_index,
            "hierarchical_path": hierarchical_path or "unknown",
            "content_length": len(chunk.page_content),
            
            # Preserve important original metadata
            "source_document": chunk.metadata.get("source_document", document_title),
            "document_filename": chunk.metadata.get("document_filename", "unknown"),
        }
        
        # Sanitize metadata to ensure ChromaDB compatibility
        return self.sanitize_metadata(metadata)

    def extract_article_content(self, article_section: str) -> List[str]:
        """
        Extract individual clauses from article content.
        Since most content is under Article headers, we need to split by sub-clauses.
        
        Returns list of clause texts with their numbers preserved.
        """
        clauses = []
        
        # Split by numbered clauses (1.1, 1.2, etc.) and sub-clauses (1.1.1, 1.1.2, etc.)
        # Pattern matches: "- 1.1." or "- 1.1.1." at the beginning of lines
        clause_pattern = r'^[-•]\s*(\d+\.\d+(?:\.\d+)?)\.\s*(.+?)(?=^[-•]\s*\d+\.\d+|\Z)'
        
        matches = re.finditer(clause_pattern, article_section, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            clause_number = match.group(1)
            clause_content = match.group(2).strip()
            
            # Combine clause number with content
            full_clause = f"{clause_number}. {clause_content}"
            clauses.append(full_clause)
        
        # If no numbered clauses found, treat entire content as one clause
        if not clauses:
            clauses = [article_section.strip()]
        
        return clauses

    def chunk_document(self, document: Document) -> Tuple[List[Document], List[Document]]:
        """
        Split document into parent and child chunks with hierarchical metadata.
        
        For legal documents:
        - Parent chunks: Complete articles with all their sub-clauses (broader context)
        - Child chunks: Individual clauses and sub-clauses (precise matching)
        
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        parent_chunks = []
        child_chunks = []
        
        # First split by markdown headers to preserve structure
        try:
            header_splits = self.header_splitter.split_text(document.page_content)
        except Exception as e:
            print(f"Header splitting failed: {e}, falling back to basic splitting")
            header_splits = [Document(page_content=document.page_content, metadata={})]
        
        for header_doc in header_splits:
            current_headers = header_doc.metadata.copy()
            
            # Check if this is an Article section (4th level header with actual content)
            is_article_section = current_headers.get("Article_Header") and len(header_doc.page_content.strip()) > 50
            
            if is_article_section:
                # For articles, create one parent chunk with the entire article
                parent_id = f"parent_{uuid.uuid4().hex[:8]}_{len(parent_chunks)}"
                
                # Create parent chunk (entire article)
                parent_chunk = Document(
                    page_content=header_doc.page_content,
                    metadata=self.enhance_metadata(
                        header_doc,
                        current_headers,
                        chunk_type="parent",
                        chunk_index=0
                    )
                )
                parent_chunk.metadata["chunk_id"] = parent_id
                parent_chunks.append(parent_chunk)
                
                # Extract individual clauses for child chunks
                article_clauses = self.extract_article_content(header_doc.page_content)
                
                for j, clause_text in enumerate(article_clauses):
                    if clause_text.strip():  # Only create chunk if content exists
                        child_id = f"child_{uuid.uuid4().hex[:8]}_{parent_id}_{j}"
                        
                        child_chunk = Document(
                            page_content=clause_text,
                            metadata=self.enhance_metadata(
                                Document(page_content=clause_text, metadata=header_doc.metadata),
                                current_headers,
                                chunk_type="child",
                                parent_id=parent_id,
                                chunk_index=j
                            )
                        )
                        child_chunk.metadata["chunk_id"] = child_id
                        child_chunks.append(child_chunk)
            
            else:
                # For non-article sections (headers without much content), use standard splitting
                if len(header_doc.page_content) > self.parent_chunk_size:
                    parent_splits = self.parent_splitter.split_documents([header_doc])
                else:
                    parent_splits = [header_doc]
                
                for i, parent_chunk in enumerate(parent_splits):
                    parent_id = f"parent_{uuid.uuid4().hex[:8]}_{len(parent_chunks)}_{i}"
                    
                    # Enhance parent chunk metadata
                    parent_chunk.metadata = self.enhance_metadata(
                        parent_chunk, 
                        current_headers,
                        chunk_type="parent",
                        chunk_index=i
                    )
                    parent_chunk.metadata["chunk_id"] = parent_id
                    parent_chunks.append(parent_chunk)
                    
                    # Create child chunks from this parent
                    if len(parent_chunk.page_content) > self.child_chunk_size:
                        child_splits = self.child_splitter.split_documents([parent_chunk])
                    else:
                        child_splits = [parent_chunk]
                    
                    for j, child_chunk in enumerate(child_splits):
                        child_id = f"child_{uuid.uuid4().hex[:8]}_{parent_id}_{j}"
                        
                        # Enhance child chunk metadata  
                        child_chunk.metadata = self.enhance_metadata(
                            child_chunk,
                            current_headers, 
                            chunk_type="child",
                            parent_id=parent_id,
                            chunk_index=j
                        )
                        child_chunk.metadata["chunk_id"] = child_id
                        child_chunks.append(child_chunk)
        
        return parent_chunks, child_chunks

    def chunk_documents(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """Process multiple documents"""
        all_parent_chunks = []
        all_child_chunks = []
        
        for doc in documents:
            parent_chunks, child_chunks = self.chunk_document(doc)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
            
        return all_parent_chunks, all_child_chunks


class HierarchicalRetriever:
    """
    Retriever that uses hierarchical chunking strategy.
    Searches child chunks for precision, returns parent chunks for context.
    """
    
    def __init__(self, collection, embed_function):
        self.collection = collection
        self.embed_function = embed_function
    
    def retrieve_with_hierarchy(self, 
                              query: str, 
                              n_results: int = 10,
                              return_type: str = "adaptive") -> List[Dict[str, Any]]:
        """
        Retrieve documents using hierarchical strategy.
        
        Args:
            query: Search query
            n_results: Number of results to return
            return_type: "child", "parent", or "adaptive"
        
        Returns:
            List of retrieved document chunks with metadata
        """
        
        # Search through child chunks for precision
        query_embedding = self.embed_function([query])
        
        # First get child chunks for precise matching
        child_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results * 2,  # Get more to have options
            include=["metadatas", "documents", "distances"],
            where={"chunk_type": "child"}
        )
        
        if return_type == "child":
            return self._format_results(child_results)
        
        elif return_type == "parent":
            # Get parent chunks for retrieved children
            parent_ids = []
            for metadata in child_results.get('metadatas', [[]])[0]:
                parent_id = metadata.get('parent_chunk_id')
                if parent_id and parent_id not in parent_ids:
                    parent_ids.append(parent_id)
            
            # Query for parent chunks
            parent_results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=len(parent_ids),
                include=["metadatas", "documents", "distances"],
                where={"chunk_id": {"$in": parent_ids}}
            )
            
            return self._format_results(parent_results)
        
        else:  # adaptive
            # Return mix of child and parent based on query type
            # For specific clauses, prefer child chunks
            # For broader concepts, prefer parent chunks
            
            # Check for specific query patterns
            is_specific_clause = any(re.search(pattern, query, re.IGNORECASE) for pattern in [
                r'\d+\.\d+\.\d+',    # 1.1.1 format
                r'\d+\.\d+',         # 1.1 format  
                r'maddə\s+\d+',      # Maddə 1
                r'article\s+\d+',    # Article 1
                r'bənd',             # specific clause
                r'clause'
            ])
            
            # Check for broader conceptual queries
            is_broad_query = any(word in query.lower() for word in [
                'hansı', 'nə', 'necə', 'niyə',  # Azerbaijani question words
                'what', 'how', 'why', 'when',    # English question words
                'haqqında', 'about', 'regarding',
                'qanun', 'law', 'hüquq', 'right'
            ])
            
            if is_specific_clause:
                # For specific clauses, return child chunks for precision
                return self._format_results(child_results)[:n_results]
            elif is_broad_query:
                # For broad queries, prefer parent chunks for context
                parent_results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=n_results,
                    include=["metadatas", "documents", "distances"],
                    where={"chunk_type": "parent"}
                )
                return self._format_results(parent_results)[:n_results]
            else:
                # Mixed approach: combine both types
                parent_results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=n_results//2,
                    include=["metadatas", "documents", "distances"],
                    where={"chunk_type": "parent"}
                )
                
                child_formatted = self._format_results(child_results)[:n_results//2]
                parent_formatted = self._format_results(parent_results)[:n_results//2]
                
                # Combine and sort by relevance score
                all_results = child_formatted + parent_formatted
                all_results.sort(key=lambda x: x.get('distance', 1.0))
                
                return all_results[:n_results]
    
    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB results into standard format"""
        formatted = []
        
        if not results.get('documents') or not results['documents'][0]:
            return formatted
        
        docs = results['documents'][0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        for i, doc in enumerate(docs):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else None
            
            formatted.append({
                'content': doc,
                'metadata': metadata,
                'distance': distance
            })
        
        return formatted