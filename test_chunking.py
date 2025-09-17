#!/usr/bin/env python3
"""Test script for HierarchicalLegalChunker.

Loads a markdown file, applies hierarchical legal chunking, and writes full
parent and child chunk contents plus metadata into a log file for inspection.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to Python path to import local modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hierarchical_chunker import HierarchicalLegalChunker
# from langchain_community.document_loaders import UnstructuredMarkdownLoader # <-- REMOVE THIS
from langchain_core.documents import Document

def load_markdown_file(file_path: str) -> List[Document]:
    """Load a markdown file as raw text to preserve header structure."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the raw text content from the file
    try:
        raw_content = Path(file_path).read_text(encoding='utf-8')
    except Exception as e:
        raise IOError(f"Could not read file {file_path}: {e}")

    # Create a single Document object from the raw content
    doc = Document(page_content=raw_content)

    # Add source metadata
    doc.metadata['source_document'] = Path(file_path).name
    doc.metadata['document_filename'] = Path(file_path).name

    return [doc] # Return as a list of one document

def display_chunks(chunks: List[Document], chunk_type: str, log_file):
    """Display chunks with their metadata."""
    log_file.write(f"\n{'='*50}\n")
    log_file.write(f"{chunk_type.upper()} CHUNKS ({len(chunks)} total)\n")
    log_file.write(f"{'='*50}\n")

    for i, chunk in enumerate(chunks, 1):
        log_file.write(f"\n--- {chunk_type.upper()} CHUNK {i} ---\n")
        log_file.write(f"Content Length: {len(chunk.page_content)} characters\n")
        log_file.write(f"Content:\n{chunk.page_content}\n")
        log_file.write("\n" + "="*50 + "\n")
        log_file.write("Metadata:\n")
        for key, value in chunk.metadata.items():
            log_file.write(f"  {key}: {value}\n")
        log_file.write("-" * 40 + "\n")

def main():
    # INPUT YOUR MARKDOWN FILE PATH HERE
    file_path = "D:\\Downloads\\code\\code\\Extra-Projects\\rag-pipeline-eqanun\\document_formatting\\mecelleler-raw\\mecelleler-cleaned-final\\cleaned_document-ailə.md"  # <-- Make sure this path is correct

    # Log file path
    log_file_path = "chunking_test_output.txt"

    # A simple check to ensure the path has been changed
    if "path/to/your/markdown/file.md" in file_path:
        print("ERROR: Please update the file_path variable with your actual markdown file path!")
        return

    try:
        # Open log file
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            # Load the markdown file
            print(f"Loading markdown file: {file_path}")
            documents = load_markdown_file(file_path)
            print(f"Loaded {len(documents)} document(s) with raw content.")

            if not documents:
                print("No documents loaded. Check the file path and content.")
                return

            # Initialize the hierarchical chunker
            chunker = HierarchicalLegalChunker(
                parent_chunk_size=2000,
                child_chunk_size=500,
                overlap=100
            )

            # Chunk the documents
            print("\nChunking documents...")
            parent_chunks, child_chunks = chunker.chunk_documents(documents)

            # Display results
            print(f"\nChunking completed!")
            print(f"Parent chunks: {len(parent_chunks)}")
            print(f"Child chunks: {len(child_chunks)}")
            print(f"Writing detailed output to {log_file_path}...")

            # Write header to log file
            log_file.write("HIERARCHICAL CHUNKING TEST RESULTS\n")
            log_file.write(f"Input file: {file_path}\n")
            log_file.write(f"Generated on: {Path(__file__).name}\n")
            log_file.write(f"{'='*60}\n\n")

            # Display parent chunks
            display_chunks(parent_chunks, "parent", log_file)

            # Display child chunks
            display_chunks(child_chunks, "child", log_file)

            # Summary statistics
            log_file.write(f"\n{'='*50}\n")
            log_file.write("SUMMARY STATISTICS\n")
            log_file.write(f"{'='*50}\n")
            log_file.write(f"Total documents processed: {len(documents)}\n")
            log_file.write(f"Total parent chunks: {len(parent_chunks)}\n")
            log_file.write(f"Total child chunks: {len(child_chunks)}\n")

            if parent_chunks:
                avg_parent_length = sum(len(c.page_content) for c in parent_chunks) / len(parent_chunks)
                log_file.write(f"Average parent chunk length: {avg_parent_length:.0f} characters\n")

            if child_chunks:
                avg_child_length = sum(len(c.page_content) for c in child_chunks) / len(child_chunks)
                log_file.write(f"Average child chunk length: {avg_child_length:.0f} characters\n")

        print(f"\nDetailed output has been written to {log_file_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()