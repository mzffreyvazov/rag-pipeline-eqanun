#!/usr/bin/env python3
import time
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import signal
import sys

# Setup logging
log_file = Path.home() / 'pdf-converter.log'  # Use user's home directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

class PDFHandler(FileSystemEventHandler):
    def __init__(self, converter, out_dir):
        self.converter = converter
        self.out_dir = out_dir
        
    def on_created(self, event):
        if event.is_dir:
            return
            
        if event.src_path.endswith('.pdf'):
            self.process_pdf(Path(event.src_path))
    
    def process_pdf(self, pdf_file):
        try:
            logging.info(f"Processing {pdf_file}...")
            result = self.converter.convert(str(pdf_file))
            md = result.document.export_to_markdown()
            out_file = self.out_dir / (pdf_file.stem + ".md")
            out_file.write_text(md, encoding="utf-8")
            logging.info(f"Successfully converted {pdf_file} to {out_file}")
        except Exception as e:
            logging.error(f"Error processing {pdf_file}: {str(e)}")

def signal_handler(sig, frame):
    logging.info('Shutting down PDF converter...')
    sys.exit(0)

def main():
    # Setup directories
    assets_dir = Path("assets/mecelleler")
    out_dir = Path("assets/mecelleler-outputs")
    out_dir.mkdir(exist_ok=True)
    
    # Create converter
    converter = DocumentConverter()
    
    # Process existing files
    logging.info("Processing existing PDF files...")
    for pdf_file in assets_dir.glob("*.pdf"):
        handler = PDFHandler(converter, out_dir)
        handler.process_pdf(pdf_file)
    
    # Setup file watcher for new files
    logging.info("Starting file watcher...")
    event_handler = PDFHandler(converter, out_dir)
    observer = Observer()
    observer.schedule(event_handler, str(assets_dir), recursive=False)
    observer.start()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logging.info("PDF converter is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(60)  # Check every minute
            logging.info("PDF converter is still running...")
    except KeyboardInterrupt:
        logging.info("Received interrupt signal")
    finally:
        observer.stop()
        observer.join()
        logging.info("PDF converter stopped")

if __name__ == "__main__":
    main()