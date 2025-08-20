from docling.document_converter import DocumentConverter
from pathlib import Path

assets_dir = Path("assets/mecelleler")
out_dir = Path("assets/mecelleler-outputs")
out_dir.mkdir(exist_ok=True)

converter = DocumentConverter()

for pdf_file in assets_dir.glob("*.pdf"):
    print(f"Processing {pdf_file}...")
    result = converter.convert(str(pdf_file))
    md = result.document.export_to_markdown()
    out_file = out_dir / (pdf_file.stem + ".md")
    out_file.write_text(md, encoding="utf-8")
    print(f"Wrote markdown to {out_file.resolve()}")
