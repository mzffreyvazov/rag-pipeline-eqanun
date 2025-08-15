from docling.document_converter import DocumentConverter
from pathlib import Path

source = "assets/mulki-mecelle-full.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)

# Export markdown and save to a file under outputs/
md = result.document.export_to_markdown()
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
out_file = out_dir / (Path(source).stem + ".md")
out_file.write_text(md, encoding="utf-8")
print(f"Wrote markdown to {out_file.resolve()}")