import re
from pathlib import Path

def enhance_markdown(text: str) -> str:
    # Replace lines that start with optional spaces + "##" + whitespace + "Maddə"
    # with the same indentation + "###### Maddə" preserving the rest of the line.
    pattern = re.compile(r'^(\s*)##\s+(Maddə\b.*)$', flags=re.MULTILINE)
    return pattern.sub(r'\1###### \2', text)

def main():
    # Simple interactive file input (no CLI)
    inp_path = input("Input markdown file path: ").strip()
    if not inp_path:
        print("No input file provided.")
        return

    inp = Path(inp_path)
    if not inp.is_file():
        print(f"Input file not found: {inp}")
        return

    out_path_str = input("Output file path (leave empty to create <name>_enhanced<ext>): ").strip()
    if out_path_str:
        out_path = Path(out_path_str)
    else:
        out_path = inp.with_name(inp.stem + "_enhanced" + inp.suffix)

    content = inp.read_text(encoding='utf-8')
    new_content = enhance_markdown(content)
    out_path.write_text(new_content, encoding='utf-8')
    print(f"Wrote enhanced file to: {out_path}")

if __name__ == '__main__':
    main()