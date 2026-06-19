from pathlib import Path 
import fitz
import sys


_ROOT = Path(__file__).resolve().parent.parent

def load_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    Open a pdf file and return ordered page records

    Each record contains:
     - page : 1-based page index (for citations)
     - text : extracted text for that page
     - source : PDF file name
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    source = pdf_path.name
    pages = []

    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            text = (page.get_text() or "").strip()
            pages.append({
                "page":page.number + 1,
                "text":text,
                "source":source
            })
    finally:
        doc.close()
    
    return pages

if __name__ == "__main__":
    default = _ROOT / "data" / "paper.pdf"
    path = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else default

    if not path.is_absolute():
        path = (_ROOT / path).resolve()
    
    rows = load_pdf_pages(path)
    print("Number of pages:", len(rows))
    if rows:
        first = rows[0]
        print("First Page: ", first["page"], "chars:", len(first["text"]))
        print(first["text"][:1200], "...")
    