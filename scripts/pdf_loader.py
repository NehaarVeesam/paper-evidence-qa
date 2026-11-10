from pathlib import Path

from pypdf import PdfReader


def load_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append({
            "text": text,
            "source": pdf_path.name,
            "page_number": page_number,
        })

    return documents