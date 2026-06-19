""" Split per-page PDF text into overlapping chunks with metadata """

from pathlib import Path
import sys

from config import CHUNK_SIZE, CHUNK_OVERLAP
from load_pdf import load_pdf_pages

_root = Path(__file__).resolve().parent.parent

def split_with_overlap(text, size, overlap):
    if overlap >= size:
        raise ValueError("Overlap must be less than chunk size")
    
    text = text.strip()
    if not text:
        return []
    
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks

def chunk_pages(pages, size, overlap):
    size = CHUNK_SIZE if size is None else size
    overlap = CHUNK_OVERLAP if overlap is None else overlap

    chunks = []
    for i in pages:
        page = i["page"]
        source = i["source"]
        data = split_with_overlap(i["text"], size, overlap)
        for j, text in enumerate(data):
            chunks.append({
                "page":page,
                "source":source,
                "text":text,
                "chunk_id":f"{page}-{j}"
            })

    return chunks

if __name__ == "__main__":
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else _root / "data"
    if not pdf.is_absolute():
        pdf = (_root / pdf).resolve()
    
    pages = load_pdf_pages(pdf)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    # print(type(chunks))
    # print(chunks[0])
    print(f"Number of pages: {len(pages)}, chunks: {len(chunks)}")
    print("First 3 chunks:")
    for i in chunks[:3]:
        print(f"Chunk {i['chunk_id']}: {i['text'][:100]}...")
