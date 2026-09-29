import os
import hashlib
import tiktoken
from pathlib import Path
from pypdf import PdfReader

from app.rag.vectorstore import vector_store

DOCS_DIR = Path("data/docs")
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = None
    return _encoder


def extract_pdf_text(path: Path) -> str:
    """Extracts text from a PDF, page by page, with page markers preserved."""
    reader = PdfReader(str(path))
    pages_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            # Keep a page marker inline so we can attach page numbers to chunks later
            pages_text.append(f"[[PAGE {page_num}]]\n{text}")
    return "\n\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    enc = get_encoder()
    if enc is not None:
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(enc.decode(chunk_tokens))
            start += chunk_size - overlap
        return chunks

    # Fallback to approximate word-based chunking if tiktoken encoder is unavailable
    words = text.split()
    word_chunk_size = max(50, int(chunk_size * 0.75))
    word_overlap = max(10, int(overlap * 0.75))
    chunks = []
    start = 0
    while start < len(words):
        end = start + word_chunk_size
        chunks.append(" ".join(words[start:end]))
        start += word_chunk_size - word_overlap
    return chunks



def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Reads all .txt/.md/.pdf files from the docs directory."""
    docs = []
    for path in docs_dir.glob("**/*"):
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({"filename": path.name, "text": text})

        elif suffix == ".pdf":
            try:
                text = extract_pdf_text(path)
                if not text.strip():
                    print(f"Warning: no extractable text in {path.name} (likely scanned/image-based — needs OCR)")
                    continue
                docs.append({"filename": path.name, "text": text})
            except Exception as e:
                print(f"Failed to read {path.name}: {e}")

    return docs


def ingest_all():
    docs = load_documents()
    if not docs:
        print(f"No documents found in {DOCS_DIR}. Add .txt, .md, or .pdf files and re-run.")
        return

    total_chunks = 0
    for doc in docs:
        vector_store.delete_by_source(doc["filename"])

        chunks = chunk_text(doc["text"])
        ids, texts, metadatas = [], [], []

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{doc['filename']}-{i}".encode()).hexdigest()
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({
                "source": doc["filename"],
                "chunk_index": i,
            })


        vector_store.upsert_chunks(ids=ids, documents=texts, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"Ingested {doc['filename']}: {len(chunks)} chunks")

    print(f"\nDone. Total chunks in store: {vector_store.count()} (added {total_chunks} this run)")


if __name__ == "__main__":
    ingest_all()