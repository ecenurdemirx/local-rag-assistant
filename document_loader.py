# Document loading and text chunking utilities
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

import config


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _extract_pdf_text(file_path):
    """Extracts readable PDF page text in document order without OCR."""
    page_texts = []
    for page in PdfReader(file_path).pages:
        text = page.extract_text()
        if text and text.strip():
            page_texts.append(text.strip())
    return "\n\n".join(page_texts)


def load_documents(folder_path=config.DOCUMENTS_DIR):
    """Loads supported text and text-based PDF documents recursively."""
    documents = []
    dir_path = Path(folder_path)
    
    if not dir_path.exists() or not dir_path.is_dir():
        return documents

    file_paths = sorted(
        path
        for path in dir_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for file_path in file_paths:
        try:
            if file_path.suffix.lower() == ".pdf":
                text = _extract_pdf_text(file_path)
                if not text:
                    print(
                        f"Warning: No extractable text found in "
                        f"{file_path.relative_to(dir_path).as_posix()}; skipping."
                    )
                    continue
            else:
                text = file_path.read_text(encoding="utf-8")
            documents.append({
                "source": file_path.relative_to(dir_path).as_posix(),
                "text": text
            })
        except (OSError, UnicodeError, PdfReadError) as error:
            print(f"Warning: Failed to read {file_path.name}: {error}")
                
    return documents

def chunk_text(text, target_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """
    Chunks text targeting 500-1000 characters (default: 800) with overlap.
    Splits at whitespace boundaries to prevent word splitting.
    """
    if target_size <= 0 or overlap < 0 or overlap >= target_size:
        raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words):
            added = len(words[end]) + (1 if end > start else 0)
            if end > start and length + added > target_size:
                break
            length += added
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        overlap_chars = 0
        next_start = end
        while next_start > start and overlap_chars < overlap:
            next_start -= 1
            overlap_chars += len(words[next_start]) + 1
        start = next_start if next_start > start else end
    return chunks
