"""
Document Ingestion Script for HR Policy Manual.
Extracts text from PDF, preserves metadata (page numbers, section titles),
generates embeddings, and stores in persistent ChromaDB.
Avoids unnecessary re-indexing by hashing source PDF and configuration.
"""

import os
import sys
import re
import json
import time
import hashlib
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = Path(os.getenv("PDF_PATH", BASE_DIR / "data" / "hr_manual.pdf"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")


def get_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def get_embeddings_model():
    """Initialize Gemini embeddings model with API key."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def extract_section_hint(text: str) -> str:
    """Extract section or policy title hint from the first few lines of page text."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines[:5]:
        if re.search(r"policy|procedure|annexure|section|conduct|rules|leave|probation|on boarding|travel", line, re.IGNORECASE):
            return line[:80]
    return "HR Policy Manual"


def clean_pdf_text(text: str) -> str:
    """Normalize whitespace and fix hyphenated split words across lines."""
    # Fix hyphenated words at linebreaks: e.g. "compa-\nny" -> "company"
    cleaned = re.sub(r"(?<=[A-Za-z])\s*-\s*\n\s*(?=[A-Za-z])", "", text)
    # Normalize multiple newlines and spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def is_already_indexed(pdf_file: Path, chroma_dir: Path) -> bool:
    """Check if the PDF has already been indexed with matching hash and model."""
    manifest_file = chroma_dir / "manifest.json"
    if not manifest_file.exists():
        return False
    try:
        data = json.loads(manifest_file.read_text())
        current_hash = get_file_sha256(pdf_file)
        if data.get("sha256") == current_hash and data.get("embedding_model") == EMBEDDING_MODEL:
            return True
    except Exception:
        return False
    return False


def ingest_document(pdf_path: Path = PDF_PATH, chroma_dir: Path = CHROMA_DIR, force: bool = False) -> dict:
    """
    Ingests PDF document into ChromaDB.
    Returns summary statistics dictionary.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}. Please place hr_manual.pdf in data/ folder.")

    if not force and is_already_indexed(pdf_path, chroma_dir):
        print(f"[INFO] ChromaDB index is already up-to-date for {pdf_path.name}. Skipping ingestion.")
        manifest = json.loads((chroma_dir / "manifest.json").read_text())
        return manifest

    print(f"[INFO] Loading PDF from: {pdf_path}...")
    loader = PyPDFLoader(str(pdf_path))
    raw_pages = loader.load()

    if not raw_pages:
        raise ValueError(f"No pages could be extracted from {pdf_path}.")

    print(f"[INFO] Successfully loaded {len(raw_pages)} pages.")

    # Process pages, clean text, and record 1-indexed page numbers
    processed_docs = []
    for idx, page in enumerate(raw_pages):
        page_num = idx + 1  # 1-indexed human page number
        raw_text = page.page_content
        cleaned_text = clean_pdf_text(raw_text)
        if not cleaned_text:
            continue  # Gracefully skip empty/image-only pages

        section_hint = extract_section_hint(cleaned_text)
        page.page_content = cleaned_text
        page.metadata["source"] = pdf_path.name
        page.metadata["page"] = page_num
        page.metadata["section"] = section_hint
        processed_docs.append(page)

    print(f"[INFO] {len(processed_docs)} non-empty pages prepared for chunking.")

    # Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "▶", "Ԃ", "•", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(processed_docs)
    print(f"[INFO] Document split into {len(chunks)} chunks.")

    # Ensure each chunk keeps its page and section metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    # Build vector store in persistent directory
    print(f"[INFO] Generating embeddings and building ChromaDB index...")
    embeddings = get_embeddings_model()
    
    # Check if partial chroma_db already has some chunks
    staging_dir = chroma_dir
    store = Chroma(
        collection_name="hr_policy",
        embedding_function=embeddings,
        persist_directory=str(staging_dir),
    )

    # Check already indexed IDs to allow resuming without re-embedding
    try:
        existing_data = store.get()
        existing_ids = set(existing_data.get("ids", []))
    except Exception:
        existing_ids = set()

    if force:
        try:
            store.delete_collection()
            store = Chroma(
                collection_name="hr_policy",
                embedding_function=embeddings,
                persist_directory=str(staging_dir),
            )
            existing_ids = set()
        except Exception:
            pass

    batch_size = 35
    total_chunks = len(chunks)
    for start_idx in range(0, total_chunks, batch_size):
        batch = chunks[start_idx : start_idx + batch_size]
        batch_ids = [f"chunk_{i}" for i in range(start_idx, start_idx + len(batch))]
        
        # Check if already indexed
        if all(bid in existing_ids for bid in batch_ids):
            print(f"[RESUME] Batch {start_idx}-{start_idx+len(batch)} already in vector store. Skipping.")
            continue

        for attempt in range(6):
            try:
                store.add_documents(documents=batch, ids=batch_ids)
                for bid in batch_ids:
                    existing_ids.add(bid)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Quota exceeded" in err_str or "ResourceExhausted" in err_str:
                    print(f"[RATE-LIMIT] Quota 429 reached. Waiting 55 seconds for quota reset (attempt {attempt+1}/6)...", flush=True)
                    time.sleep(55)
                else:
                    if attempt == 5:
                        raise RuntimeError(f"Failed to index batch {start_idx}-{start_idx+len(batch)}: {e}")
                    print(f"[WARN] Ingestion issue, retrying in 15s... ({e})", flush=True)
                    time.sleep(15)

        indexed_count = min(start_idx + batch_size, total_chunks)
        print(f"[PROGRESS] Indexed {indexed_count}/{total_chunks} chunks...", flush=True)
        # 5-second pacing between batches to reduce 429 occurrences
        time.sleep(5)

    # Write manifest
    manifest = {
        "source_file": pdf_path.name,
        "sha256": get_file_sha256(pdf_path),
        "total_pages": len(raw_pages),
        "indexed_pages": len(processed_docs),
        "total_chunks": total_chunks,
        "embedding_model": EMBEDDING_MODEL,
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (chroma_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[SUCCESS] Ingestion completed. Vector database saved to {chroma_dir}")
    return manifest


if __name__ == "__main__":
    force_rebuild = "--force" in sys.argv or "--rebuild" in sys.argv
    try:
        stats = ingest_document(force=force_rebuild)
        print(f"[RESULT] Stats: {json.dumps(stats, indent=2)}")
    except Exception as exc:
        print(f"[ERROR] Ingestion failed: {exc}")
        sys.exit(1)
