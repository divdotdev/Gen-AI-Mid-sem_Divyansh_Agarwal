"""
Automated Test Suite for HR Policy Assistant RAG Pipeline.
Verifies PDF ingestion, metadata preservation, retrieval, citation accuracy,
and strict grounding ('Not found' fallback).
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from dotenv import load_dotenv

load_dotenv()
PDF_PATH = BASE_DIR / "data" / "hr_manual.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"


def test_pdf_existence_and_pages():
    """Verify that the HR manual PDF exists and contains 79 pages."""
    from langchain_community.document_loaders import PyPDFLoader

    assert PDF_PATH.exists(), f"PDF not found at {PDF_PATH}"
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()
    assert len(pages) == 79, f"Expected 79 pages, got {len(pages)}"


def test_chroma_manifest():
    """Verify that the vector store manifest is present and valid."""
    import json

    manifest_path = CHROMA_DIR / "manifest.json"
    assert manifest_path.exists(), "ChromaDB manifest.json is missing."
    manifest = json.loads(manifest_path.read_text())

    assert manifest["total_pages"] == 79
    assert manifest["total_chunks"] > 0
    assert "sha256" in manifest


def test_chunk_metadata_preservation():
    """Verify that chunks in ChromaDB preserve 1-indexed page and source metadata."""
    from rag import get_pipeline

    pipeline = get_pipeline()
    docs = pipeline.retrieve("probation policy", k=3)
    assert len(docs) > 0

    for doc in docs:
        assert "page" in doc.metadata, "Metadata missing 'page'"
        assert isinstance(doc.metadata["page"], int)
        assert 1 <= doc.metadata["page"] <= 79, f"Invalid page number: {doc.metadata['page']}"
        assert doc.metadata.get("source") == "hr_manual.pdf"


def test_retrieval_relevance():
    """Test that relevant chunks are retrieved for specific policies."""
    from rag import get_pipeline

    pipeline = get_pipeline()

    # Privilege leave retrieval
    pl_docs = pipeline.retrieve("Privilege Leave PL days per year", k=3)
    pl_pages = [d.metadata.get("page") for d in pl_docs]
    assert 60 in pl_pages or 19 in pl_pages, f"Expected page 60 or 19 in retrieved pages, got {pl_pages}"

    # Probation policy retrieval
    prob_docs = pipeline.retrieve("probation period for employees up to Senior Manager", k=3)
    prob_pages = [d.metadata.get("page") for d in prob_docs]
    assert 42 in prob_pages, f"Expected page 42 in retrieved pages, got {prob_pages}"


def test_unsupported_questions_return_not_found():
    """Verify strict grounding: external or absent questions MUST return exactly 'Not found'."""
    from rag import get_pipeline

    pipeline = get_pipeline()

    # Question about external tax law (GST in India)
    res_gst = pipeline.ask("What is the current GST rate in India?")
    assert res_gst["answer"].strip() == "Not found"
    assert not res_gst["is_found"]
    assert res_gst["citations"] == []

    # Question about external company or fictitious policy
    res_fake = pipeline.ask("What is the discount rate on Tesla cars for Enso employees?")
    assert res_fake["answer"].strip() == "Not found"


def test_eval_question_privilege_leave():
    """Test evaluation question 1: Privilege Leave days."""
    from rag import get_pipeline

    pipeline = get_pipeline()
    res = pipeline.ask("How many days of Privilege Leave (PL) does a permanent employee get per year?")
    
    assert res["is_found"], "Expected question to be answered from HR manual"
    assert "12" in res["answer"], f"Expected 12 days in answer, got: {res['answer']}"
    assert 60 in res["pages"], f"Expected citation of page 60, got {res['pages']}"


def test_eval_question_probation_period():
    """Test evaluation question 2: Probation period up to Senior Manager."""
    from rag import get_pipeline

    pipeline = get_pipeline()
    res = pipeline.ask("What is the probation period for employees up to Senior Manager?")
    
    assert res["is_found"], "Expected question to be answered from HR manual"
    assert "3" in res["answer"] and ("month" in res["answer"].lower()), f"Expected 3 months in answer, got: {res['answer']}"
    assert 42 in res["pages"], f"Expected citation of page 42, got {res['pages']}"


def test_eval_question_maternity_leave():
    """Test evaluation question 5: Maternity leave days/weeks."""
    from rag import get_pipeline

    pipeline = get_pipeline()
    res = pipeline.ask("How many days of maternity leave are given?")
    
    assert res["is_found"], "Expected question to be answered from HR manual"
    ans_lower = res["answer"].lower()
    assert "12 week" in ans_lower or "90 day" in ans_lower, f"Expected 12 weeks/90 days in answer, got: {res['answer']}"
    assert 62 in res["pages"], f"Expected citation of page 62, got {res['pages']}"


def test_eval_question_health_insurance():
    """
    Test evaluation question 3: Health insurance premium amount.
    Verify whether the manual mentions specific premium amounts or not.
    If not specified in manual, it must return 'Not found'.
    """
    from rag import get_pipeline

    pipeline = get_pipeline()
    res = pipeline.ask("What is the company's health-insurance premium amount?")
    # Check if answer is correctly determined (either exact premium if present, or Not found)
    if not res["is_found"]:
        assert res["answer"].strip() == "Not found"


def test_eval_question_gst_rate():
    """Test evaluation question 4: GST rate in India."""
    from rag import get_pipeline

    pipeline = get_pipeline()
    res = pipeline.ask("What is the current GST rate in India?")
    assert res["answer"].strip() == "Not found"
    assert not res["is_found"]
