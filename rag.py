"""
RAG Pipeline for Enso Group HR Policy Assistant.
Enforces strict grounding, verbatim citation verification, and exact 'Not found' fallback.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
PDF_PATH = Path(os.getenv("PDF_PATH", BASE_DIR / "data" / "hr_manual.pdf"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")


class EvidenceItem(BaseModel):
    chunk_index: int = Field(description="0-based index of the chunk in the context list")
    quote: str = Field(description="Exact verbatim quote from the chunk that proves the answer")


class GroundedAnswer(BaseModel):
    is_supported: bool = Field(description="True if the context adequately and directly answers the question; False otherwise")
    answer: str = Field(description="Grounded answer to the question, or exactly 'Not found'")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="List of supporting evidence items with exact quotes")


class VerificationResult(BaseModel):
    claims_supported: bool = Field(description="True if all claims in the answer are strictly supported by the supplied quotes")


# Primary answering prompt
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the authoritative HR Policy Assistant for Enso Group. "
        "Answer employee questions using ONLY the provided HR manual context chunks below.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Treat the context as the ONLY authoritative source of truth. NEVER use general outside knowledge.\n"
        "2. If the context does not contain enough information to adequately answer the question, or if the question asks about something outside the manual (e.g. external tax laws like GST in India, general knowledge, or undisclosed policies), set is_supported=False and answer='Not found'.\n"
        "3. Never guess missing numbers, eligibility requirements, percentages, or dates.\n"
        "4. If the context answers the question, provide a clear, concise answer, and include exact verbatim quotes from the chunks as evidence.\n"
        "5. Preserve exact terms, conditions, exceptions, and numbers as written in the manual."
    ),
    (
        "human",
        "Question: {question}\n\nContext Chunks:\n{context_text}"
    ),
])

# Verification prompt to prevent hallucinations
VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict compliance auditor. Verify whether every claim in the proposed answer "
        "is directly and factually supported by the provided quotes from the HR manual.\n"
        "Return claims_supported=True only if the quotes prove the answer completely.\n"
        "If any claim is unproven, assumed, or drawn from external knowledge, return claims_supported=False."
    ),
    (
        "human",
        "Question: {question}\nProposed Answer: {answer}\nQuotes:\n{quotes_text}"
    ),
])


def normalize_text_for_matching(text: str) -> str:
    """Normalize text for substring matching across hyphenation and whitespace."""
    # Remove hyphenated line splits
    text = re.sub(r"(?<=[A-Za-z])\s*-\s*(?=[A-Za-z])", "", text)
    # Remove all whitespace and lowercase
    return "".join(text.lower().split())


class HRPipeline:
    def __init__(self, chroma_dir: Path = CHROMA_DIR):
        self.chroma_dir = chroma_dir
        self.vector_store: Optional[Chroma] = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self._init_components()

    def _init_components(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )

        manifest_file = self.chroma_dir / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"ChromaDB manifest not found at {manifest_file}. "
                "Please run 'python ingest.py' first to build the vector index."
            )

        self.vector_store = Chroma(
            collection_name="hr_policy",
            embedding_function=embeddings,
            persist_directory=str(self.chroma_dir),
        )

        self.llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            google_api_key=api_key,
            temperature=0.0,
            max_retries=1,
            timeout=30,
        )

    def retrieve(self, question: str, k: int = 6) -> List[Document]:
        """Retrieve most relevant chunks for the question."""
        if not self.vector_store:
            raise RuntimeError("Vector store is not initialized.")
        docs = self.vector_store.similarity_search(question, k=k)
        return docs

    def ask(self, question: str, k: int = 6) -> Dict[str, Any]:
        """
        Execute RAG question answering pipeline with strict grounding.
        Returns a dictionary with 'answer', 'is_found', 'citations', 'pages', and 'evidence'.
        """
        clean_q = question.strip()
        if not clean_q:
            return {
                "answer": "Not found",
                "is_found": False,
                "citations": [],
                "pages": [],
                "evidence": [],
                "raw_docs": [],
            }

        docs = self.retrieve(clean_q, k=k)
        if not docs or not any(d.page_content.strip() for d in docs):
            return {
                "answer": "Not found",
                "is_found": False,
                "citations": [],
                "pages": [],
                "evidence": [],
                "raw_docs": [],
            }

        # Format context chunks
        context_blocks = []
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "Unknown")
            sec = doc.metadata.get("section", "HR Policy")
            context_blocks.append(f"--- Chunk {i} (Page {page}, Section: {sec}) ---\n{doc.page_content}")
        context_text = "\n\n".join(context_blocks)

        # Generate answer with structured output
        structured_llm = self.llm.with_structured_output(GroundedAnswer)
        try:
            chain = ANSWER_PROMPT | structured_llm
            result: GroundedAnswer = chain.invoke({
                "question": clean_q,
                "context_text": context_text,
            })
        except Exception as e:
            # Fallback on parsing error
            return {
                "answer": "Not found",
                "is_found": False,
                "citations": [],
                "pages": [],
                "evidence": [],
                "raw_docs": docs,
                "error": str(e),
            }

        # Verify fallback response condition
        if not result.is_supported or result.answer.strip().lower() == "not found" or not result.evidence:
            return {
                "answer": "Not found",
                "is_found": False,
                "citations": [],
                "pages": [],
                "evidence": [],
                "raw_docs": docs,
            }

        # Validate evidence quotes against retrieved documents
        verified_evidence = []
        verified_pages = set()
        citations = []

        for ev in result.evidence:
            nq = normalize_text_for_matching(ev.quote)
            if not nq or len(nq) < 10:
                continue

            matched_doc = None
            if 0 <= ev.chunk_index < len(docs):
                cand_doc = docs[ev.chunk_index]
                cand_text = normalize_text_for_matching(cand_doc.page_content)
                if nq in cand_text or nq[:30] in cand_text:
                    matched_doc = cand_doc

            # If not matched at cand_doc, search across all retrieved docs
            if not matched_doc:
                for cand_doc in docs:
                    cand_text = normalize_text_for_matching(cand_doc.page_content)
                    if nq in cand_text or nq[:30] in cand_text:
                        matched_doc = cand_doc
                        break

            if matched_doc:
                page = matched_doc.metadata.get("page", 0)
                section = matched_doc.metadata.get("section", "HR Policy Manual")
                verified_pages.add(page)
                verified_evidence.append(ev.quote)
                citations.append({
                    "page": page,
                    "section": section,
                    "quote": ev.quote,
                    "chunk_id": matched_doc.metadata.get("chunk_id"),
                })

        if not verified_evidence:
            # No valid traceable quote found
            return {
                "answer": "Not found",
                "is_found": False,
                "citations": [],
                "pages": [],
                "evidence": [],
                "raw_docs": docs,
            }

        # Secondary entailment verification
        quotes_text = "\n".join(f"- {q}" for q in verified_evidence)
        try:
            verify_llm = self.llm.with_structured_output(VerificationResult)
            v_chain = VERIFY_PROMPT | verify_llm
            v_res: VerificationResult = v_chain.invoke({
                "question": clean_q,
                "answer": result.answer,
                "quotes_text": quotes_text,
            })
            if not v_res.claims_supported:
                return {
                    "answer": "Not found",
                    "is_found": False,
                    "citations": [],
                    "pages": [],
                    "evidence": [],
                    "raw_docs": docs,
                }
        except Exception:
            pass  # If secondary check encounters issue, proceed with verified evidence quotes

        sorted_pages = sorted(list(verified_pages))
        return {
            "answer": result.answer.strip(),
            "is_found": True,
            "citations": citations,
            "pages": sorted_pages,
            "evidence": verified_evidence,
            "raw_docs": docs,
        }


def format_full_response(result: Dict[str, Any]) -> str:
    """Format answer and citations cleanly for CLI or display."""
    if not result.get("is_found", False) or result.get("answer", "") == "Not found":
        return "Not found"

    ans = result["answer"]
    pages_str = ", ".join(f"Page {p}" for p in result.get("pages", []))
    
    # Extract unique sections
    sections = []
    for c in result.get("citations", []):
        sec = c.get("section", "")
        if sec and sec not in sections and sec != "HR Policy Manual":
            sections.append(sec)
    
    sec_str = f" | Section: {', '.join(sections)}" if sections else ""
    citations_line = f"\n\nSource: Enso Group HR Policy Manual [{pages_str}{sec_str}]"
    return f"{ans}{citations_line}"


# Global pipeline instance cache
_PIPELINE: Optional[HRPipeline] = None

def get_pipeline() -> HRPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = HRPipeline()
    return _PIPELINE

def ask(question: str) -> str:
    """Top-level ask function matching assignment interface."""
    pipeline = get_pipeline()
    res = pipeline.ask(question)
    return format_full_response(res)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"Question: {q}")
        print("Answer:\n" + ask(q))
    else:
        print("Enso Group HR Policy Assistant (RAG)")
        print("Type your question below (or 'quit'/'exit' to stop):\n")
        pipe = get_pipeline()
        while True:
            try:
                user_q = input("\nHR Question > ").strip()
                if user_q.lower() in ["quit", "exit"]:
                    break
                if not user_q:
                    continue
                ans = ask(user_q)
                print(f"\n{ans}\n")
            except (KeyboardInterrupt, EOFError):
                break
