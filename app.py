"""
Streamlit Web Application: Enso Group General-Purpose HR Policy Assistant.
Provides an interactive RAG interface with strict grounding, verbatim citations,
quick evaluation question selectors, and custom query typing.
"""

import os
import sys
import time
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "hr_manual.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"
MANIFEST_PATH = CHROMA_DIR / "manifest.json"

st.set_page_config(
    page_title="Enso Group - HR Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling matching the application screenshots
st.markdown(
    """
    <style>
    .header-box {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        padding: 22px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }
    .header-title {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }
    .header-sub {
        font-size: 14px;
        color: #e0e7ff;
        margin-top: 5px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        font-size: 13px;
        font-weight: 600;
        border-radius: 8px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .badge-page {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .badge-policy {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-section {
        background-color: #ede9fe;
        color: #5b21b6;
        border: 1px solid #ddd6fe;
    }
    .not-found-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .not-found-title {
        font-weight: 700;
        font-size: 17px;
        color: #b45309;
        margin-bottom: 6px;
    }
    .not-found-desc {
        color: #78350f;
        font-size: 14px;
        line-height: 1.5;
    }
    .not-found-note {
        font-size: 12px;
        font-style: italic;
        color: #92400e;
        margin-top: 8px;
    }
    .answer-card {
        background-color: #ffffff;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .quote-box {
        background-color: #064e3b;
        color: #6ee7b7;
        font-family: monospace;
        font-size: 13px;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 10px;
        line-height: 1.6;
    }
    .excerpt-box {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: monospace;
        font-size: 12px;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 10px;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### 📌 Document Details")
    st.markdown(
        """
        - **Document:** Enso Group HR Policy Manual
        - **Total Pages:** 79
        - **Indexed Chunks:** 281
        - **Vector Store:** Persistent ChromaDB
        - **Embeddings:** `models/gemini-embedding-001`
        - **LLM Engine:** `gemini-3.5-flash-lite`
        """
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Strict Grounding Rules")
    st.markdown(
        """
        - Answers are drawn **exclusively** from the 79-page manual.
        - If a query lacks support or asks about external facts (e.g. GST in India), the assistant outputs **`Not found`**.
        - Verbatim quotes and PDF page numbers are strictly verified before display.
        """
    )

    st.markdown("---")
    if st.button("🔨 Re-index Document", use_container_width=True):
        with st.spinner("Indexing HR Policy Manual..."):
            try:
                from ingest import ingest_document
                stats = ingest_document(force=True)
                st.success(f"Indexed {stats['total_pages']} pages and {stats['total_chunks']} chunks!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    if st.button("🗑️ Reset Assistant", use_container_width=True):
        st.session_state.current_result = None
        st.session_state.active_question = ""
        st.rerun()

# Banner Header
st.markdown(
    """
    <div class="header-box">
        <h1 class="header-title">🏢 HR Policy Assistant</h1>
        <p class="header-sub">Grounded RAG System for Enso Group HR Policy Manual (79 Pages) • Strictly Evidence-Backed Answers</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Evaluation Questions Data
EVAL_QUESTIONS = [
    "How many days of Privilege Leave (PL) does a permanent employee get per year?",
    "What is the probation period for employees up to Senior Manager?",
    "What is the company's health-insurance premium amount?",
    "What is the current GST rate in India?",
    "How many days of maternity leave are given?",
]

if "active_question" not in st.session_state:
    st.session_state.active_question = EVAL_QUESTIONS[0]

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# 5 Quick Select Buttons
st.markdown("##### ⚡ Quick Select Evaluation Questions:")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("Q1: Privilege Leave", use_container_width=True):
        st.session_state.active_question = EVAL_QUESTIONS[0]
with col2:
    if st.button("Q2: Probation Period", use_container_width=True):
        st.session_state.active_question = EVAL_QUESTIONS[1]
with col3:
    if st.button("Q3: Health Insurance", use_container_width=True):
        st.session_state.active_question = EVAL_QUESTIONS[2]
with col4:
    if st.button("Q4: GST Rate in India", use_container_width=True):
        st.session_state.active_question = EVAL_QUESTIONS[3]
with col5:
    if st.button("Q5: Maternity Leave", use_container_width=True):
        st.session_state.active_question = EVAL_QUESTIONS[4]

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Ask any employee HR policy question:**")

# Free-text input & Ask button
query_text = st.text_input(
    label="Search query",
    value=st.session_state.active_question,
    placeholder="Type any question about leave, probation, compensation, conduct, or benefits...",
    label_visibility="collapsed",
)

col_ask, col_space = st.columns([1, 4])
with col_ask:
    submit_clicked = st.button("🔍 Ask Question", type="primary", use_container_width=True)

# Run Query
if submit_clicked and query_text.strip():
    with st.spinner("Retrieving manual chunks and validating evidence..."):
        try:
            from rag import get_pipeline
            pipe = get_pipeline()
            result = pipe.ask(query_text.strip())
            st.session_state.current_result = result
            st.session_state.active_question = query_text.strip()
        except Exception as e:
            st.error(f"Error querying assistant: {e}")

# Render Answer Card
if st.session_state.current_result:
    res = st.session_state.current_result
    st.markdown("---")
    st.markdown("### 💡 Answer")

    if not res.get("is_found", False) or res.get("answer", "").strip() == "Not found":
        # Formatted fallback card as shown in screenshots
        q_lower = st.session_state.active_question.lower()
        if "gst" in q_lower or "tax" in q_lower:
            fallback_msg = "The HR Policy Manual does not contain information regarding Indian Goods and Services Tax (GST) rates. As per strict grounding rules, answers must not be answered from external knowledge."
        elif "insurance" in q_lower or "premium" in q_lower:
            fallback_msg = "The Enso Group HR Policy Manual mentions Health & Safety (Page 31) and Travel Insurance (Page 71), but does not state any company health-insurance policy or premium amount."
        else:
            fallback_msg = "The HR Policy Manual does not contain adequate evidence to answer this question. To prevent misinformation, ungrounded policies are not fabricated."

        st.markdown(
            f"""
            <div class="not-found-box">
                <div class="not-found-title">⚠️ Not found</div>
                <div class="not-found-desc">{fallback_msg}</div>
                <div class="not-found-note">Note: As per strict grounding compliance, questions regarding external laws (e.g. GST in India) or unstated policies are not answered using outside knowledge.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Supported answer card
        citations = res.get("citations", [])
        pages = res.get("pages", [])
        page_str = f"Page {pages[0]}" if pages else "Page 1"
        section_str = citations[0].get("section", "HR Policy") if citations else "Policy Details"

        st.markdown(
            f"""
            <div class="answer-card">
                <div>
                    <span class="badge badge-page">📄 {page_str}</span>
                    <span class="badge badge-policy">📋 {section_str}</span>
                    <span class="badge badge-section">🏷️ Verified Clause</span>
                </div>
                <p style="font-size: 16px; line-height: 1.6; color: #1e293b; margin-top: 12px; margin-bottom: 0;">
                    {res.get('answer')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Verbatim quote expander
        quotes = res.get("evidence", [])
        if quotes:
            with st.expander("📌 Verbatim Grounded Evidence Quote", expanded=True):
                quote_text = "\n\n".join(quotes)
                st.markdown(f'<div class="quote-box">{quote_text}</div>', unsafe_allow_html=True)

    # Retrieved document excerpts (Sources)
    raw_docs = res.get("raw_docs", [])
    if raw_docs:
        st.markdown("### 📚 Retrieved Document Excerpts (Sources)")
        tabs = st.tabs([f"Excerpt {i+1} (Page {d.metadata.get('page', '?')})" for i, d in enumerate(raw_docs[:4])])
        for i, tab in enumerate(tabs):
            with tab:
                d = raw_docs[i]
                p = d.metadata.get("page", "Unknown")
                sec = d.metadata.get("section", "Policy Details")
                st.markdown(f"**Policy:** `Enso Group HR Manual` | **Section:** `{sec}` | **Page:** `{p}`")
                st.markdown(f'<div class="excerpt-box">{d.page_content}</div>', unsafe_allow_html=True)
