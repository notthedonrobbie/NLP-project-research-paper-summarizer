"""Streamlit interface for the scientific research assistant."""

from __future__ import annotations

import hashlib
import os

import streamlit as st

from research_assistant import ResearchAssistant
from research_assistant.models import RetrievedChunk, Summary
from research_assistant.rag import OpenAIAnswerGenerator

st.set_page_config(page_title="PaperTrail Research Assistant", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem;}
      [data-testid="stMetric"] {background: #f6f8fb; border: 1px solid #e6eaf0;
        border-radius: 12px; padding: 14px 18px;}
      .source-note {color: #526070; font-size: .9rem;}
      .field-card {border-left: 4px solid #376f65; background: #f6f9f8;
        padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def source_label(item: RetrievedChunk, index: int) -> str:
    pages = ", ".join(str(page) for page in item.chunk.page_numbers)
    return f"S{index} · Page {pages} · {item.chunk.section} · score {item.score:.3f}"


def summary_markdown(document_title: str, summary: Summary) -> str:
    lines = [f"# {document_title}", "", "## Overview", "", summary.overview, ""]
    for field, text in summary.sections.items():
        lines.extend([f"## {field}", "", text, ""])
    lines.extend(["---", f"Generated with {summary.method} summarization. Verify against the original paper."])
    return "\n".join(lines)


with st.sidebar:
    st.title("PaperTrail")
    st.caption("Scientific papers, made navigable.")
    st.divider()
    st.subheader("Processing")
    retrieval_label = st.selectbox(
        "Retrieval method", ["BM25 (offline)", "TF-IDF (offline)", "Dense embeddings", "Hybrid BM25 + dense"]
    )
    summary_label = st.selectbox("Summarization", ["TextRank (offline)", "BART (optional download)"])
    top_k = st.slider("Evidence passages", min_value=2, max_value=10, value=5)
    chunk_size = st.slider("Chunk size (characters)", 500, 1800, 1000, 100)
    st.subheader("Answer generation")
    generator_label = st.radio(
        "Generator",
        ["Extractive evidence (offline)", "OpenAI RAG (optional)"],
        help="Offline mode quotes evidence sentences. OpenAI mode writes a fluent answer using only retrieved passages.",
    )
    openai_model = st.text_input("OpenAI model", value="gpt-4o-mini", disabled=not generator_label.startswith("OpenAI"))
    st.caption("Set OPENAI_API_KEY in your environment to use OpenAI RAG. Uploaded text is sent to OpenAI only in that mode.")
    st.divider()
    st.caption("Digital English PDFs only · Scanned papers require OCR · Always verify important claims against the paper.")

st.title("Understand a paper. Keep the evidence.")
st.write("Upload a scientific PDF to create a structured research summary and ask questions with page-level source passages.")

uploaded = st.file_uploader("Research paper (PDF)", type=["pdf"])
process = st.button("Process paper", type="primary", disabled=uploaded is None, use_container_width=True)

if process and uploaded is not None:
    data = uploaded.getvalue()
    fingerprint = hashlib.sha256(data).hexdigest()
    retrieval_method = {
        "BM25 (offline)": "bm25",
        "TF-IDF (offline)": "tfidf",
        "Dense embeddings": "dense",
        "Hybrid BM25 + dense": "hybrid",
    }[retrieval_label]
    summary_method = "abstractive" if summary_label.startswith("BART") else "textrank"
    try:
        with st.spinner("Extracting the paper and building its searchable index…"):
            assistant = ResearchAssistant.from_pdf(
                data,
                title=uploaded.name.rsplit(".", 1)[0],
                retrieval_method=retrieval_method,
                chunk_size=chunk_size,
                overlap=min(150, chunk_size // 3),
            )
        with st.spinner("Generating the structured summary…"):
            summary = assistant.summarize(summary_method)
        st.session_state.update(
            assistant=assistant,
            summary=summary,
            fingerprint=fingerprint,
            answer=None,
            last_question="",
        )
        st.success("Paper processed successfully.")
    except Exception as exc:
        st.error(str(exc))

assistant = st.session_state.get("assistant")
if assistant is None:
    st.info("Choose a PDF above to begin. No paper is included with this repository; use your own authorized document.")
    col1, col2, col3 = st.columns(3)
    col1.markdown("### 1. Structured summary\nProblem, objectives, methods, datasets, findings, and limitations.")
    col2.markdown("### 2. Grounded answers\nSemantic or lexical retrieval finds evidence before an answer is produced.")
    col3.markdown("### 3. Inspectable sources\nEvery answer includes passages and PDF page numbers for verification.")
else:
    summary: Summary = st.session_state.summary
    st.subheader(assistant.document.title)
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Pages", assistant.document.page_count)
    metric2.metric("Searchable chunks", len(assistant.chunks))
    metric3.metric("Retriever", assistant.retrieval_method.upper())
    metric4.metric("Summary", summary.method.title())

    summary_tab, qa_tab, document_tab = st.tabs(["Research brief", "Ask the paper", "Extracted document"])

    with summary_tab:
        st.markdown("### Overview")
        st.write(summary.overview)
        st.download_button(
            "Download research brief",
            summary_markdown(assistant.document.title, summary),
            file_name="research-brief.md",
            mime="text/markdown",
        )
        st.markdown("### Research information")
        for field, text in summary.sections.items():
            st.markdown(f"**{field}**")
            st.write(text)

    with qa_tab:
        st.markdown("### Ask a document-grounded question")
        suggestions = [
            "What problem does this paper address?",
            "Which methodology do the authors propose?",
            "What dataset was used?",
            "What were the main experimental results?",
            "What limitations did the authors identify?",
        ]
        suggestion_columns = st.columns(len(suggestions))
        for column, suggestion in zip(suggestion_columns, suggestions):
            if column.button(suggestion.split("?")[0], key=suggestion, help=suggestion):
                st.session_state.question_input = suggestion
        with st.form("question-form"):
            question = st.text_input(
                "Question",
                key="question_input",
                placeholder="e.g. Which evaluation metric improved the most?",
            )
            submitted = st.form_submit_button("Retrieve evidence and answer", type="primary")
        if submitted:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    generator = None
                    if generator_label.startswith("OpenAI"):
                        if not os.getenv("OPENAI_API_KEY"):
                            raise RuntimeError("Set OPENAI_API_KEY, or switch to offline extractive generation.")
                        generator = OpenAIAnswerGenerator(openai_model)
                    with st.spinner("Retrieving supporting passages…"):
                        st.session_state.answer = assistant.ask(question, top_k=top_k, generator=generator)
                        st.session_state.last_question = question
                except Exception as exc:
                    st.error(str(exc))

        answer = st.session_state.get("answer")
        if answer is not None:
            st.markdown(f"**Question:** {st.session_state.last_question}")
            st.markdown("### Answer")
            st.info(answer.text) if answer.insufficient_evidence else st.write(answer.text)
            st.caption(f"Generation mode: {answer.generation_mode}")
            if answer.citations:
                st.markdown("### Cited passages")
                for item in answer.citations:
                    evidence_index = answer.evidence.index(item) + 1
                    with st.expander(source_label(item, evidence_index), expanded=True):
                        st.write(item.chunk.text)
            with st.expander("All retrieved evidence"):
                for index, item in enumerate(answer.evidence, start=1):
                    st.markdown(f"**{source_label(item, index)}**")
                    st.write(item.chunk.text)
                    st.divider()

    with document_tab:
        st.markdown("### Verify against the extracted text")
        st.caption("Extraction errors can occur with multi-column layouts, equations, tables, and figures.")
        page_number = st.selectbox(
            "PDF page", options=[page.page_number for page in assistant.document.pages]
        )
        page = assistant.document.pages[page_number - 1]
        st.text_area("Page text", page.text, height=500, label_visibility="collapsed")
