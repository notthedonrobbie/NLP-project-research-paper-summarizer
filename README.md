# PaperTrail — Intelligent Research Assistant

A working Python/Streamlit prototype for scientific PDF summarization, research-information extraction, semantic retrieval, and document-grounded question answering. It implements the project outline without requiring a paid API or training an LLM from scratch.

## Features

- Extracts selectable PDF text while retaining one-based PDF page numbers.
- Removes repeated headers/footers and common line-break artifacts.
- Builds overlapping, section-labelled chunks that never lose their source page.
- Produces an overview and six structured fields: problem, objectives, methodology, dataset/setup, findings, and limitations/future work.
- Compares TextRank and optional BART summarization.
- Supports BM25, TF-IDF, sentence-transformer dense retrieval, and reciprocal-rank-fusion hybrid retrieval.
- Uses FAISS automatically when installed; otherwise dense retrieval uses NumPy.
- Answers offline by quoting relevant evidence, or optionally uses OpenAI with a strict evidence-only prompt.
- Shows source identifiers, page numbers, retrieval scores, and all retrieved passages.
- Computes ROUGE-1/2/L and Precision@k/Recall@k from independent human annotations.

> Offline answers are intentionally extractive: they quote document sentences rather than pretending to be a locally generated LLM response. Important outputs should always be checked against the original paper.

## Project structure

```text
research-assistant/
├── app.py                         # Streamlit application
├── research_assistant/
│   ├── pdf_processor.py           # PDF extraction and cleanup
│   ├── chunking.py                # Page-aware overlapping chunks
│   ├── summarization.py           # TextRank and optional BART
│   ├── retrieval.py               # BM25, TF-IDF, dense, hybrid, FAISS
│   ├── rag.py                     # Evidence-only answer generation
│   ├── pipeline.py                # High-level Python API
│   ├── evaluation.py              # ROUGE and retrieval metrics
│   └── models.py                  # Typed data structures
├── scripts/evaluate.py            # Reproducible experiment runner
├── data/evaluation.example.json   # Annotation schema
├── notebooks/                     # Google Colab demonstration
├── report/                        # Technical report template and rubric
├── tests/                         # Offline unit and integration tests
└── results/                       # Generated experiment outputs
```

## Quick start

Requires Python 3.10 or newer.

```bash
cd research-assistant
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a digital English research PDF, and select **Process paper**. The default configuration is fully offline after installation.

## Optional capabilities

Install transformer, embedding, and OpenAI integrations:

```bash
python -m pip install -r requirements-ml.txt
```

The first dense-retrieval or BART run downloads the selected model from Hugging Face and may require substantial memory. To accelerate dense retrieval with FAISS:

```bash
python -m pip install faiss-cpu
```

For fluent LLM answers, set `OPENAI_API_KEY` in the environment, then choose **OpenAI RAG** in the sidebar. Only the question and retrieved excerpts—not the entire PDF—are sent to OpenAI in this mode. Do not enable external generation for confidential documents unless that transmission is authorized.

## Python API

```python
from research_assistant import ResearchAssistant

assistant = ResearchAssistant.from_pdf("paper.pdf", retrieval_method="bm25")
summary = assistant.summarize(method="textrank")
print(summary.overview)
print(summary.sections)

answer = assistant.ask("What dataset was used?", top_k=5)
print(answer.text)
for citation in answer.citations:
    print(citation.chunk.page_numbers, citation.chunk.text)
```

Use `retrieval_method="dense"` or `"hybrid"` after installing the ML dependencies.

## Architecture

```text
PDF → page extraction → cleanup → section/page-aware chunks
                                  ├── TextRank / BART → structured research brief
                                  └── retriever index
Question → BM25 / TF-IDF / dense / hybrid → top-k evidence
                                            ├── offline evidence quotation
                                            └── optional evidence-only LLM
Evidence → source IDs + PDF page numbers → inspection interface
```

The pipeline deliberately separates retrieval from generation. This makes evidence inspectable and allows retrieval configurations to be evaluated independently.

## Experiments

### 1. Prepare an independent evaluation set

Copy `data/evaluation.example.json` and create a reference summary plus 5–10 questions per paper. Write references **without looking at system output**. For retrieval metrics, list the chunk IDs that independently judged relevant to each question. Chunk IDs depend on chunking settings, so annotate after fixing `chunk_size` and `overlap`; the output records the configuration and ID order.

### 2. Run comparisons

```bash
python scripts/evaluate.py paper.pdf data/my_annotations.json \
  --methods bm25 tfidf --summarizers textrank --k 5 \
  --output results/baseline.json
```

With optional ML dependencies:

```bash
python scripts/evaluate.py paper.pdf data/my_annotations.json \
  --methods bm25 dense hybrid --summarizers textrank abstractive \
  --output results/comparison.json
```

The JSON report contains measured ROUGE and retrieval scores, generated summaries, retrieved IDs, Q&A predictions, and blank human-rating fields. No scores are pre-populated or assumed. ROUGE measures overlap with a reference, not factual correctness. Use the rubric in `report/evaluation_rubric.md` for correctness, relevance, evidence support, and citation accuracy.

### 3. Recommended research comparison

Use approximately 20–50 appropriately licensed papers from one focused area. Keep chunking and `k` fixed during the primary comparison, then run a separate sensitivity analysis. Report mean scores and per-paper variation, inspect failure cases, and include insufficient-evidence questions. Do not use the generated answers to create their own references.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Tests cover PDF extraction, repeated-header removal, chunk boundaries, source pages, lexical retrieval, hybrid fusion, grounded answers, insufficient-evidence behavior, structured summaries, evaluation metrics, and an end-to-end offline PDF workflow.

## Google Colab

Open `notebooks/research_assistant_colab.ipynb` in Colab. Upload or clone this project first, install `requirements.txt`, and run the cells in order. Dense/BART models may require a GPU runtime and additional download time.

## Limitations and responsible use

- Scanned PDFs require OCR, which is not included.
- Multi-column layouts, equations, tables, figures, and unusual headings can reduce extraction quality.
- Chunk overlap does not guarantee that all cross-page context is retained.
- Dense embeddings improve meaning-based matching but do not guarantee relevant retrieval.
- Extractive answers may be less fluent; LLM answers can still hallucinate despite grounding instructions.
- Automatically identified limitations may be incomplete, especially when authors do not state them explicitly.
- Use authorized/open-access documents, avoid unnecessary document retention, and consult the original paper before relying on generated output.

## Future work

Multi-paper comparison, OCR, table/figure extraction, cross-page chunks, hierarchical long-document summaries, literature-review matrices, multi-annotator evaluation, and calibrated answer-confidence/faithfulness metrics.
