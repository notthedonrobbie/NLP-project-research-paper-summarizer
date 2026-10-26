# PaperTrail — Intelligent Research Assistant

PaperTrail is a working Python/Streamlit prototype for scientific PDF summarization, research-information extraction, lexical or semantic retrieval, and document-grounded question answering. Its default workflow is fully offline after installation and does not require a paid API or model training.

## Current report: 595 full-text research papers

The main technical report evaluates **595 successfully extracted full papers** linked from the 600-record `data/ai_research_papers.csv`. Four HTTP 404 downloads and one PDF parser-limit failure were excluded with the user's approval. The accepted corpus contains **12,836 pages and 6,607,965 extracted words**. Bulk PDFs and extracted full text remain local and are excluded from Git.

Known-item title-to-full-text retrieval gave TF-IDF **0.949 MRR and 0.985 Hit@10**, compared with **0.926 and 0.971** for BM25. Six-sentence TextRank summaries achieved macro **ROUGE-1/2/L = 0.345/0.106/0.179** against author abstracts; LEAD achieved **0.332/0.101/0.187**. These are known-item and abstract-alignment results, not natural-query relevance, human summary-quality, or QA-accuracy claims.

Reproduce the full-text corpus, analysis, and figures:

```bash
python scripts/build_full_text_corpus_parallel.py
python scripts/evaluate_full_text_corpus.py
node scripts/render_full_text_charts.mjs
```

Results are stored under `results/full_text/`, including exclusions, extraction diagnostics, per-paper ranks, generated summaries, and ROUGE scores. The earlier 600-record abstract experiment remains under `results/ai_research_papers/`, with its report archived at `report/archive/technical_report_600_abstracts.md`. The three-paper pilot is also archived separately at `report/archive/technical_report_three_paper_pilot.md`; its results must not be mixed with the full-text evaluation.

## Implemented features

- Extracts selectable PDF text while retaining one-based PDF page numbers.
- Removes repeated headers/footers and common line-break artifacts.
- Builds overlapping, section-labelled chunks that never cross a page boundary.
- Produces an overview and six structured fields: problem, objectives, methodology, dataset/setup, findings, and limitations/future work.
- Provides LEAD and TextRank extractive summaries, plus optional BART summarization.
- Supports BM25, TF-IDF, sentence-transformer dense retrieval, and reciprocal-rank-fusion hybrid retrieval.
- Uses FAISS automatically when installed; otherwise dense retrieval uses NumPy.
- Answers offline by quoting relevant evidence, or optionally uses OpenAI with a strict evidence-only prompt.
- Shows source identifiers, page numbers, retrieval scores, and all retrieved passages.
- Computes ROUGE-1/2/L and Precision@k/Recall@k from independent human annotations.
- Includes a real three-paper pilot, human-review rubric, retained raw outputs, charts, and reproduction scripts.

**Defaults:** TF-IDF retrieval, TextRank summarization, top-k = 5, and offline evidence quotation. TF-IDF was selected because it had the best aggregate pilot Precision@5 and Recall@5; this is not evidence of general superiority.

> Offline answers are intentionally extractive: they quote document sentences rather than pretending to be locally generated LLM prose. Important outputs should always be checked against the original paper.

## Project structure

```text
research-assistant/
├── app.py                         # Streamlit web application
├── research_assistant/            # Commented processing and NLP modules
├── scripts/
│   ├── demo.py                    # Offline terminal demonstration
│   ├── download_corpus.py         # URL download + SHA-256 verification
│   ├── evaluate.py                # Single-paper experiment runner
│   ├── reproduce_pilot.py         # Complete measured pilot in one command
│   ├── aggregate_pilot.py         # Metrics, intervals, and human ratings
│   ├── build_full_text_corpus_parallel.py  # Resumable 600-record acquisition
│   ├── evaluate_full_text_corpus.py        # 595-paper retrieval and ROUGE
│   └── render_full_text_charts.mjs         # Current full-text report figures
├── data/
│   ├── ai_research_papers.csv     # 600 supplied metadata/abstract records
│   ├── full_text/                 # Local-only PDFs and extracted page JSON
│   ├── corpus_manifest.json       # Earlier pilot source URLs and hashes
│   ├── annotations/               # Independent pilot summaries and Q&A
│   └── human_scores.json          # One-reviewer pilot Q&A ratings
├── notebooks/                     # Google Colab demonstration
├── report/                        # 3,137-word report, archives, and figures
├── results/full_text/             # Current profiles, ranks, and summaries
├── results/pilot/                 # Archived three-paper experiment evidence
├── tests/                         # Offline unit and integration tests
├── DEMO.md                        # Timed presentation script
└── SUBMISSION_CHECKLIST.md        # Requirement-to-artifact mapping
```

## Quick start: web demo

Requires Python 3.10 or newer.

```bash
cd research-assistant
python -m venv .venv
```

Activate `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the printed local URL, upload a digitally generated English research PDF, and select **Process paper**. The included pilot paper can be restored with:

```bash
python scripts/download_corpus.py
```

For the Kimi Work/local preview launcher, use `npm run dev`; it forwards host and port arguments to Streamlit. See `DEMO.md` for a seven-minute presentation and expected behavior.

## Terminal demo

```bash
python scripts/demo.py
```

By default this uses the TextRank paper, TF-IDF, and TextRank. Supply another PDF or options with, for example:

```bash
python scripts/demo.py path/to/paper.pdf --retrieval bm25 --summarizer lead
```

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

**Evaluation boundary:** dense, hybrid, BART, and external-LLM paths are implemented but were not run in the retained pilot, so the project makes no performance claims for them.

## Python API

```python
from research_assistant import ResearchAssistant

assistant = ResearchAssistant.from_pdf("paper.pdf", retrieval_method="tfidf")
summary = assistant.summarize(method="textrank")
print(summary.overview)
print(summary.sections)

answer = assistant.ask("What dataset was used?", top_k=5)
print(answer.text)
for citation in answer.citations:
    print(citation.chunk.page_numbers, citation.chunk.text)
```

Use `retrieval_method="dense"` or `"hybrid"` after installing the ML dependencies. Use `summarize(method="lead")` for the extractive opening-sentence baseline.

## Architecture

```text
PDF → page extraction → cleanup → section/page-aware chunks
                                  ├── LEAD / TextRank / optional BART → research brief
                                  └── retriever index
Question → BM25 / TF-IDF / dense / hybrid → top-k evidence
                                            ├── offline evidence quotation
                                            └── optional evidence-only LLM
Evidence → source IDs + PDF page numbers → inspection interface
```

The pipeline deliberately separates retrieval from generation so failures can be traced to extraction, retrieval, evidence selection, or external generation.

## Measured pilot

The corpus contains three public NLP papers: *Attention Is All You Need*, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, and *TextRank: Bringing Order into Text*. It has 42 pages, 205 chunks, 15 answerable questions, and three deliberately unsupported questions. PDFs are excluded from Git; `data/corpus_manifest.json` records source URLs and verified SHA-256 hashes.

### Headline results

| Measure | LEAD | TextRank |
|---|---:|---:|
| Macro ROUGE-1 | 0.334 | 0.327 |
| Macro ROUGE-2 | 0.092 | 0.088 |
| Macro ROUGE-L | 0.186 | 0.211 |

| Retriever | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.467 | 0.233 | 0.356 | 0.472 | 0.267 | 0.583 |
| TF-IDF | 0.467 | 0.194 | 0.422 | 0.522 | 0.333 | 0.678 |

The final offline generator correctly abstained on all three unsupported questions and over-refused one of 15 answerable questions. One-reviewer mean scores out of 2 were correctness 1.39, relevance 1.39, evidence support 1.89, and citation accuracy 1.33. These are small-pilot findings, not benchmark claims. See `report/technical_report.md` for intervals, tuning effects, failure analysis, ethics, and limitations.

### Reproduce everything measured

```bash
python scripts/reproduce_pilot.py
```

This verifies or downloads the PDFs, runs LEAD/TextRank and BM25/TF-IDF on all three papers, and rebuilds `results/pilot/aggregate.json`. Existing PDFs with unexpected hashes are not overwritten. Pre-tuning evidence remains under `results/pilot/initial/`; human ratings remain in `data/human_scores.json`.

To regenerate the report charts:

```bash
npm install
node scripts/render_charts.mjs
```

To run a new single-paper evaluation:

```bash
python scripts/evaluate.py paper.pdf data/my_annotations.json \
  --methods bm25 tfidf --summarizers lead textrank --k 5 \
  --output results/my_evaluation.json
```

Chunk IDs depend on chunking settings. Fix `chunk_size` and `overlap` before relevance annotation, and write references independently rather than from generated answers. Use `data/evaluation.example.json` for the annotation schema and `report/evaluation_rubric.md` for human review.

## Tests and validation

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m compileall -q app.py research_assistant scripts
```

Tests cover PDF extraction, repeated-header removal, chunk boundaries, source pages, lexical retrieval and plural normalization, hybrid fusion, grounded answers and abstention, structured summaries, evaluation metrics, and an end-to-end offline PDF workflow.

## Google Colab

Open `notebooks/research_assistant_colab.ipynb` in Colab. Upload or clone this project first, install `requirements.txt`, and run the cells in order. Dense/BART models may require a GPU runtime and additional download time.

## Limitations and responsible use

- Scanned PDFs require OCR, which is not included.
- Multi-column layouts, equations, tables, figures, and unusual headings can reduce extraction quality.
- Chunk overlap does not guarantee that all cross-page context is retained.
- Sparse retrieval can miss paraphrases; dense embeddings do not guarantee relevant retrieval.
- Extractive answers may be incomplete or less fluent; LLM answers can still hallucinate despite grounding instructions.
- Abstention heuristics can over-refuse valid questions and do not guarantee factual reliability.
- The current full-text evaluation has no human summary or QA ratings; title-based known-item retrieval and abstract-referenced ROUGE have limited scope.
- Nine accepted papers have abstract-token coverage below 0.80, so extraction loss can affect retrieval and summarization.
- The archived three-paper pilot used one reviewer, with no inter-rater agreement estimate.
- Use authorized/open-access documents, avoid unnecessary document retention, and consult the original paper before relying on generated output.

## Submission

`SUBMISSION_CHECKLIST.md` maps every assignment requirement to its artifact. The remaining external action is to publish under the student's own GitHub/OSF account and insert the resulting URL in the submission portal. For GitHub, set your own author identity, commit the staged files, add the new repository as `origin`, and push:

```bash
git config user.name "notthedonrobbie"
git config user.email "rjnprsnth@gmail.com"
git commit -m "Complete PaperTrail NLP research assistant"
git remote add origin https://github.com/notthedonrobbie/NLP-project-research-paper-summarizer
git push -u origin main
```


