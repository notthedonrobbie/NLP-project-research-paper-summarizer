# Submission readiness checklist

## Assignment requirements

| Requirement | Concrete submission artifact | Status |
|---|---|---|
| Problem definition and NLP justification | `report/technical_report.md`, Section 1 | Complete |
| Input-data description | Report Section 2; source CSV; full-text corpus profile | Complete |
| Preprocessing and embedding techniques | Report Section 3; extraction and evaluation scripts; retrieval module | Complete |
| Proposed NLP/LLM models and pipelines | Report Sections 3–4; summarization, retrieval, RAG, and pipeline modules | Complete |
| Tool stack and platforms | Report Section 3.1; README; `requirements*.txt` | Complete |
| Ethical considerations and mitigations | Report Section 6; README responsible-use section | Complete |
| Expected output and value proposition | Report Section 1; full-text architecture; working web prototype | Complete |
| Technical report, 3,000–4,000 words | `report/technical_report.md` — 3,137 words by the repository token check | Complete |
| Architecture, model selection, and evaluation | Report Sections 3–5; 595-query retrieval and 595-paper ROUGE results | Complete |
| Working web or notebook prototype | `app.py`; `notebooks/research_assistant_colab.ipynb`; `DEMO.md` | Complete |
| Well-commented code and README | Source modules, scripts, tests, and `README.md` | Complete |
| Demonstration of the solution | Seven-minute web script in `DEMO.md`; reproducible full-text experiment | Complete |
| Evaluation evidence | `results/full_text/evaluation.json`; ranks, summaries, ROUGE, bootstrap intervals | Complete |
| Corpus integrity and exclusions | `results/full_text/corpus_profile.json`; five documented excluded records | Complete |
| One-command analysis reproduction | `scripts/evaluate_full_text_corpus.py` | Complete |

## Current full-text results

- Corpus: 595 of 600 requested papers, 12,836 pages, and 6,607,965 extracted words.
- Retrieval: TF-IDF MRR 0.949 and Hit@10 0.985; BM25 MRR 0.926 and Hit@10 0.971.
- Summarization: TextRank macro ROUGE-1/2/L 0.345/0.106/0.179; LEAD 0.332/0.101/0.187.
- Figures: `report/figures/full-text-architecture.png`, `full-text-retrieval.png`, and `full-text-summarization.png`.

## Evidence boundaries to preserve

- Four HTTP 404 downloads and one parser-limit failure were excluded; do not claim that all 600 PDFs succeeded.
- Full-text retrieval uses title queries and is not natural-language QA. Printed titles in PDFs can make this known-item task optimistic.
- Summaries use body text with abstracts and references conservatively excluded, and author abstracts as ROUGE references.
- ROUGE measures abstract alignment, not factual faithfulness or human-rated usefulness. No full-corpus human summary or QA evaluation was performed.
- Nine papers have abstract-token coverage below 0.80; extraction limitations must remain visible.
- Dense retrieval, hybrid retrieval, BART, and external OpenAI generation were not evaluated on this corpus.
- The earlier abstract-only and three-paper experiments are archived separately and must not be mixed with current results.
- Bulk PDFs and extracted full-text JSON are local-only and excluded from Git.

## Final external step

- [ ] Set the student's own Git author identity, commit the staged files, create a GitHub or OSF repository under their account, push `main`, and paste the resulting URL into the submission portal. Exact GitHub commands are in `README.md`.

Public publication is intentionally not performed by project automation. Before publishing, the student should confirm authorship details, course policy, dataset provenance/licensing, and that no private files or credentials have been added.
