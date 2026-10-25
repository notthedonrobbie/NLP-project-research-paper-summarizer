# PaperTrail demonstration guide

Use the included TextRank paper for a predictable, fully offline demonstration. The complete run takes about 7 minutes.

## Setup before presenting

```bash
python -m pip install -r requirements.txt
python scripts/download_corpus.py
streamlit run app.py
```

Alternatively, `npm run dev` launches the same Streamlit app using the project's local Python environment. Open the printed local URL, select `data/papers/textrank.pdf` in the file picker, and keep the original PDF available for verification.

## Timed web demonstration

### 0:00–1:00 — Problem and value

Explain that PaperTrail helps a student triage a scientific PDF without losing the connection between generated output and source evidence. It provides a research brief, document-grounded answers, and page-level passages; it does not replace close reading.

### 1:00–2:30 — Process and summarize

1. Confirm **TF-IDF** retrieval and **TextRank** summarization in the sidebar.
2. Upload `textrank.pdf` and select **Process paper**.
3. Show the page/chunk count and overview.
4. Open the six structured fields: problem, objectives, methodology, dataset/setup, findings, and limitations/future work.
5. Point out that LEAD and TextRank were measured; BART is optional and was not evaluated in the pilot.

Expected result: an extractive overview and populated research fields. Wording is source-dependent, so do not promise a fixed paragraph.

### 2:30–4:30 — Grounded question answering

1. Ask: **Which two extraction applications are investigated?**
2. Show the answer, source IDs, cited PDF pages, and retrieved passages.
3. Expand a passage and compare it with the same page in the original PDF.
4. Ask: **How are sentences connected in the TextRank sentence-extraction graph?**
5. Use the second answer to discuss incomplete evidence selection: high evidence support does not guarantee a complete answer.

Expected result: answers quote retrieved source sentences with inspectable citations. The offline generator is deliberately extractive and may include incomplete or extra information.

### 4:30–5:30 — Insufficient-evidence behavior

Ask: **What F1 score did TextRank achieve on SQuAD?**

Expected result: `The uploaded paper does not provide enough evidence to answer this question.` Explain that this behavior was measured on three unsupported questions, but one answerable pilot question was also over-refused; abstention is not a factual-reliability guarantee.

### 5:30–6:15 — Extraction inspection and ethics

Open the extracted page text and explain that scanned PDFs need OCR, while multi-column layouts, equations, tables, and figures may extract imperfectly. Mention that the default path is local; optional OpenAI mode transmits the question and retrieved excerpts and must not be used for unauthorized confidential material.

### 6:15–7:00 — Evidence and limitations

Show `results/pilot/aggregate.json` or the report charts. State the measured headline: TF-IDF Precision@5/Recall@5 was 0.333/0.678 versus BM25 0.267/0.583 on 15 answerable questions. Emphasize the limitations: three papers, 18 total questions, one reviewer, and no measured claims for dense, hybrid, BART, or external-LLM modes.

## Terminal fallback

If the browser is unavailable, run:

```bash
python scripts/demo.py
```

This prints the same core demonstration: summary, research fields, an answerable question with citations, and an unsupported-question check.

## Reproduce the measured pilot

```bash
python scripts/reproduce_pilot.py
node scripts/render_charts.mjs
```

The first command downloads/hash-checks the three public PDFs, reruns LEAD/TextRank and BM25/TF-IDF evaluation, and rebuilds `results/pilot/aggregate.json`. The second regenerates report figures after `npm install`. Human ratings remain in `data/human_scores.json`; they are not regenerated automatically.
