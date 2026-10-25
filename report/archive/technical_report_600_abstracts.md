# PaperTrail: Abstract-Level Discovery and Summarization of AI Research Papers

## Abstract

The rapid growth of artificial-intelligence research makes it difficult for students and researchers to identify papers relevant to a particular interest. This report presents an abstract-level configuration of **PaperTrail**, an NLP research assistant that combines metadata validation, lexical retrieval, extractive summarization, and source-linked inspection. The evaluation uses the supplied `ai_research_papers.csv` dataset containing **600 unique arXiv-linked records**, with titles, authors, publication dates, author abstracts, and links. The records span November 2008 to 10 April 2025 and contain 105,022 abstract words, with a mean abstract length of 175.04 words. Because the dataset contains no full text, question-answer annotations, or independent generated-summary references, the report does not reuse earlier full-paper QA or ROUGE results. Instead, it evaluates a reproducible known-item task: each paper title becomes a query, and the correct abstract must be retrieved from all 600 abstracts. BM25 achieved 0.905 mean reciprocal rank and 0.965 Hit@10; TF-IDF achieved 0.903 and 0.972 respectively. Their paired MRR difference was small and its bootstrap interval included zero. The results show that simple lexical methods are effective for title-based known-item search, but they do not establish performance on natural research questions. The principal value is transparent, low-cost paper discovery with explicit evidence boundaries and direct links to source records.

## 1. Problem Definition, NLP Justification, and Value

The target problem is **research-paper discovery and triage under information overload**. A student beginning a literature review may know a topic, method, or paper title but still need to locate the most relevant records, inspect their abstracts, and produce a short briefing before committing to full-paper reading. A spreadsheet can store bibliographic records, but it does not rank documents by textual similarity or identify central sentences. General-purpose generative systems can answer questions fluently, yet their responses may rely on parametric memory rather than the supplied corpus. PaperTrail therefore treats the task as corpus-conditioned information access: retrieval and generated outputs must remain connected to an identifiable source record.

NLP is appropriate because the central operations involve language rather than tabular filtering alone. Tokenization and normalization make lexical matching consistent; TF-IDF and BM25 estimate relevance between a query and an abstract; sentence segmentation and TextRank support extractive summarization; and optional sentence embeddings provide a route to semantic matching when lexical overlap is insufficient. These techniques address different failure modes. Metadata validation ensures that a paper can be traced, retrieval ranks candidate papers, and summarization reduces reading effort. Combining them is more useful than presenting an unranked list or an unsupported generated review.

The expected output is a ranked set of papers with titles, dates, source links, and matched abstracts, together with an optional concise abstract brief. The value proposition is faster navigation and improved inspectability, not automatic assessment of scientific quality. No controlled user study was conducted, so the report does not claim a measured time saving. The intended users are students and researchers performing initial discovery; the system should not make publication, funding, medical, legal, or other high-stakes decisions.

## 2. Dataset Description and Audit

### 2.1 Source, structure, and scope

The dataset is the user-supplied file `data/ai_research_papers.csv`. It contains 600 rows and five columns: `Title`, `Authors`, `Published`, `Summary`, and `Link`. Each row represents one research-paper record, and every link is an arXiv abstract-page URL. The file has no missing cells, exact duplicate rows, or duplicate normalized titles. All 600 links and all 600 abstracts are unique. Its SHA-256 hash is `feb8262a4512b5677797d16c4ca65435d19ea0d2d286bfe9defc475c820863f8`, allowing the evaluated version to be distinguished from later edits.

| Field | Role in the prototype | Important limitation |
|---|---|---|
| Title | Known-item query and display label | Generic titles may share few distinctive terms with the abstract |
| Authors | Bibliographic display and approximate author-count analysis | Comma-separated names are not a normalized author authority file |
| Published | Temporal profiling and potential filters | Dates describe the supplied record and do not establish venue or peer review |
| Summary | Indexed author abstract and summarization input | Not full text and not an independent reference summary |
| Link | Source traceability and manual verification | Accessibility and licensing must be checked before redistribution |

The most important scope boundary is that `Summary` contains an **author abstract**, not the paper’s complete content. The CSV provides no sections, tables, figures, citations, page numbers, relevance judgments, or question-answer annotations. Consequently, it supports abstract-level retrieval and summarization experiments, but it cannot substantiate claims about full-paper QA, page-level citation accuracy, table understanding, or factual coverage of results. Full-paper processing would require retrieving authorized documents through the supplied links and evaluating them separately.

### 2.2 Descriptive statistics

Publication timestamps range from 21 November 2008 to 10 April 2025. The distribution is strongly skewed toward recent work: 2023 contains 112 records, 2024 contains 208, and the partial 2025 period contains 103. Together these account for 423 papers, or 70.5% of the dataset. By contrast, 2008, 2015, and 2016 contribute one record each. The year 2025 must not be interpreted as a decline in research output because collection ends in April.

![Paper records by publication year](figures/ai-dataset-year-distribution.png)

The abstracts contain 105,022 words under the evaluation script’s whitespace-normalized word-expression count. Abstract length ranges from 28 to 305 words, with a median of 175, mean of 175.04, and 90th percentile of 249. Titles have a median length of 10 words and a mean of 10.62. A comma-separated estimate gives a median of three authors per paper, mean of 3.67, and maximum of 25, although this is only an approximation because author naming conventions may vary.

These statistics confirm that the dataset is large enough for a meaningful retrieval pilot, but size alone does not establish representativeness. The collection procedure, search query, venue coverage, language filter, and sampling frame were not documented in the CSV. The records therefore should be treated as a supplied convenience corpus rather than a random sample of all AI research. One visible example of possible keyword noise is *A local characterization for the Cuntz semigroup of AI-algebras*, where “AI” has a mathematical meaning unrelated to artificial intelligence. Such records make domain validation necessary before drawing field-level conclusions.

## 3. Preprocessing, Embeddings, and System Architecture

The first stage validates the exact five-column schema and checks for missing values and duplicate links. Stable identifiers such as `paper-0001` are assigned according to CSV row order, while the original link remains the external source identifier. Whitespace normalization converts embedded line breaks in titles and abstracts to single spaces without rewriting technical content. Publication strings are parsed as timezone-aware UTC timestamps. The evaluation also records file hashes and per-paper ranks so that failures can be inspected rather than hidden behind an aggregate score.

Abstracts are indexed as complete documents because their median length is only 175 words. Splitting them into 1000-character PDF-style chunks would produce unnecessary fragmentation and complicate the one-relevant-document evaluation. In the retrieval experiment, only `Summary` is indexed; titles are deliberately excluded so that a query cannot succeed merely by matching an identical title field. This design tests whether title terminology can locate the corresponding abstract while preserving a realistic separation between query and document text.

![Abstract-level research discovery architecture](figures/ai-dataset-architecture.png)

The sparse TF-IDF representation uses English stop-word removal, unigrams and bigrams, sublinear term frequency, and normalized vectors. It captures exact and phrase-level lexical evidence without model training. BM25 instead represents each abstract through token frequencies, inverse document frequency, term-frequency saturation, and document-length normalization. Both approaches run locally and are appropriate for a 600-document corpus. Optional `all-MiniLM-L6-v2` sentence embeddings remain relevant for paraphrase-oriented search, but dense and hybrid configurations were not evaluated on this dataset and receive no performance claims. Sentence-BERT provides the basis for reusable sentence embeddings, while reciprocal-rank fusion can combine rankings without adding incompatible raw scores. [Reimers and Gurevych (2019)](kimi-cite://web?url=https%3A%2F%2Faclanthology.org%2FD19-1410%2F "citation") [Cormack, Clarke, and Buettcher (2009)](kimi-cite://web?url=https%3A%2F%2Fcormack.uwaterloo.ca%2Fcormacksigir09-rrf.pdf "citation")

The output architecture links retrieval and summarization to an inspection interface. A user should be able to move from a ranked result to its abstract and arXiv page before using it in a literature review. The existing Streamlit prototype demonstrates PDF summarization and grounded QA, while the new CSV experiment is implemented in `scripts/evaluate_ai_research_papers.py` and shares the same BM25 and TF-IDF retrievers. This distinction is explicit: a CSV-upload interface and cross-paper QA are integration work, not features claimed as evaluated here.

## 4. Model Selection and Summarization Strategy

BM25 was selected as a strong classical retrieval baseline because it balances term rarity, repeated occurrences, and document length. The implementation uses `k1 = 1.5` and `b = 0.75`, common default settings that were fixed before evaluation. [Robertson and Zaragoza (2009)](kimi-cite://web?url=https%3A%2F%2Fdoi.org%2F10.1561%2F1500000019 "citation") TF-IDF provides a contrasting vector-space baseline with sublinear frequency scaling and bigram features. Neither method is trained on the evaluation queries, so there is no fitted classifier requiring a train/test split. Hyperparameters were not tuned against the reported results; this avoids presenting best-on-test settings as independent evidence.

The summarization component retains LEAD and TextRank. LEAD selects opening sentences and is a useful baseline because abstracts often state the problem and contribution early. TextRank constructs a graph in which sentences are vertices and lexical similarities are weighted edges, then ranks sentences by graph centrality before restoring source order. [Mihalcea and Tarau (2004)](kimi-cite://web?url=https%3A%2F%2Faclanthology.org%2FW04-3252%2F "citation") This unsupervised method requires no training corpus and is suitable for short documents. Optional BART summarization could produce abstractive briefs, but it was not run and is not presented as an evaluated improvement.

No ROUGE score is reported for this CSV experiment. The author abstract is the input document, not an independently written target summary; scoring extracted sentences against the same abstract would reward overlap and longer outputs rather than establish summary usefulness. A valid summarization evaluation would require independent reference briefs or structured human judgments of coverage, faithfulness, concision, and readability. Similarly, the absence of QA annotations means that answer correctness, evidence support, abstention, and citation accuracy cannot be inferred from retrieval ranks. Keeping these tasks separate prevents a strong known-item result from becoming an unsupported claim about a complete research assistant.

## 5. Experimental Design and Metrics

The automatic evaluation is **known-item title-to-abstract retrieval**. For each of the 600 records, the cleaned title is issued as a query against all 600 cleaned abstracts. Only the abstract from the same row is relevant. Each retriever returns at most ten candidates with positive scores, and the target’s rank is retained. A target absent from the top ten is treated as a miss and contributes zero to reciprocal rank and all hit-rate measures.

Hit@k is the proportion of queries whose target appears at rank k or better. Because there is exactly one relevant document per query, Hit@k is equivalent to Recall@k for this task. Mean reciprocal rank summarizes the full top-ten ranking:

`MRR = (1/N) × Σ 1/rank_i`

where `N = 600` and a miss contributes zero. Unlike Precision@k in multi-relevance evaluation, the primary emphasis here is whether the user finds one known paper quickly. The experiment reports Hit@1, Hit@3, Hit@5, Hit@10, MRR, and the number of misses. These measures are complementary: Hit@1 reflects immediate exact retrieval, while MRR also distinguishes a target at rank two from one at rank nine.

Uncertainty was estimated with 5,000 paper-level bootstrap resamples using seed 2026. Percentile intervals were calculated for MRR and hit rates. A paired bootstrap also compared TF-IDF minus BM25 MRR using the same resampled papers, preserving the within-paper comparison. These intervals describe variability across the supplied records; they do not repair collection bias or turn title queries into natural-language relevance judgments. No external LLM, BART, dense retriever, or hybrid retriever was included in the measured comparison.

Several design choices were made to keep the comparison fair and interpretable. Both retrievers received identical cleaned titles and abstracts, used their established default configurations, and were evaluated over the same 600 targets. No result-dependent parameter search was performed. The index was not divided into training and test partitions because these unsupervised methods do not learn from relevance labels; nevertheless, the reported scores remain descriptive of this fixed corpus rather than estimates for unseen collections. The target title was excluded from the indexed abstract text, preventing direct duplication of the complete title field. However, this cannot remove the natural lexical relationship between a title and its abstract, so the task should be understood as a diagnostic of known-item navigation rather than a substitute for realistic topic search.

A stronger subsequent experiment would use a held-out set of natural queries written before system rankings are inspected. Annotators would mark all relevant records, including cases where several papers address the same question, and a second reviewer would check a sample for agreement. Precision, recall, and normalized discounted cumulative gain could then measure ranking quality beyond a single target. Temporal analysis could additionally test whether retrieval degrades on newer terminology. These extensions would provide more externally meaningful evidence without changing the lightweight retrieval architecture.

## 6. Results and Error Analysis

Both lexical methods performed strongly on the known-item task. BM25 obtained MRR **0.905** with a 95% bootstrap interval of **[0.884, 0.925]**, while TF-IDF obtained **0.903 [0.882, 0.924]**. Both placed the correct abstract first for 86.8% of titles. TF-IDF was slightly higher at k = 5 and k = 10, whereas BM25 was higher at k = 3. The median rank among targets found in the top ten was one for both methods.

| Retriever | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Misses in top 10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.905 | 0.868 | 0.935 | 0.947 | 0.965 | 21 |
| TF-IDF | 0.903 | 0.868 | 0.927 | 0.953 | 0.972 | 17 |

![Known-item retrieval results](figures/ai-known-item-retrieval.png)

The differences should not be overstated. TF-IDF minus BM25 MRR was **−0.0016**, with a paired 95% interval of **[−0.0173, 0.0136]** that includes zero. TF-IDF found 17 rather than 21 targets in the top ten, but this does not establish a general advantage. The appropriate interpretation is that the two methods are comparably effective for this corpus and query type. TF-IDF remains a reasonable lightweight default, while BM25 provides a competitive alternative and slightly better Hit@3.

Error analysis reveals why the task remains difficult despite high aggregate scores. Generic titles such as *The Human Factor in AI Safety* and *Responsible AI by Design in Practice* contain frequent domain terms but few distinctive lexical cues; both methods missed these targets in the top ten. *Responsible AI Research Needs Impact Statements Too* was also missed by both. Its abstract is only 28 words and uses wording including “responsible artificial intelligence” and “ethical AI,” leaving little overlap with several title terms. *When Should Algorithms Resign? A Proposal for AI Governance* ranked eighth under BM25 but was absent under TF-IDF; the abstract uses “algorithmic resignation,” illustrating morphological and phrasing variation that exact lexical representations may miss.

These failures support testing dense retrieval and query expansion in future work, but they do not prove that either will help. Semantic models can retrieve topically similar yet incorrect papers, especially when many records share broad AI terminology. A more realistic evaluation should collect natural questions such as “Which papers study governance requirements for high-risk AI?” and annotate every relevant abstract. Multiple annotators should adjudicate relevance, and results should be reported by topic and publication period. A separate human summarization study should assess whether generated briefs preserve the paper’s problem, approach, findings, and limitations without adding claims.

## 7. Ethics, Reliability, and Limitations

The primary ethical risk is **misleading research communication**. An abstract may omit important limitations, numerical context, or corrections present in the full paper. A fluent summary can amplify that omission, and a highly ranked result is not necessarily a high-quality or peer-reviewed study. PaperTrail mitigates this risk by retaining titles and arXiv links, exposing source text, and avoiding quality judgments. Users must consult the original paper before citing or applying a claim. NIST’s AI Risk Management Framework similarly emphasizes validity, transparency, human oversight, and evaluation in context rather than reliance on presentation fluency. [NIST AI RMF](kimi-cite://web?url=https%3A%2F%2Fnvlpubs.nist.gov%2Fnistpubs%2Fai%2Fnist.ai.100-1.pdf "citation")

Dataset governance is a second concern. The supplied CSV does not document its collection query, sampling procedure, license, or intended redistribution terms. Public arXiv links improve traceability but do not by themselves grant permission to republish every abstract or PDF. The dataset should therefore be accompanied by provenance and license checks before public release. Author names should be used for bibliographic attribution, not automated employment, nationality, gender, or quality inference. If optional external generation is enabled, users should also understand that queries and excerpts may leave the local machine; the measured lexical pipeline itself requires no external API.

Several limitations constrain the conclusions. The corpus is temporally imbalanced, possibly affected by keyword noise, and not demonstrably representative across AI subfields. Abstract-only indexing excludes methodological details, experiments, tables, and limitations. Title queries are easier than many natural questions because titles often contain terminology repeated in the target abstract, which helps explain the high hit rates. The one-relevant-item assumption does not measure how many other papers are also relevant to a topic. Finally, no independent annotations or human ratings were supplied, so the report makes no summarization-quality, QA-correctness, or user-productivity claims.

## 8. Reproducibility and Demonstration

The complete CSV analysis can be reproduced from the project root with:

```bash
python scripts/evaluate_ai_research_papers.py
node scripts/render_ai_research_paper_charts.mjs
```

The first command validates the dataset, computes descriptive statistics, runs both retrievers over all 600 titles, performs bootstrap analysis, and writes two artifacts: `results/ai_research_papers/dataset_profile.json` and `known_item_retrieval.json`. The second recreates the report figures after `npm install`. Per-query titles, links, and ranks are retained so that the reported failures can be checked directly. The implementation uses Python, pandas, NumPy, scikit-learn, Streamlit, ECharts, and the existing PaperTrail retrieval modules.

For a live demonstration, the existing web prototype can still show PDF processing, extractive summaries, evidence retrieval, and insufficient-evidence behavior. The CSV experiment should be presented as the new abstract-level evaluation rather than as evidence that the web interface already supports CSV upload. A coherent demonstration therefore has two parts: run the reproducible 600-paper retrieval evaluation, then use the web prototype to show how an authorized full paper can be inspected after discovery. Future integration could add CSV upload, year filters, cross-paper result tables, and links from discovered records to full-paper processing.

## 9. Conclusion

The supplied dataset increases the evaluation corpus from a small paper pilot to **600 unique research records** and supports a reproducible abstract-level retrieval study. BM25 and TF-IDF both achieved approximately 0.90 MRR and retrieved more than 96% of target abstracts within the top ten. Their near-identical MRR and overlapping uncertainty intervals support a conclusion of comparable known-item performance, not a claim that one retriever is universally superior. The work also establishes clear evidence boundaries: titles and abstracts support discovery and brief extractive summarization, while full-paper QA, citation accuracy, and summary usefulness require additional documents and annotations. PaperTrail’s value therefore lies in transparent, efficient research triage backed by inspectable sources and measured limitations. The most important next step is a natural-query, multi-annotator relevance set, followed by dense/hybrid retrieval comparisons and independent human evaluation of generated briefs.
