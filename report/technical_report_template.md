# An Intelligent Research Assistant for Scientific Document Summarization and Question Answering

## Abstract

Complete after the experiments. State the research problem, methods, evaluation collection, measured findings, and main conclusion without claiming unobserved improvements.

## 1. Introduction

Describe the literature-review problem, limitations of keyword search and ungrounded generation, research question, objectives, and contributions.

## 2. Related Work

Discuss scientific-document summarization, TextRank, transformer summarization, dense retrieval, BM25, Retrieval-Augmented Generation, and grounded-answer evaluation. Add verified academic citations.

## 3. Dataset and Annotation

Report paper sources and licences, subject area, number of papers, inclusion criteria, annotation procedure, inter-annotator process, and train/test separation if applicable. Do not redistribute restricted full texts.

## 4. Methodology

Document PDF extraction, cleaning, section detection, chunk size/overlap, TextRank and BART settings, embedding model, BM25/TF-IDF/dense/hybrid retrieval, `k`, RAG prompts, citation mechanism, and insufficient-evidence handling.

## 5. Experimental Design

Specify hardware/software, model versions, random seeds where relevant, fixed configurations, baselines, ROUGE-1/2/L, Precision@k/Recall@k, human rubric, and statistical reporting. Separate primary comparisons from sensitivity analyses.

## 6. Results

Insert tables and charts generated from actual `results/*.json` files. Include per-paper results and variability, not only averages. Leave this section empty until experiments have run.

## 7. Error Analysis and Reliability

Inspect retrieval misses, missing technical details, incorrect page references, unsupported statements, over-refusals, and failures to acknowledge insufficient evidence. Include representative examples with source-page verification.

## 8. Discussion and Limitations

Interpret measured results relative to the research question. Discuss PDF layouts, OCR, equations, tables, long context, annotation subjectivity, external API privacy, and generalizability.

## 9. Conclusion and Future Work

Summarize only conclusions supported by the experiments. Consider multi-paper comparison, table/figure extraction, hierarchical summarization, OCR, and improved faithfulness evaluation.

## References

Add complete, verified references for all models, libraries, datasets, and related research cited above.

## Appendices

Include annotation instructions, the human rubric, prompts, configuration files, and links to reproducible result files.
