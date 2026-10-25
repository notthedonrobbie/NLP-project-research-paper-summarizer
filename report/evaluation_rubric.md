# Human evaluation rubric

Evaluate Q&A independently using the original PDF and retrieved evidence. Record 0, 1, or 2 in the blank `human_scores` fields produced by `scripts/evaluate.py`.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Correctness | Incorrect or contradicts the paper | Partly correct but incomplete/ambiguous | Correct and covers the essential reference information |
| Relevance | Does not address the question | Addresses only part of the question | Directly addresses the full question |
| Evidence support | Main claims unsupported | Some claims supported, others unsupported | Every material claim is supported by cited evidence |
| Citation accuracy | Missing or wrong sources | Relevant source present but incomplete/imprecise | Sources and page references support the claims precisely |

Also record whether insufficient-evidence behaviour is appropriate:

- **Correct acknowledgement:** the document lacks the requested information and the system says so.
- **Over-refusal:** sufficient document evidence exists, but the system refuses.
- **Unsupported answer:** the document lacks evidence, but the system answers anyway.

For summaries, rate factual correctness, coverage, readability, and preservation of technical details separately from ROUGE. Where feasible, use two annotators, resolve rubric disagreements, and report agreement or adjudication procedures.
