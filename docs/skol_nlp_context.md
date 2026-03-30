# Context Primer: Mycological Taxonomy Section Classifier

## Task
Classify standard sections of mycological taxonomy papers (e.g., Description, Etymology, Habitat, Discussion, Taxonomy) using NLP.

## Approach
1. Embed text segments using a **multilingual SBERT model** (e.g., `paraphrase-multilingual-mpnet-base-v2`), producing a 768-dimensional vector per segment.
2. Train a **logistic regression classifier** (or small MLP as a step up) on those vectors using labeled examples. The classifier learns decision boundaries in the 768-dimensional embedding space.
3. At inference: text → SBERT → 768-vector → classifier → predicted label.

## Why Multilingual
SBERT embeddings are language-neutral: semantically similar text from any language lands near the same point in vector space. A single classifier trained on those vectors generalizes across languages without retraining. The Latin-heavy terminology of mycological writing (species names, morphological terms) makes this domain especially well-suited to cross-lingual transfer.

## Key Decisions So Far
- Single cross-lingual classifier preferred over per-language classifiers (sparse domain data, Latin anchors embeddings across languages).
- Start with logistic regression; only escalate to MLP if accuracy plateaus.
- Validate accuracy broken down per language to catch any weak spots.

## Broader Project
This classifier is likely a component of the Synoptic Key of Life (SKoL) project, which synthesizes materials across multiple graduate courses (IST664, IST690, IST691, IST718, IST769, IST782) into a unified system diagram. Related materials are in the GitHub repo: https://github.com/piggyatbaqaqi/skol
