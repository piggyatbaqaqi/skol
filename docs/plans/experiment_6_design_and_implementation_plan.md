# SKoL Experiment 6: Layered Segmentation Pipeline — Design and Implementation Plan

## 1\. Problem statement

The current segmentation pipeline mixes several distinct abstraction layers into one CRF pass: page-layout recognition (headers, footers, captions, tables), article/treatment boundary detection, and line-level semantic labeling all influence the same transition matrix. This causes the model to treat label transitions and physical text boundaries as equivalent signals, producing spurious splits mid-paragraph and mid-sentence. Line granularity is currently the raw newline from PDF/XML extraction, which is unreliable on its own — OCR artifacts frequently produce one or two words per physical line.

The goal of Experiment 6 is to decompose this into independently trainable layers, each with a narrow responsibility, its own algorithm, and its own training signal.

## 2\. Proposed architecture

1. **Layout filter** — strips or tags page headers, footers, figure captions, and tables. Physical/layout knowledge only; no taxonomy content.  
2. **Article segmenter** — splits a source document (which may be a single born-digital article or a whole scanned issue/book) into individual articles, identifying front-matter regions (title, abstract, introduction, bibliography).  
3. **Treatment detector** — within a taxonomic article, locates treatment boundaries, informed by journal-specific house style. Handles cases like a new genus immediately followed by a new species, where each taxonomic level is a separate treatment.  
4. **Line labeler (CRF)** — assigns a section label (description, habitat, distribution, taxonomic citation, etc.) to each line within a single treatment. Existing SBERT \+ CRF architecture, retrained once layout noise is removed upstream.  
5. **Span grouper** — merges label-homogeneous line runs into coherent spans. Owns paragraph/sentence integrity as a hard constraint rather than an emergent property of CRF label transitions. Includes a learned line-join classifier and separate handling for keys/tables, which don't reduce to sentence extraction.

## 3\. Per-layer design

### 3.1 Layout filter

- **Algorithm**: lightweight per-line classifier (logistic regression or small MLP) on positional and repetition features. Supplemented by an unsupervised signal: near-identical text recurring at the same page position across consecutive pages is a strong header/footer indicator independent of any label.  
- **Training data**: reusable from the existing CRF training set — page headers, figure captions, and tables are already first-class labels there.  
- **Complexity estimate**: **2 units**

### 3.2 Article segmenter

- **Algorithm**: sequence classifier over line/paragraph units for front-matter categories (title, abstract, introduction, bibliography), plus boundary detection between articles for multi-article sources.  
- **Training data**: check first whether the corpus's ingestion metadata already encodes article boundaries for free — born-digital sources (Plazi, BHL, BioStor) are typically downloaded one article per file, which is usable boundary supervision with no new labeling. Older scanned issues/books will need new labels, since front-matter categories aren't captured by the current schema at all.  
- **Complexity estimate**: **5 units**

### 3.3 Treatment detector

- **Algorithm**: hybrid — a house-style-driven pattern matcher (using per-journal GGserver configs from Plazi) with an ML fallback classifier for out-of-config journals or ambiguous cases.  
- **Training data**: two sources. (1) Distant supervision from Plazi's TreatmentBank output for journals they've already processed — validate the detector against their existing treatment segmentation rather than hand-labeling from scratch. (2) For journals outside Plazi's coverage, new labeling informed by the accumulated style notes from Claude-API label validation, covering variant styles such as new-genus-then-new-species.  
- **Complexity estimate**: **8 units**

### 3.4 Line labeler (CRF)

- **Algorithm**: unchanged — SBERT embeddings with a CRF head (`torchcrf` or `sklearn-crfsuite`), regex/gazetteer features for reliable label locking (e.g. MycoBank number → taxonomic citation).  
- **Training data**: existing BIO-labeled data, largely reusable. Requires retraining once layout labels are removed from the transition matrix, since the label space and feature set both simplify.  
- **Complexity estimate**: **3 units**

### 3.5 Span grouper

- **Algorithm**: a per-boundary line-join classifier (join vs. break) rather than a single global heuristic. Features include line length relative to a local running median (not a fixed threshold), run-length of preceding short/long lines, terminal punctuation, mid-hyphenation, capitalization of the following line, and the CRF's own line label (so keys/tables get different join behavior than prose). Sentence segmentation tools (spaCy sentencizer, Punkt, pySBD) are only applied after reflow, and only to spans labeled as prose — keys and tables get a structure-aware joiner instead.  
- **Training data**: silver data first — where a raw line-broken version and a clean/reflowed version of the same text both exist (schema-constrained pipeline output, extracted description fields), align them to auto-derive join/break labels with no manual annotation. New labeling reserved for cases silver alignment can't resolve, particularly heavy-OCR documents with no clean counterpart.  
- **Complexity estimate**: **5 units**

## 4\. Data strategy

- **Silver vs. gold separation**: keep auto-derived (silver) and carefully annotated (gold) labels clearly separated in the training pipeline for the span grouper and treatment detector, to avoid compounding errors — especially since Mistral outputs are already reused as schema seeds elsewhere in the project.  
- **Evaluation**: extend the existing difficulty-stratification approach (description length, feature density, Latin phrasing, nesting load) to build held-out eval sets per new layer, so each layer's quality can be verified independently before it feeds the next.

## 5\. Complexity estimates

Unit definition: 1 complexity unit ≈ the effort to implement a simple ingest for one new journal (i.e., a straightforward house-style configuration case).

| Layer | Estimate (units) |
| :---- | :---- |
| Layout filter | 2 |
| Article segmenter | 5 |
| Treatment detector | 8 |
| Line labeler (CRF) retrain | 3 |
| Span grouper | 5 |
| Cross-layer eval framework | 3 |
| Pipeline integration / orchestration | 3 |
| **Total** | **\~29** |

These are engineering-judgment estimates, not measured effort — intended for relative sequencing and prioritization, not a committed schedule.

## 6\. Open questions / risks

- Does the corpus's ingestion metadata actually preserve article-level file boundaries for born-digital sources? This materially changes the article segmenter estimate if not.  
- How much of the corpus has a "clean" reflowed counterpart usable for span grouper silver-data alignment, versus requiring new labeling?  
- What fraction of target journals are covered by Plazi's TreatmentBank versus requiring net-new treatment boundary labeling?  
- Risk of error compounding if silver labels derived from one layer's output are used to train a downstream layer without a gold-labeled check.

## 7\. Suggested phasing

1. Layout filter (low complexity, unblocks CRF retrain)  
2. Line labeler retrain (validates that layout removal actually improves labeling quality)  
3. Span grouper (addresses the most visible current failure — paragraph/sentence splitting)  
4. Article segmenter  
5. Treatment detector (highest complexity; benefits from lessons learned in the article segmenter)  
6. Cross-layer eval framework and full pipeline integration, running in parallel with the above

