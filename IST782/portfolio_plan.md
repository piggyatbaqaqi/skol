# MS Applied Data Science Portfolio Plan
## La Monte Henry Piggy Yarroll

---

## Document Structure

### 1. Title Page and Overview
- Name, SUID, Email
- Table of Contents
- Executive Summary: The SKOL Project as the unifying thread

### 2. Introduction: Purpose and Motivation

**Key Narrative:** I started the MS in Applied Data Science specifically to gain the skills needed to complete the Synoptic Key of Life (SKOL) project. SKOL originated in 2019 as a personal research project without a clear path to implementation. The degree provided the systematic training in data science that made the project achievable.

**Content from about.html:**
- SKOL is a synoptic key for mycological taxonomy
- Unlike dichotomous keys, synoptic keys allow flexible starting points
- Goal: Enable machines to read taxonomic literature so humans can search by description
- Applies ML/NLP to make fungal taxonomy literature accessible to researchers and enthusiasts

**Personal Context:**
- Amateur mycologist familiar with the challenge of species identification
- Recognized need for semantic search over taxonomic literature
- Each course project advanced a specific component of SKOL

---

## 3. Program Learning Goals - Mapping to Projects

### Learning Goal 1: Collect, Store, and Access Data
**Technologies leveraged for data management across the full data lifecycle**

| Project | Evidence |
|---------|----------|
| **IST769** | CouchDB document store for taxa records and JSON structures; Redis for embedding caching; PySpark data pipelines; metadata tracking with provenance (journal names, BibTeX, URLs, scrape dates) |
| **IST664** | Web scraping shell scripts (wget) for mycological journals; YEDDA annotation tool for corpus labeling; 190 labeled journal issues, 60,754 paragraphs |
| **IST718** | Ingestion of 1,000+ unannotated journal issues; OCR processing of page scans; Parquet format for resting state |
| **IST690** | Django/PostgreSQL backend; CouchDB integration for taxa retrieval; REST API design |

**Key Deliverables:**
- IST769 Final Project Proposal (pipeline architecture)
- IST664 data preprocessing pipeline code
- Django application with CouchDB MCP integration

---

### Learning Goal 2: Create Actionable Insight Across Contexts
**Full data science lifecycle applied to scientific research domain**

| Project | Insight Created |
|---------|-----------------|
| **IST664** | Semantic search enabling researchers to find relevant species descriptions using natural language queries rather than exact keywords |
| **IST691** | JSON feature extraction from prose descriptions to support UI menu generation for description building |
| **IST718** | Paragraph classification (Nomenclature/Description/Misc) enabling automated content extraction at scale |
| **IST690** | Public-facing website at synoptickeyof.life making research accessible to taxonomists worldwide |

**Societal Context:**
- Opens taxonomic literature to enthusiastic amateurs, not just professionals
- Supports biodiversity research and species identification
- Addresses challenge that terminology evolves over time and authors use different vocabulary

**Key Deliverables:**
- IST664 Team Report: MycoSearch semantic search system
- IST691 Paper: Feature extraction pipeline
- Live website deployment

---

### Learning Goal 3: Apply Visualization and Predictive Models
**Models generating actionable insight from mycological data**

| Project | Models/Visualizations |
|---------|----------------------|
| **IST664** | SBERT (Sentence-BERT) embeddings with all-mpnet-base-v2; Cosine similarity search; Distribution visualizations of paragraph distances (Figures B, C in report) |
| **IST718** | Suite of classifiers: BernoulliNB, AdaBoostClassifier, RandomForestClassifier, SGDClassifier, RidgeClassifier, OneVsRestClassifier; **Best result: Logistic Regression with TF-IDF + suffix features achieving 94% accuracy** |
| **IST691** | LLM evaluation (ChatGPT 4.0, Llama 3.3 70B, Gemma3 27B/12B, Mistral 7B); Fine-tuning with Hugging Face; Training/evaluation loss curves; Jaccard distance metrics |

**Predictive Model Details:**
- Text classification: Nomenclature vs Description vs Miscellaneous Exposition
- Semantic similarity: 768-dimensional vector embeddings
- Feature extraction: Structured JSON from unstructured text

**Key Deliverables:**
- IST718 Jupyter notebook with classifier comparisons
- IST664 embedding implementation
- IST691 fine-tuning code and loss visualizations

---

### Learning Goal 4: Use Programming Languages (R and Python)
**Python-centric implementation across all projects**

| Technology | Application |
|------------|-------------|
| **Python** | Primary language for all projects |
| **PySpark** | Distributed processing pipeline (IST718, IST769) |
| **Django** | Web framework for IST690 website |
| **React JS** | Frontend interactivity (IST690) |
| **scikit-learn** | Text classification (IST664, IST718) |
| **Hugging Face/Transformers** | LLM fine-tuning (IST691) |
| **sentence-transformers** | SBERT embeddings (IST664) |
| **pandas/numpy** | Data manipulation throughout |
| **Redis-py** | Embedding cache management |
| **CouchDB libraries** | Document store integration |

**Code Repositories:**
- https://github.com/piggyatbaqaqi/skol
- https://github.com/piggyatbaqaqi/dr-drafts-mycosearch

**Key Python Classes Developed:**
- `Taxon` class: Encapsulates nomenclature, descriptions, metadata
- `SKOL` class: Prepares taxon objects for embedding
- `SKOL_TAXA` class: CouchDB-based taxa loading for search

---

### Learning Goal 5: Communicate Insights to Broad Audiences
**Communication to technical and non-technical stakeholders**

| Deliverable | Audience | Format |
|-------------|----------|--------|
| **IST664 Team Report** | Technical/Academic | 16-page paper with code, diagrams, results |
| **IST718 Final Presentation** | Mixed audience | PowerPoint + video recording |
| **IST691 Conference Paper** | Academic | IEEE-style paper format |
| **IST690 Website** | General public, researchers | Interactive web application |
| **MASMC 2025 Poster** | Mycology community | Conference poster presentation |
| **About page** | Website visitors | Accessible explanation of SKOL |

**Key Communication Elements:**
- System architecture diagrams (Figures D, E in IST664)
- Flow diagrams showing data pipelines
- Color-coded search results for intuitive understanding
- User-facing documentation and README files

---

### Learning Goal 6: Apply Ethics in Data and Model Development
**Ethical considerations in open science**

| Ethical Principle | Application in SKOL |
|-------------------|---------------------|
| **Transparency** | Open source under GPLv3; all code publicly available on GitHub |
| **Open Access** | Uses only open access mycological literature; results freely available |
| **Reproducibility** | Documented pipelines, versioned models, clear provenance tracking |
| **Bias Awareness** | Acknowledged limitations: initial focus on English-language mycological literature; Latin translation challenges documented |
| **Privacy** | No personal data collected; focuses on published scientific literature |
| **Attribution** | Proper citation of source journals; links back to original publications |

**Ethical Challenges Addressed:**
- Model limitations clearly documented (overfitting in IST691, Latin support gaps)
- User feedback mechanisms built into website
- Collaborative development with proper attribution to all contributors

---

## 4. Synthesis: How SKOL Demonstrates Integrated Learning

The SKOL project is not just a collection of course assignments but a coherent system where each course contributed essential capabilities:

```
IST664 (NLP)           -> Semantic Search Foundation
        |
        v
IST718 (Big Data)      -> Scalable Classification Pipeline
        |
        v
IST769 (Data Mgmt)     -> Production Data Architecture
        |
        v
IST691 (Deep Learning) -> Feature Extraction for UI
        |
        v
IST690 (Independent)   -> Public Website Deployment
        |
        v
    synoptickeyof.life  -> Live Production System
```

**Integration Points:**
- Taxon class developed incrementally across projects
- MycoSearch forked and extended for domain-specific search
- Data flows from web scrapers through classification to embeddings to search

---

## 5. Areas of Strength and Challenge

### Strengths
- **NLP/Text Processing**: Strong foundation from IST664; comfortable with embeddings, transformers, text classification
- **System Integration**: Successfully connected multiple technologies (PySpark, CouchDB, Redis, Django)
- **Domain Expertise**: Mycological background enabled meaningful feature engineering

### Challenges
- **LLM Fine-tuning**: IST691 revealed overfitting with small training sets; need larger annotated corpus
- **Latin Language Support**: Many taxonomic descriptions include Latin; current models struggle
- **Scale**: Moving from 170 annotated issues to 1,000+ requires ongoing engineering

---

## 6. Lifelong Learning Plan

### Immediate Next Steps
- Expand training corpus for LLM fine-tuning
- Add web crawlers for continuous ingestion of new publications
- Implement decision tree UI for guided description building

### Continuing Education
- Follow developments in transformer architectures and domain-specific fine-tuning
- Engage with mycology community (Western PA Mushroom Club, MASMC conferences)
- Contribute to open source ML/NLP tools

### Career Application
- Apply data science skills to scientific research domains
- Continue developing SKOL as open-source contribution to mycology community

---

## 7. References

### Key Conceptual Works
- Reimers, N., Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Jang, A. et al. (2023). Mistral 7B
- Gisolfi, N. (2024). Dr Draft's state-of-the-art (SOTA) Literature Search

### Course Deliverables (Appendices)
1. **IST664**: IST664_Team3_Balasi_Murphy_Osuga_Yarroll.pdf
2. **IST691**: IST691_final_project_paper.pdf, mistral_transfer_learning.ipynb
3. **IST718**: IST718_Final_Report_FINAL.docx, IST_718_Final_Project_Classifier.ipynb
4. **IST769**: IST769 Final Project Proposal.pdf, ist769_skol.ipynb
5. **IST690**: IST 690 SKOL Website.pdf

---

## 8. Appendix Organization

```
Portfolio/
├── 01_Overview/
│   ├── overview.pdf
│   └── resume.pdf
├── 02_Written_Paper/
│   └── portfolio_paper.pdf
├── 03_IST664_NLP/
│   ├── README.txt
│   ├── IST664_Team3_Balasi_Murphy_Osuga_Yarroll.pdf
│   └── SKOL_presentation3.pptx
├── 04_IST691_Deep_Learning/
│   ├── README.txt
│   ├── IST691_final_project_paper.pdf
│   └── mistral_transfer_learning.ipynb
├── 05_IST718_Big_Data/
│   ├── README.txt
│   ├── IST718_Final_Report_FINAL.docx
│   └── IST_718_Final_Project_Classifier.ipynb
├── 06_IST769_Data_Management/
│   ├── README.txt
│   ├── IST769_Final_Project_Proposal.pdf
│   └── ist769_skol.ipynb
├── 07_IST690_Independent_Study/
│   ├── README.txt
│   └── IST_690_SKOL_Website.pdf
└── 08_Video_Presentation/
    └── portfolio_presentation.mp4
```

---

## Key Dates

- **Draft Due**: February 28, 2026
- **Final Portfolio Due**: March 28, 2026

---

## Video Presentation Outline (10 minutes)

1. **Introduction** (1 min): Personal background, why SKOL matters
2. **Learning Goal Highlights** (6 min): One slide per goal with project examples
3. **Live Demo** (2 min): Show synoptickeyof.life semantic search
4. **Conclusion** (1 min): Synthesis and future directions
