# SKOL Data Technologies

## SKOL (Core Pipeline)

### Databases & Data Stores

- **CouchDB** — document store for articles, taxa, PDFs, annotations, and JSON feature structures
- **Redis** — in-memory cache for SBERT embeddings and trained classifier models
- **Neo4j** — graph database for hierarchical taxonomic clustering
- **Parquet** (via PyArrow) — intermediate columnar data files

### Distributed Computing

- **PySpark** / Spark MLlib / Spark NLP — distributed text classification and processing

### Deep Learning & NLP

- **Sentence Transformers** (SBERT, all-mpnet-base-v2) — 768-dim semantic embeddings
- **Hugging Face Transformers** + **PEFT** — fine-tuning Mistral 7B with LoRA
- **Outlines** — constrained decoding with JSON schema for structured LLM output
- **TensorFlow / PyTorch** — RNN/LSTM models (with CUDA/GPU support)
- **Scikit-learn** — Logistic Regression, Random Forest, GBT classifiers

### Text Extraction & Data Ingestion

- **PyMuPDF / pymupdf4llm** — PDF text extraction
- **Pytesseract** — OCR fallback for scanned documents
- **BeautifulSoup4** — HTML/XML parsing for web scraping
- **feedparser** — RSS feed ingestion (Ingenta Connect, Pensoft)
- **Habanero** — Crossref API for bibliography metadata
- **bibtexparser** — BibTeX citation parsing
- **Pronto** — OBO ontology parsing (PATO, FAO)

### Data Science Stack

- NumPy, Pandas, SciPy, matplotlib

### Infrastructure

- Docker Compose (CouchDB, Redis, Neo4j services)

---

## SKOL Django (Web Application)

### Web Framework

- **Django** — backend web framework
- **Django REST Framework** — REST API layer
- **React** — frontend (separate JS build)

### Databases

- **SQLite / PostgreSQL** — Django ORM for user accounts, collections, metadata
- **CouchDB** — reads annotated documents and taxa from core pipeline
- **Redis** — embedding vectors and model caching (shared with core)

### Authentication

- **django-allauth** — social auth (GitHub, Google, ORCID)
- **PyJWT** — JWT token handling

### Integration

- **dr-drafts-mycosearch** — custom package providing embedding computation and semantic search
- **skol core modules** — text classification and taxonomy extraction

---

## Key Architecture Pattern

CouchDB and Redis are the shared data layer between the two components — the
core pipeline writes documents and embeddings, and the Django app reads them
for search and display.
