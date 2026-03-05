# SKOL System Architecture

<!-- Render: cd IST782 && make -->

## Color Legend

| Color | Course | Contribution |
|-------|--------|-------------|
| Gray (#BBBBBB) | External | Journal web sites (not my work) |
| Blue (#4477AA) | IST 718 Big Data Analytics | PySpark classifier pipeline |
| Cyan (#66CCEE) | IST 664 NLP | Taxon class, SBERT embeddings, MycoSearch |
| Gold (#CCBB44) | IST 691 Deep Learning | Mistral JSON feature extraction |
| Coral (#EE6677) | IST 769 Adv. Database Mgmt | CouchDB, Redis, Neo4j, pipeline orchestration |
| Purple (#AA3377) | IST 690 Independent Study | Ingesters, web app, constrained decoding, deployment |

Palette: Paul Tol "bright" qualitative scheme (colorblind-safe, print/projection robust).

## Architecture Diagram

```mermaid
flowchart LR
    %% ── Paul Tol bright qualitative palette ─────────────────
    classDef external fill:#BBBBBB,stroke:#888888,color:#000000
    classDef ist718  fill:#4477AA,stroke:#2E5C8A,color:#FFFFFF
    classDef ist664  fill:#66CCEE,stroke:#3DA8CC,color:#000000
    classDef ist691  fill:#CCBB44,stroke:#A89930,color:#000000
    classDef ist769  fill:#EE6677,stroke:#CC4455,color:#000000
    classDef ist690  fill:#AA3377,stroke:#882255,color:#FFFFFF


    %% ── Legend ──────────────────────────────────────────────────
    subgraph Legend[" "]
        direction LR
        Lext["External"]:::external ~~~
        L718["IST 718"]:::ist718 ~~~
        L664["IST 664"]:::ist664 ~~~
        L691["IST 691"]:::ist691 ~~~
        L769["IST 769"]:::ist769 ~~~
        L690["IST 690"]:::ist690
    end

    %% ── Taxon Construction ──────────────────────────────────
    TAXON["Build Taxons<br/>(Nomenclature + Description)"]:::ist664

    %% ── Embedding Branch ────────────────────────────────────
    SBERT["SBERT Embeddings<br/>(all-mpnet-base-v2, 768-dim)"]:::ist664

    REDIS[("Redis<br/>Embeddings, Models")]:::ist769

    CLUSTER["Hierarchical Clustering<br/>(Agglomerative)"]:::ist769

    NEO4J[("Neo4j<br/>Pseudoclades")]:::ist769

    %% ── JSON Branch ─────────────────────────────────────────
    JSON_EX["JSON Feature Extraction<br/>(Mistral SLM)"]:::ist691

    CONSTRAIN["Constrained Decoding<br/>(Outlines + JSON schema)"]:::ist690

    VOCAB["Vocabulary Normalization<br/>(Ontology similarity)"]:::ist690

    COUCH_TAXA[("CouchDB<br/>Taxa, JSON Features")]:::ist769

    %% ── Web Application ─────────────────────────────────────
    WEBAPP["Django / React Web App<br/>synoptickeyof.life"]:::ist690

    %% ── User-facing Features ────────────────────────────────
    SEARCH["Description Search<br/>(Cosine Similarity)"]:::ist664

    DT_FEAT["Feature Suggestions<br/>(Decision Trees)"]:::ist690

    COLLECT["Collection Manager,<br/>Discussion Threads"]:::ist690

    %% ── Flow ────────────────────────────────────────────────
    subgraph Flow[" "]
        direction LR
	TAXON --> SBERT
	TAXON --> JSON_EX

	SBERT --> REDIS
	SBERT --> CLUSTER
	CLUSTER --> NEO4J

	JSON_EX --> CONSTRAIN
	CONSTRAIN --> VOCAB
	VOCAB --> COUCH_TAXA

	REDIS --> WEBAPP
	COUCH_TAXA --> WEBAPP
	NEO4J -.-> WEBAPP

	WEBAPP --> SEARCH
	WEBAPP --> DT_FEAT
	WEBAPP --> COLLECT
    end
```
