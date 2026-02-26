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
flowchart TD
    %% ── Paul Tol bright qualitative palette ─────────────────
    classDef external fill:#BBBBBB,stroke:#888888,color:#000000
    classDef ist718  fill:#4477AA,stroke:#2E5C8A,color:#FFFFFF
    classDef ist664  fill:#66CCEE,stroke:#3DA8CC,color:#000000
    classDef ist691  fill:#CCBB44,stroke:#A89930,color:#000000
    classDef ist769  fill:#EE6677,stroke:#CC4455,color:#000000
    classDef ist690  fill:#AA3377,stroke:#882255,color:#FFFFFF
    %% ── Legend (vertical row down) ────────────
    subgraph Legend["Legend"]
        direction TB
        Lext["External"]:::external ~~~
        L718["IST 718<br/>Big Data Analytics"]:::ist718 ~~~
        L664["IST 664<br/>NLP"]:::ist664 ~~~
        L691["IST 691<br/>Deep Learning"]:::ist691 ~~~
        L769["IST 769<br/>Adv. Database Mgmt"]:::ist769 ~~~
        L690["IST 690<br/>Independent Study"]:::ist690
    end
    style Legend fill:#FFFFFF,stroke:#CCCCCC
```
