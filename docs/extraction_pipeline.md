# Treatment extraction — components, inspectors, and per-document pipelines

## Background

Two findings pushed the architecture from monolithic to compositional:

1. **The v3 negative result.** `production_v3_jats` (trained on 1 724
   JATS-converted docs) scored macro F1 **0.132** against
   `skol_golden_ann_hand_v2`. `production_v3_hand` (trained on 160
   hand-annotated PDF docs) scored **0.459** on the same gold. JATS
   training data did not improve classification of PDF-extracted
   docs — they're structurally too different at the line level
   (page headers, multi-column, OCR noise vs. clean XML structure).
   The cross-distribution training hypothesis failed.

2. **The plain-JATS revelation.** A sample of 10 000 docs from
   `skol_dev` showed that **77% of JATS-format docs lack TaxPub
   markup** (1 880 of 2 443). These are taxonomic articles from
   journals like MDPI's *J. of Fungi*, Frontiers, Mycoscience, and
   older Persoonia volumes. They have *some* structure — but not
   the structure the original extraction pipeline assumed. Worse,
   spot-checks of two such docs revealed **two different
   typographic conventions** for marking treatment sections:

   - **Mycoscience** uses paragraph-leading-keyword + colon:
     `<p>Etymology: In memoriam of Dr. …</p>`
   - **MDPI** uses numbered `<sec><title>` text:
     `<sec><title>3.2.3. Description</title>…</sec>`

   The labels are deterministic *if* you know where to look — but
   "where to look" depends on the publishing journal.

The first finding rules out cross-distribution-training-as-a-crutch.
The second finding rules out a clean two-path split. Together they
point at a different shape of solution: **let document properties
drive component selection**, instead of hardcoding paths.

The earlier draft of this design lived at `docs/extraction_paths.md`
under a two-path framing (XML vs. PDF). It has been retired because
the framing didn't survive contact with the corpus. This document
replaces it.

## Goals

1. **Composability.** Each extraction concern (OCR, section
   labeling, entity span detection, treatment assembly) is a
   self-contained component with declared inputs and outputs. New
   document types add inspectors + components, not paths.
2. **Property-driven dispatch.** A document's pipeline is *computed*
   from its observed properties, not selected from a fixed enum.
3. **Deterministic-first, model-second.** Where deterministic
   extraction is possible (XML elements, keyword conventions),
   prefer it. Only invoke trained models (classifier, CRF) for the
   parts deterministic extractors can't cover.
4. **Treatment-format neutrality.** Whether a Treatment originated
   from TaxPub XML, MDPI numbered titles, or PDF classifier output,
   the downstream consumers (embed, span layer for highlights,
   Django) see the same `Treatment` object shape.

## Non-goals

- **Not a generic NLP pipeline framework.** This is scoped to
  taxonomic-treatment extraction; we won't borrow spaCy or build a
  parallel general-purpose tool.
- **Not changing the `Treatment` schema** (per
  [treatment_architecture.md §Phase 5](treatment_architecture.md)).
  Components produce TaggedBlock-list or Span-list outputs that
  feed the existing `taxon.py` assembler.
- **Not retiring the YEDDA format on the PDF path.** Predicted
  YEDDA `.ann` attachments remain the artifact the v3/v4 classifiers
  emit and the hand annotators consume.

## Core abstraction

Three concepts:

```
┌───────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Inspectors   │ ─► │     Dispatcher   │ ─► │   Components    │
│  (cheap, pure │    │  selects + orders│    │ (deterministic  │
│   property-   │    │  components from │    │  extractors,    │
│   computing)  │    │  property set    │    │  detectors,     │
└───────────────┘    └──────────────────┘    │  models)        │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Treatment(s)   │
                                              └─────────────────┘
```

- **Inspectors** are cheap, pure, idempotent functions that read a
  document and return small named properties (`has_taxpub_markup`,
  `pdf_is_image_only`, `has_taxonomic_signal`, …).
- **Components** are operations on the document state. Each
  declares preconditions over properties + previous components'
  outputs, and what it produces (text, entity spans, section
  labels, treatments).
- **The dispatcher** runs inspectors, computes the property set,
  selects every component whose preconditions are satisfied,
  topologically orders them, executes, and merges their outputs.

What this *avoids*:

- **Path explosion.** No `if jats_taxpub: … elif plain_jats: …
  elif pdf: …`. New document types add components, not branches.
- **Hidden coupling.** Components declare what they need and what
  they produce, in a single registry. The dispatcher is the only
  place that knows about composition.
- **Wasted compute.** A component only runs when its preconditions
  hold. TaxPub extractor doesn't run on plain JATS; gnfinder
  doesn't run when XML already has scientific-name markup; OCR
  doesn't run on docs that already have a text layer.

## Inspectors (catalog)

Inspectors are *fast* (≤ tens of milliseconds per doc), *pure*
(no side effects), and *cached* per pipeline invocation. They form
the dependency boundary for component selection.

| Inspector | What it computes | Cost | Cache scope |
|---|---|---|---|
| `inspect_attachments` | `has_xml`, `has_pdf`, `has_plaintext`, `has_markdown` from the doc's `_attachments` keys | < 1 ms | per-doc; trivial |
| `inspect_xml_root` | `xml_format ∈ {taxpub, jats, other, none}`, `xml_namespace`, `is_well_formed` | ~ 5 ms | per-doc; cache parsed root |
| `inspect_taxpub_markup` | `has_taxpub_markup` — presence of `<tp:taxon-treatment>` elements | ~ 5 ms | requires `has_xml=True` |
| `inspect_jats_taxonomic_keywords` | `has_keyword_titled_sections` (`<sec><title>` matches taxonomic terms), `has_keyword_labeled_paragraphs` (`<p>` leading text matches `^(Etymology|Diagnosis|Holotype|…):`) | ~ 20 ms | requires `has_xml=True` |
| `inspect_plaintext_signal` | `has_taxonomic_signal` (sp. nov., binomial, MB-number present anywhere in body) | ~ 5 ms | requires `has_plaintext=True` |
| `inspect_pdf_text_layer` | `pdf_has_text_layer`, `pdf_is_image_only` — does the PDF carry extractable text? | ~ 50 ms | requires `has_pdf=True` |
| `inspect_markdown_structure` | `has_markdown_headers` (markdown OCR output's `# Heading` syntax) | ~ 5 ms | requires `has_markdown=True` |

Inspectors run in dependency order (the cheap ones first; the
expensive ones only when their preconditions hold). The property
set produced is a flat dict that components read.

## Components (catalog)

Components fall into four categories by what they produce:

1. **Text-producing** — OCR, plaintext extraction. Input: image or
   structured source. Output: plain text or markdown.
2. **Entity-span-producing** — gnfinder, particle detectors, XML
   scrape. Input: text or XML. Output: `List[Span]` (character
   offsets + label + source).
3. **Section-labeling** — XML extractors, keyword labelers, the
   classifier, the CRFs. Input: text + optional structural hints.
   Output: per-passage section labels (TaggedBlock list).
4. **Assembling** — the treatment assembler (state machine over
   labeled blocks). Input: TaggedBlock list. Output: `List[Treatment]`.

### Text-producing components

| Component | Requires | Produces | Cost | Notes |
|---|---|---|---|---|
| `ocr_tesseract` | `has_pdf && pdf_is_image_only` | `article.txt` attachment | High (seconds/doc) | The baseline OCR. Reasonable accuracy on clean scans, struggles with multi-column or low resolution. |
| `ocr_markdown_llm` | `has_pdf && pdf_is_image_only` | `article.md` attachment, optionally `article.txt` | High (seconds/doc, GPU) | A newer Markdown-emitting OCR system (e.g. Marker, Olmocr, Nougat, Mistral OCR). Output is structured markdown — headers, lists, tables — which feeds straight into `markdown_section_labeler`. The doc-level choice between this and `ocr_tesseract` is configurable; markdown OCR is the better default once a chosen system is validated on the corpus. |
| `plaintext_from_jats` | `has_xml && (xml_format == jats || xml_format == taxpub) && !has_plaintext` | `article.txt` attachment | Low | Already exists in `ingestors/extract_plaintext.py`. Listed here for completeness; if `has_plaintext=False` after the JATS-having inspector pass, this fills the gap. |

### Entity-span-producing components

| Component | Requires | Produces | Cost | Notes |
|---|---|---|---|---|
| `jats_metadata_extractor` | `has_xml` | bibliographic metadata (doi, authors, year, journal, references) | Low | Works on any JATS or TaxPub doc. |
| `jats_figure_extractor` | `has_xml` | `Figure-caption` Span list | Low | `<fig>` / `<caption>` elements. |
| `jats_scientific_name_extractor` | `has_xml` | `TaxonName` Span list | Low | TaxPub: `<tp:taxon-name>` + `<tn-part>`s. Plain JATS: `<italic>` filtered by Genus-species pattern + nomenclatural context. |
| `jats_mb_link_extractor` | `has_xml` | `MB-number` Span list | Low | `<ext-link>` with mycobank URL. Sparse; complemented by `mb_number_detector`. |
| `mb_number_detector` | `has_plaintext` | `MB-number` Span list | Low | Regex on plaintext. Catches inline mentions XML doesn't tag. |
| `gnfinder_taxonname_detector` | `has_plaintext` | `TaxonName` Span list | Medium (REST API) | Per [treatment_architecture.md §Phase 2](treatment_architecture.md). Use *only* when XML scientific-name extraction is unavailable or insufficient. |
| `gnparser_authorship_detector` | `has_plaintext` + TaxonName spans | `Author`, `Year` Span list | Medium (REST API) | Runs in 80-char window after each TaxonName. |
| `particle_detector` | `has_plaintext` | DOI, Page-ref, Fungarium-code Span list | Low | Regex + Redis-loaded fungarium codes. Per [treatment_architecture.md §Phase 2](treatment_architecture.md). |
| `page_header_detector` | `has_plaintext` | `Page-header` line annotations | Low | Heuristic per [page-header-detection.md](page-header-detection.md). PDFs only; JATS/markdown sources don't need it. |
| `section_header_detector` | `has_plaintext` | `section-header` Span list | Low | Regex over short, title-case-or-all-caps lines matching known section names ("Taxonomy", "Materials and methods", …). Per [treatment_architecture.md §Section-header blocks](treatment_architecture.md). |

### Section-labeling components

These produce per-passage section labels. The dispatcher selects
several; their outputs *merge* (see "Output merging" below) so the
deterministic extractors fill what they can and the model-based
labelers fill the rest.

| Component | Requires | Produces | Cost | Notes |
|---|---|---|---|---|
| `taxpub_treatment_extractor` | `has_taxpub_markup` | TaggedBlock list — full structural decomposition of each `<tp:taxon-treatment>` | Low | Reuses internals of `ingestors/jats_to_yedda.py`. Output is the YEDDA-equivalent TaggedBlock list, in memory; no on-disk YEDDA file. |
| `taxonomic_keyword_section_labeler` | `has_xml && (has_keyword_titled_sections \|\| has_keyword_labeled_paragraphs)` | TaggedBlock list — labels for `<sec><title>` and keyword-prefixed `<p>` elements | Low | Handles both Mycoscience-style (paragraph-leading `Etymology:`) and MDPI-style (`<title>3.2.3. Description</title>`) by normalising title text and looking up against a taxonomic-keyword dictionary. Leaves unlabeled `<p>` elements for `crf_treatment` to fill. |
| `markdown_section_labeler` | `has_markdown && has_markdown_headers` | TaggedBlock list — labels for `# Heading` / `## Sub-heading` constructs that match taxonomic keywords | Low | The markdown OCR equivalent of `taxonomic_keyword_section_labeler`. Same keyword dictionary; different host syntax. |
| `crf_layout` | `has_plaintext` | per-line layout labels (Page-header / Figure-caption / Table / Key / Bibliography / Index / ToC-entry / Other) | Medium (CPU-bound CRF Viterbi) | v4's Pass 1 (per [v4_classifier_plan.md](v4_classifier_plan.md)). PDF-only — for JATS-sourced plaintext, layout labels are either provided by `jats_*_extractor` outputs or vacuously empty. |
| `crf_treatment` | `has_plaintext` | per-line treatment labels (Nomenclature / Description / Diagnosis / Etymology / Materials-examined / Materials-and-methods / Type-designation / Biology / Phylogeny / New-combinations / Notes / Misc-exposition) | Medium (CPU-bound CRF Viterbi) | v4's Pass 2. **Fills gaps** left by deterministic labelers. On a fully-labeled plain-JATS doc this is a no-op; on a partially-labeled doc this labels the unlabeled regions; on a PDF doc this carries the full labeling load. |
| `classifier_logistic_v3` | `has_plaintext` | YEDDA `.ann` content + per-line treatment labels | High (Spark startup ~30 s + inference) | Today's production model. Retires when `crf_treatment` ships. Kept here for migration completeness. |

### Assembling components

| Component | Requires | Produces | Cost | Notes |
|---|---|---|---|---|
| `treatment_assembler` | Merged TaggedBlock list | `List[Treatment]` | Low | The state machine in `taxon.py::group_paragraphs()`, redesigned per [treatment_architecture.md §Phase 5](treatment_architecture.md). Section-header spans (from `section_header_detector` or XML) feed in as hard treatment boundaries; the rest is the existing logic. |

## Object decomposition

The components and inspectors listed above are described abstractly.
This section pins down the concrete interface classes, the catalog
infrastructure that registers and looks them up, and the
descriptor-versus-instance pattern that keeps the catalog cheap while
allowing components to hold heavyweight runtime state.

### Borrowed pattern: ngautonml's catalog

We adopt the catalog abstraction from
[autonlab/ngautonml/ngautonml/catalog/](https://gitlab.com/autonlab/ngautonml/-/tree/main/ngautonml/catalog):
`Catalog[T]` (abstract generic), `MemoryCatalog[T]` (in-memory
implementation with tag indices), and `CatalogElementMixin` (provides
`name` + `tags` properties for objects to be registered). The pattern
gives us:

- **Tag-indexed lookup.** `catalog.lookup_by_tag_and(category="section_labeler")`
  returns every component declaring that tag, in O(1) per tag-value.
- **Directory autoloading.** `catalog.load(Path)` walks a directory
  tree, imports every `.py` file (skipping `_test.py` and dunder
  files), and calls each module's `register(catalog, **kwargs)`
  function. Adding a new component = drop a file + write its `register`.
- **Descriptor / instance split.** The catalog stores cheap
  descriptors. A descriptor knows how to construct a heavier
  `*Instance` object that holds loaded models, REST clients, or
  Spark sessions.

ngautonml is Apache-2.0; **we copy `catalog.py`, `memory_catalog.py`,
and `catalog_element_mixin.py` into `skol_classifier/extraction/catalog/`**
with the copyright header preserved. The total is ~300 lines of code
with no runtime dependencies of its own. We do not import ngautonml
as a library because (a) its broader release cadence isn't aligned
with skol's, and (b) we'd pull in unrelated modules (algorithms,
executor, deciders). Once ngautonml's catalog becomes a standalone
PyPI package, we'll switch to importing it.

### Interface hierarchy

Two parallel hierarchies. **Inspectors** compute document properties;
they don't transform state. **Components** transform state — they
read attachments and properties, and produce text, entity spans,
section labels, or treatments.

```
CatalogElementMixin                  ← supplies name + tags

Inspector(CatalogElementMixin, ABC)            ← descriptor
    @abstractmethod
    requires: FrozenSet[str]                   ← property names needed first
    @abstractmethod
    inspect(doc: Doc, props: Dict[str, Any]) -> Dict[str, Any]

Component(CatalogElementMixin, ABC)            ← descriptor
    @abstractmethod
    requires_props: FrozenSet[str]             ← property names needed
    @abstractmethod
    requires_outputs: FrozenSet[str]           ← other components' outputs needed
    @abstractmethod
    produces_outputs: FrozenSet[str]           ← what this component contributes
    @abstractmethod
    preconditions(props: Dict[str, Any]) -> bool   ← runtime gate
    @abstractmethod
    instance_constructor: Type[ComponentInstance]

ComponentInstance(ABC)                         ← per-pipeline runtime object
    @abstractmethod
    run(state: PipelineState) -> None

Component subtypes (parallel to ngautonml's Algorithm subtypes):
    TextProducer(Component)              ← OCR, plaintext extraction
    EntityDetector(Component)            ← gnfinder, regex, XML scrape
    SectionLabeler(Component)            ← XML/keyword labelers + CRFs
    Assembler(Component)                 ← treatment_assembler
```

Why two layers (`Component` descriptor + `ComponentInstance`
executor)? The catalog gets populated at import time, before any
document is processed. We don't want every CRF model loaded into
memory just because its component is *registered* — only when it's
*selected and run*. The descriptor is cheap and shareable; the
instance is per-pipeline-run, may hold loaded models, REST clients,
or any other transient state.

```python
# Schematic life-cycle.
catalog: ComponentCatalog
catalog.load(Path("skol_classifier/extraction/components"))
# Catalog now contains ~18 cheap descriptors; no models loaded.

for doc in skol_dev:
    pipeline_state = PipelineState(doc=doc)
    selected = dispatcher.select(catalog, pipeline_state.props)
    for component in selected:
        instance = component.create_instance(config=pipeline_state.config)
        instance.run(pipeline_state)   # loads models lazily; mutates state
    pipeline_state.treatments        # the assembler's output
```

### Catalogs

Two parameterised catalogs:

```python
from skol_classifier.extraction.catalog import MemoryCatalog

InspectorCatalog = MemoryCatalog[Inspector]
ComponentCatalog = MemoryCatalog[Component]
```

`MemoryCatalog[T]` is the copy of ngautonml's class. It exposes:

- `register(obj: T, name: str = None, tags: Dict = None) -> str`
- `lookup_by_name(name: str) -> T`
- `lookup_by_tag_and(**tags) -> Dict[str, T]`
- `all_objects() -> Iterable[T]`
- `tagtypes -> Set[str]`, `tagvals(tagtype) -> Set[str]`
- `load(module_directory: Path)` — recursive autoloader

### Tag schema

Every `Component` declares the following tags in its `_tags`
class attribute:

| Tag | Values | Used by |
|---|---|---|
| `category` | `text_producer`, `entity_detector`, `section_labeler`, `assembler` | Dispatcher to find labelers vs detectors vs assemblers |
| `cost` | `low`, `medium`, `high` | Future cost-aware scheduler; right now just metadata |
| `source` | `skol_native`, `external` (REST), `model` (loaded ML weights) | Operational triage; "all external-call components" is a useful filter for outage handling |
| `produces` | one or more of: `text`, `treatment_labels`, `taxon_spans`, `mb_spans`, `page_header_lines`, `figure_spans`, `metadata`, `treatments` | Dispatcher dependency resolution |
| `requires_props` | property names this component reads from inspector output | Dispatcher precondition check |
| `requires_outputs` | output kinds this component reads from other components | Dispatcher dependency resolution |

`Inspector` tags are simpler:

| Tag | Values |
|---|---|
| `category` | `inspector` (always; lets us mix inspectors and components in one combined catalog if needed) |
| `cost` | `low`, `medium`, `high` |
| `produces` | one or more property names (overlapping with the property name set the inspector returns) |

The `requires` and `produces` tags are denormalised copies of the
component's `requires_props` / `requires_outputs` / `produces_outputs`
properties. They live in tags so the catalog can answer
"which components produce `taxon_spans`?" in O(1). The properties on
the class itself are the source of truth used by the dispatcher's
selection and ordering logic; the tags are a derived index.

### Registration

Mirroring ngautonml: every component module declares a
`register(catalog, **kwargs)` function. The catalog's `load(dir)`
walks the directory and calls each module's `register`.

Directory layout:

```
skol_classifier/extraction/
  __init__.py
  catalog/                       ← ngautonml catalog copy
    __init__.py
    catalog.py                   ← Catalog[T] abstract base
    memory_catalog.py            ← MemoryCatalog[T] implementation
    catalog_element_mixin.py     ← CatalogElementMixin
  interfaces.py                  ← Inspector / Component / ComponentInstance ABCs
  state.py                       ← PipelineState
  dispatcher.py                  ← selection + topological-sort + execution
  inspectors/
    __init__.py
    attachments.py
    xml_root.py
    taxpub_markup.py
    jats_taxonomic_keywords.py
    plaintext_signal.py
    pdf_text_layer.py
    markdown_structure.py
  components/
    __init__.py
    taxpub_treatment_extractor.py
    taxonomic_keyword_section_labeler.py
    markdown_section_labeler.py
    crf_layout.py
    crf_treatment.py
    classifier_logistic_v3.py
    ocr_tesseract.py
    ocr_markdown_llm.py
    jats_metadata.py
    jats_figure_extractor.py
    jats_scientific_name_extractor.py
    jats_mb_link_extractor.py
    mb_number_detector.py
    gnfinder_taxonname_detector.py
    gnparser_authorship_detector.py
    particle_detector.py
    page_header_detector.py
    section_header_detector.py
    treatment_assembler.py
```

Bootstrap:

```python
# skol_classifier/extraction/__init__.py
from pathlib import Path
from .catalog.memory_catalog import MemoryCatalog
from .interfaces import Inspector, Component

inspector_catalog: MemoryCatalog[Inspector] = MemoryCatalog()
inspector_catalog.load(Path(__file__).parent / "inspectors")

component_catalog: MemoryCatalog[Component] = MemoryCatalog()
component_catalog.load(Path(__file__).parent / "components")
```

### Worked examples

#### Example 1 — `AttachmentsInspector` (simplest case)

```python
# skol_classifier/extraction/inspectors/attachments.py
from typing import Any, Dict
from ..catalog import MemoryCatalog
from ..catalog.catalog_element_mixin import CatalogElementMixin
from ..interfaces import Inspector

class AttachmentsInspector(CatalogElementMixin, Inspector):
    _name = "attachments"
    _tags = {
        "category": "inspector",
        "cost": "low",
        "produces": ["has_xml", "has_pdf", "has_plaintext", "has_markdown"],
    }
    requires = frozenset()  # no prerequisites

    def inspect(self, doc, props):
        atts = doc.get("_attachments") or {}
        return {
            "has_xml": "article.xml" in atts,
            "has_pdf": "article.pdf" in atts,
            "has_plaintext": "article.txt" in atts,
            "has_markdown": "article.md" in atts,
        }


def register(catalog: MemoryCatalog[Inspector], **kwargs) -> None:
    inspector = AttachmentsInspector()
    catalog.register(inspector, inspector.name, inspector.tags)
```

#### Example 2 — `TaxpubTreatmentExtractor` (deterministic, stateless)

```python
# skol_classifier/extraction/components/taxpub_treatment_extractor.py
from typing import Any, Dict
from ingestors.jats_to_yedda import jats_xml_to_tagged_blocks
from ..catalog import MemoryCatalog
from ..catalog.catalog_element_mixin import CatalogElementMixin
from ..interfaces import Component, ComponentInstance, SectionLabeler
from ..state import PipelineState


class TaxpubTreatmentExtractorInstance(ComponentInstance):
    """Stateless executor — reads XML, emits TaggedBlock list."""

    def run(self, state: PipelineState) -> None:
        xml = state.get_attachment("article.xml").decode("utf-8")
        blocks = jats_xml_to_tagged_blocks(xml)
        state.add_section_labels(
            source="taxpub_treatment_extractor",
            blocks=blocks,
            priority=10,   # highest — structural markup wins
        )


class TaxpubTreatmentExtractor(CatalogElementMixin, SectionLabeler):
    _name = "taxpub_treatment_extractor"
    _tags = {
        "category": "section_labeler",
        "cost": "low",
        "source": "skol_native",
        "produces": ["treatment_labels", "taxon_spans", "figure_spans"],
        "requires_props": ["has_taxpub_markup"],
    }
    requires_props = frozenset({"has_taxpub_markup"})
    requires_outputs = frozenset()
    produces_outputs = frozenset({"treatment_labels", "taxon_spans", "figure_spans"})
    instance_constructor = TaxpubTreatmentExtractorInstance

    def preconditions(self, props: Dict[str, Any]) -> bool:
        return bool(props.get("has_taxpub_markup"))


def register(catalog: MemoryCatalog[Component], **kwargs) -> None:
    descriptor = TaxpubTreatmentExtractor()
    catalog.register(descriptor, descriptor.name, descriptor.tags)
```

#### Example 3 — `CrfTreatmentLabeler` (heavyweight, model-loaded Instance)

```python
# skol_classifier/extraction/components/crf_treatment.py
from typing import Any, Dict, Optional
import redis
from ..catalog import MemoryCatalog
from ..catalog.catalog_element_mixin import CatalogElementMixin
from ..interfaces import Component, ComponentInstance, SectionLabeler
from ..state import PipelineState
from skol_classifier.v4.crf_treatment import load_model_from_redis


class CrfTreatmentLabelerInstance(ComponentInstance):
    """Holds the loaded CRF model + SBERT cache reference for one
    pipeline invocation.  Constructed lazily by the descriptor; the
    same Instance can label many docs in a single batch."""

    def __init__(self, redis_client: redis.Redis, model_key: str):
        super().__init__()
        self._model = load_model_from_redis(redis_client, model_key)
        self._sbert_cache = redis_client  # same client, different key prefix

    def run(self, state: PipelineState) -> None:
        already_labeled_ranges = state.get_locked_ranges_for("treatment_labels")
        unlabeled_lines = state.get_unlabeled_lines(already_labeled_ranges)
        if not unlabeled_lines:
            return
        features = state.feature_assembler.build(unlabeled_lines, self._sbert_cache)
        predictions = self._model.decode(features)
        state.add_section_labels(
            source="crf_treatment",
            blocks=zip(unlabeled_lines, predictions),
            priority=4,   # lower than deterministic labelers
        )


class CrfTreatmentLabeler(CatalogElementMixin, SectionLabeler):
    _name = "crf_treatment"
    _tags = {
        "category": "section_labeler",
        "cost": "medium",
        "source": "model",
        "produces": ["treatment_labels"],
        "requires_props": ["has_plaintext"],
    }
    requires_props = frozenset({"has_plaintext"})
    requires_outputs = frozenset()  # gracefully no-op if no labels-to-fill
    produces_outputs = frozenset({"treatment_labels"})
    instance_constructor = CrfTreatmentLabelerInstance

    def preconditions(self, props: Dict[str, Any]) -> bool:
        return bool(props.get("has_plaintext"))


def register(catalog: MemoryCatalog[Component], **kwargs) -> None:
    descriptor = CrfTreatmentLabeler()
    catalog.register(descriptor, descriptor.name, descriptor.tags)
```

The descriptor knows the Redis key and how to load the model, but
*doesn't* load it. The Instance loads the model when constructed by
the dispatcher, just before its first `run()`. If the dispatcher
processes 1 000 docs in one invocation, it constructs *one*
`CrfTreatmentLabelerInstance` and calls `run()` 1 000 times.

#### Example 4 — `MBNumberDetector` (stateless regex)

```python
# skol_classifier/extraction/components/mb_number_detector.py
import re
from typing import Any, Dict, List
from ..catalog import MemoryCatalog
from ..catalog.catalog_element_mixin import CatalogElementMixin
from ..interfaces import Component, ComponentInstance, EntityDetector
from ..state import PipelineState
from ingestors.spans import Span


_MB_RE = re.compile(
    r"\bMB\s*(\d{5,7})\b|\bMycoBank\s+#?\s*(\d{5,7})\b",
    re.IGNORECASE,
)


class MBNumberDetectorInstance(ComponentInstance):
    def run(self, state: PipelineState) -> None:
        text = state.get_attachment("article.txt").decode("utf-8")
        spans: List[Span] = []
        for m in _MB_RE.finditer(text):
            value = m.group(1) or m.group(2)
            spans.append(Span(
                start=m.start(), end=m.end(),
                label="MB-number", text=m.group(),
                source="mb_number_detector", confidence=0.95,
                metadata={"value": value},
            ))
        state.add_spans(source="mb_number_detector", spans=spans)


class MBNumberDetector(CatalogElementMixin, EntityDetector):
    _name = "mb_number_detector"
    _tags = {
        "category": "entity_detector",
        "cost": "low",
        "source": "skol_native",
        "produces": ["mb_spans"],
        "requires_props": ["has_plaintext"],
    }
    requires_props = frozenset({"has_plaintext"})
    requires_outputs = frozenset()
    produces_outputs = frozenset({"mb_spans"})
    instance_constructor = MBNumberDetectorInstance

    def preconditions(self, props: Dict[str, Any]) -> bool:
        return bool(props.get("has_plaintext"))


def register(catalog: MemoryCatalog[Component], **kwargs) -> None:
    descriptor = MBNumberDetector()
    catalog.register(descriptor, descriptor.name, descriptor.tags)
```

### `PipelineState`

The mutable per-document object that components read and write.
Sketch:

```python
class PipelineState:
    doc: Dict[str, Any]
    props: Dict[str, Any]
    config: Dict[str, Any]

    # Component contributions, keyed by producing component name.
    _label_contributions: Dict[str, List[TaggedBlock]]
    _span_contributions: Dict[str, List[Span]]
    _label_priorities: Dict[str, int]

    # Cached attachments (lazy fetch from CouchDB).
    _attachment_cache: Dict[str, bytes]

    # Shared services injected by the dispatcher.
    feature_assembler: FeatureAssembler  # for CRF input
    redis_client: redis.Redis
    couchdb_client: Any

    def get_attachment(self, name: str) -> bytes: ...
    def add_section_labels(self, source: str, blocks: List[TaggedBlock],
                            priority: int) -> None: ...
    def add_spans(self, source: str, spans: List[Span]) -> None: ...
    def get_locked_ranges_for(self, output_kind: str) -> List[Tuple[int, int]]: ...
    def get_unlabeled_lines(self, locked: List[Tuple[int, int]]) -> List[Line]: ...

    @property
    def treatments(self) -> List[Treatment]:
        """Available after the assembler has run."""
```

The Assembler reads `_label_contributions` (merged per the
deterministic-first rule) and `_span_contributions` (resolved per
the specific-over-general rule) and produces `treatments`.

### Dispatcher implementation

The dispatcher's selection-and-execution algorithm sketched earlier
in the "Dispatcher" section becomes:

```python
class Dispatcher:
    def __init__(self,
                 inspectors: MemoryCatalog[Inspector],
                 components: MemoryCatalog[Component],
                 config: Dict[str, Any]):
        self._inspectors = inspectors
        self._components = components
        self._config = config

    def extract(self, doc: Dict[str, Any]) -> List[Treatment]:
        state = PipelineState(doc=doc, config=self._config)

        # Phase 1 — inspectors in dependency order.
        for inspector in self._topological(self._inspectors.all_objects()):
            if inspector.requires.issubset(state.props.keys()):
                state.props.update(inspector.inspect(doc, state.props))

        # Phase 2 — component selection.
        selected = [
            c for c in self._components.all_objects()
            if c.preconditions(state.props)
        ]

        # Phase 3 — topological sort by output-dependency.
        plan = self._sort_by_outputs(selected)

        # Phase 4 — execute. Components in the same stratum are
        # independent; running them concurrently is a future
        # optimisation.
        for stratum in plan:
            for descriptor in stratum:
                instance = descriptor.instance_constructor(
                    **self._instance_kwargs(descriptor),
                )
                instance.run(state)

        # Phase 5 — assembler runs last, consumes merged state.
        assembler = self._components.lookup_by_name("treatment_assembler")
        assembler.instance_constructor().run(state)
        return state.treatments

    def _topological(self, items): ...
    def _sort_by_outputs(self, selected: List[Component]) -> List[List[Component]]: ...
    def _instance_kwargs(self, descriptor: Component) -> Dict[str, Any]: ...
```

The dispatcher is the *only* code that knows about composition.
Every new component is invisible to the dispatcher until its `register`
runs at catalog load — and even then, the dispatcher treats it like
any other component. Adding a new document format is purely additive.

### Dispatcher subtleties

- **Components are *unaware* of each other.** A
  `taxpub_treatment_extractor` doesn't know `crf_treatment`
  exists. The dispatcher decides whether both run, in what order,
  and how their outputs merge.
- **Failure is local.** If `gnfinder` throws because the REST
  service is down, that component's output is empty; the rest of
  the pipeline runs unaffected. Errors are recorded in the
  pipeline result, not propagated.
- **Cost-aware execution is a future optimisation.** The first
  cut just runs components in dependency order. If components
  become expensive enough to warrant scheduling, the dispatcher
  can be extended with a cost model that reads the `cost` tag
  from each descriptor and reorders within a stratum.
- **Parallelism within a stratum.** Components in the same
  topological stratum have no dependencies on each other and can
  run concurrently. The first cut runs them serially; switching
  to `concurrent.futures.ThreadPoolExecutor` is a one-line change
  once we have evidence it pays off.

### Tests

Each component and inspector gets a peer test file
(`*_test.py`). The component tests cover:

- `preconditions(props)` returns True/False for representative
  property dicts.
- `run(state)` on a fixture state produces the expected
  contribution to `state._label_contributions` /
  `state._span_contributions`.
- `register(catalog)` registers exactly one descriptor under the
  expected name.

Catalog-level tests cover the dispatcher: given a known property
set, the selection picks the expected component set; the topo-sort
returns a legal execution order.

Component-instance tests focus on the heavyweight cases
(CrfTreatmentLabelerInstance) — model loading, prediction shape,
caching.

## Dispatcher

The concrete dispatcher class lives in "Object decomposition §
Dispatcher implementation" above; its subtleties (component
independence, local failure, cost-aware execution, intra-stratum
parallelism) are listed there too. This section header is preserved
as a navigation anchor for readers landing from external links.

## Output merging

Two merge surfaces: section labels and entity spans.

### Section labels — "deterministic-first" rule

The labelers' outputs cover (possibly overlapping) ranges of the
document. Merge precedence, highest to lowest:

1. `taxpub_treatment_extractor` (highest: structural markup)
2. `taxonomic_keyword_section_labeler` (deterministic typographic
   convention)
3. `markdown_section_labeler` (deterministic markdown header
   convention)
4. `crf_treatment` (model-based; fills gaps)
5. `classifier_logistic_v3` (model-based; fills gaps; retiring)

A higher-priority labeler's contribution **locks** a range; lower-
priority labelers only get to label the *complement*. So
`crf_treatment` runs on the unlabeled remainder of a partially-
labeled plain-JATS doc; it never overrides a label that a
deterministic labeler emitted.

Within the same priority level, labels are required to be
range-disjoint by construction (a single XML walker can't emit
overlapping labels for the same paragraph), so no further conflict
arises in practice.

### Entity spans — "specific-over-general" rule

Multiple span sources (XML scrape, gnfinder, regex detectors) can
emit Spans over overlapping character ranges. Conflict resolution:

1. **Shorter (more specific) span wins** over a longer span that
   contains it. Example: `<tp:taxon-name>` annotating "Mycena
   cristinae" beats a gnfinder span over "Mycena cristinae J.S.
   Oliveira sp. nov." that includes the authorship.
2. **Higher confidence wins** on equal length. XML-derived spans
   have `confidence=1.0`; gnfinder typically `0.8-0.95`;
   particle_detector inside `Materials-examined` `0.9` else `0.6`.
3. **Higher-priority source wins** on ties (the same source order
   as labelers above).

The merged Span list is what populates Treatment's `*_spans`
fields after assembly.

## Concrete pipelines for known document types

Each example shows: properties → selected components → execution.

### TaxPub JATS (Pensoft articles, ~23% of JATS docs)

Properties: `has_xml=True, xml_format=taxpub, has_taxpub_markup=True`.

Selected components (in dependency order):

```
[ jats_metadata_extractor,
  jats_figure_extractor,
  jats_scientific_name_extractor,
  jats_mb_link_extractor,
  taxpub_treatment_extractor,
  section_header_detector,         # for plaintext if present
  particle_detector,               # fills in fungarium codes etc.
  treatment_assembler ]
```

Not selected: `crf_treatment` (the taxpub extractor covers
everything; the unlabeled-remainder is empty). `page_header_detector`
and `crf_layout` (no PDF layout artefacts in XML).

### Plain JATS with paragraph-leading keywords (Mycoscience, older Persoonia)

Properties: `has_xml=True, xml_format=jats,
has_taxpub_markup=False, has_keyword_labeled_paragraphs=True`.

Selected:

```
[ jats_metadata_extractor,
  jats_figure_extractor,
  jats_scientific_name_extractor,  # italic + Genus-species pattern
  taxonomic_keyword_section_labeler,
  mb_number_detector,              # regex on plaintext
  gnfinder_taxonname_detector,     # backup for the parts XML missed
  crf_treatment,                   # fills paragraphs without a keyword label
  treatment_assembler ]
```

`crf_treatment` consumes a sequence of `<p>` elements where some
are already labeled (by the keyword labeler) and the rest carry no
label. The CRF predicts labels only for the unlabeled ones; the
deterministic-first merge rule preserves the keyword labels.

### Plain JATS with numbered `<sec><title>` keywords (MDPI, Frontiers)

Properties: same as above but `has_keyword_titled_sections=True`
instead of (or in addition to) `has_keyword_labeled_paragraphs`.

Selected components are identical to the previous case. The
`taxonomic_keyword_section_labeler` component handles both styles
internally (normalises title text: strips leading section numbers,
then matches against the keyword dictionary).

### PDF with text layer (most hand-curated docs)

Properties: `has_plaintext=True, has_pdf=True,
pdf_has_text_layer=True, has_xml=False`.

Selected:

```
[ page_header_detector,
  section_header_detector,
  mb_number_detector,
  gnfinder_taxonname_detector,
  gnparser_authorship_detector,
  particle_detector,
  crf_layout,                      # Pass 1: removes Page-header, etc.
  crf_treatment,                   # Pass 2: labels the remainder
  treatment_assembler ]
```

This is what v4 is for. The two CRFs collaborate as designed in
[v4_classifier_plan.md](v4_classifier_plan.md).

### Image-only PDF (no text layer)

Properties: `has_pdf=True, pdf_is_image_only=True,
has_plaintext=False`.

Selected:

```
[ ocr_markdown_llm  OR  ocr_tesseract,    # configurable choice
  # ...the chosen OCR produces text/markdown.
  # The dispatcher then *re-runs* inspectors on the now-populated
  # attachments and proceeds with the appropriate downstream
  # pipeline (PDF-with-text or markdown).
  ... ]
```

OCR is **just another component** that runs at the front of the
pipeline. After it succeeds, the dispatcher re-evaluates
properties and continues. If `ocr_markdown_llm` produces a
markdown attachment, the doc now has `has_markdown=True` and
`markdown_section_labeler` becomes selectable.

### Markdown OCR output (image-PDF that went through markdown OCR)

Properties: `has_markdown=True, has_markdown_headers=True`.

Selected:

```
[ markdown_section_labeler,
  mb_number_detector,
  gnfinder_taxonname_detector,
  gnparser_authorship_detector,
  particle_detector,
  crf_treatment,                   # fills paragraphs the headers didn't label
  treatment_assembler ]
```

Markdown is structurally close to plain JATS with numbered titles
— the section labels live in `# Description` / `## Diagnosis`
headers. The keyword dictionary is shared with the JATS labelers.

### Future: LaTeX articles

Hypothetical properties: `has_latex=True, has_latex_section_markup=True`.

Pipeline shape would mirror plain JATS / markdown:

```
[ latex_metadata_extractor,
  latex_section_labeler,           # \section{Description} → label
  ...                              # entity detectors as above
  crf_treatment,
  treatment_assembler ]
```

No structural changes — add a `latex_inspector`, a
`latex_section_labeler`, and a `latex_metadata_extractor`. The CRF
and assembler don't know about LaTeX.

### Future: arbitrary plaintext / HTML / ePub

Each new format adds inspectors + labelers + entity extractors as
needed. The CRF, particle detectors, gnfinder, gnparser, and
treatment_assembler are all format-agnostic — they reuse without
modification.

## Migration sequence

The current monolithic pipeline goes:

```
doc → predict_classifier → article.txt.ann → extract_treatments → Treatment
```

Migration to the component model lands in three commits, each
shippable on its own:

### Commit 1 — Scaffold the dispatcher + two components

| File | Status |
|---|---|
| `skol_classifier/extraction/inspectors.py` | New: 7 inspectors |
| `skol_classifier/extraction/dispatcher.py` | New: registry, selection, ordering |
| `skol_classifier/extraction/components/taxpub.py` | New: wraps `jats_to_yedda.jats_xml_to_tagged_blocks` |
| `skol_classifier/extraction/components/classifier_logistic.py` | New: wraps the existing classifier+YEDDA-read flow |
| `bin/extract_treatments_to_couchdb.py` | Modified: call dispatcher instead of unconditional classifier path |

Test plan: feed in a known TaxPub doc and a known PDF doc; assert
the dispatcher selects `taxpub_treatment_extractor` for the first
and `classifier_logistic_v3` for the second; assert resulting
`Treatment` objects match what the monolithic pipeline produces
today (field-equality, not bit-equality).

After this commit, the monolithic flow is replaced by a dispatcher
flow with identical externals. No new functionality; just the
plumbing change.

### Commit 2 — Add `taxonomic_keyword_section_labeler`

Adds the component, the dictionary of taxonomic keywords (with
canonical-form mapping), and the inspector that detects when it
should be selected. Adds the deterministic-first merge logic.

After this commit, plain-JATS docs get partial labels from the
keyword convention, with `classifier_logistic_v3` filling the
remainder.

### Commit 3 — Replace `classifier_logistic_v3` with v4 CRFs

Adds `crf_layout` and `crf_treatment` as components. Retires
`classifier_logistic_v3` (kept in the registry but no longer
selected when the CRFs are available).

After this commit, all documents go through CRF-based labeling
where a model is needed; the deterministic-first merge rule still
preserves XML and keyword-derived labels.

### Optional Commit 4 — OCR sub-pipeline

Adds `ocr_tesseract` and `ocr_markdown_llm` components. Adds
`inspect_pdf_text_layer` and `inspect_markdown_structure`
inspectors. Image-only PDFs become extractable.

### Optional Commit 5 — Markdown OCR labeling

Adds `markdown_section_labeler`. Image PDFs OCR'd to markdown get
the same deterministic labeling treatment as plain-JATS docs.

Each commit is independently rollback-able. Each adds capability
without breaking the previous shape.

## Open questions

- **Component cost model.** Right now the dispatcher orders by
  dependency only. A cost model (component declares its expected
  wall-clock + memory budget; dispatcher prefers cheap-equivalents
  when multiple components could satisfy the same precondition)
  would matter once we have 15+ components. Defer until concrete.
- **Cross-doc caching.** SBERT embeddings, gnfinder API calls,
  fungarium-code lookups — all are per-line- or per-string-
  uniqueable. Keyed by content hash, they cache trivially across
  docs. Decided storage in [v4_classifier_plan.md](v4_classifier_plan.md);
  this doc just declares which components benefit.
- **Markdown OCR system choice.** Marker (PyPI: marker-pdf),
  Olmocr, Nougat, Mistral OCR are candidates. Each emits markdown
  with different fidelity to taxonomic conventions (headers,
  inline italics for species names, etc.). The choice belongs in a
  separate component-validation step, not here.
- **Per-doc inspector caching.** Inspectors are cheap, but on a
  full-corpus re-run they'd compute the same properties redundantly.
  Worth persisting to CouchDB as a doc-level `pipeline_props` field
  after the first dispatcher run? Probably yes — small data, helps
  pipeline-debugging. Defer to Commit 1's implementation.
- **Deterministic-first vs. confidence-weighted.** Today's rule is
  "deterministic labelers lock their ranges; models fill gaps." A
  future variant is "everything contributes labels with confidence;
  highest-confidence-per-range wins, with deterministic-labeler
  confidence pinned to 1.0." Functionally equivalent until we add
  a deterministic labeler with imperfect precision (e.g., a
  taxonomic keyword the labeler matches but the author meant
  something else). Defer until needed.

## Rollback

Each commit is reversible. Reverting:

- **Commit 1** reverts the dispatcher; restore the monolithic
  `extract_treatments_to_couchdb.py` from git history.
- **Commit 2** removes the keyword labeler; plain-JATS docs go
  back to fully-classifier-labeled.
- **Commit 3** removes the v4 CRFs; the dispatcher falls back to
  `classifier_logistic_v3` (still in the registry, just no longer
  selected by default).
- **Commit 4/5** are pure-additions; revert removes the OCR /
  markdown surfaces without affecting anything else.

## What this document supersedes

- **`docs/extraction_paths.md`** (now retired). That document's
  framing — "two paths, XML and PDF" — didn't survive the
  plain-JATS finding. The relevant content from it has been
  recast in component-and-property terms here.

## What this document does NOT supersede

- **[treatment_architecture.md](treatment_architecture.md)** — still
  the source of truth for the Layer 1 (YEDDA) / Layer 2 (Span)
  data model, the `Treatment` schema, and Phase 1–6 sequencing.
  This document refines *how* that architecture is implemented in
  the pipeline; the data shapes are unchanged.
- **[v4_classifier_plan.md](v4_classifier_plan.md)** — still the
  spec for the CRF Pass 1 + Pass 2 model. This document says
  *where* those CRFs slot in as components; the training,
  features, and evaluation details remain in the v4 plan.
- **[page-header-detection.md](page-header-detection.md)** — still
  the design for the page-header heuristic. This document just
  names `page_header_detector` as the component wrapping it.
