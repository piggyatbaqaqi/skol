# Segment-detector scope: which treatment sections get feature extraction?

Motivating question from the 2026-07-08 batch-2 review session: the
brat annotation surface (and the segment classifier trained on it)
covers only `description` and `diagnosis`.  Every other Treatment
section — `materials_examined`, `type_designation`, `etymology`,
`distribution`, `biology`, `notes`, `key`, `figure_captions` — is
excluded.  Is that scoping correct, or would broadening the segment
detector to cover more sections improve the feature-extraction
pipeline?

## Why description / diagnosis only, structurally

These two sections carry the **morphological character-value pairs**
that make up the prose payload of a taxonomic paper.  "Pileus 3–5 cm,
brown at maturity" is one feature-segment.  "Ascospores 13.5–17 × 5–8
μm, brown, ellipsoid" is another.  The segment classifier's job is to
lift those pairs into structured feature vectors.

Other sections carry different content types, with different
ontologies and different downstream users:

| Section              | Content type                                | Ontology                                                                                       |
|----------------------|---------------------------------------------|------------------------------------------------------------------------------------------------|
| Description / Diagnosis | Morphological characters                 | Anatomical part × property × measurement — rich, open-vocabulary                               |
| Materials examined   | Collection records                          | Country, locality, coordinates, date, collector, collection#, herbarium, host, type-status — fixed ~10 slots |
| Type designation     | Same as Materials examined but for the holotype specifically | Same fixed slots                                                                              |
| Etymology            | Name-derivation prose                       | Usually one sentence — little to segment                                                       |
| Distribution         | Geographic entities                         | Country / state list                                                                           |
| Biology              | Host + substrate + ecology + phenology      | Mixed                                                                                          |
| Notes                | Freeform discussion + comparative claims    | Unbounded                                                                                      |

## Would broadening dilute the model?

Yes — if we naively add `Locality`, `Collector`, `Herbarium-code` as
sibling segment classes to `Pileus`, `Ascospores`, `Asci`, the CRF
learns from very different feature contexts per class and each new
class gets fewer training examples.  Class imbalance would tilt
harder toward Description because Materials-examined paragraphs are
roughly an order of magnitude shorter than description paragraphs.

But that's a strawman — nobody's arguing "just cram everything into
one model."  The real question is whether **separate, section-specific
detectors** are worth building.

## Where the line should sit

**Description / Diagnosis**: keep the current segment classifier
scoped as-is.  Morphological character extraction is the flagship.
Don't dilute.

**Materials examined + Type designation**: high downstream value,
structurally regular, worth building **separately**.  Fields are
country / state / locality / coordinates / date / collector /
collection# / herbarium / type-status.  This is closer to a Named
Entity Recognition task than a segment-classifier task — the context
features that identify `Pileus` don't help identify `MEXU 26354`.
Even a rule-based first pass (grep for coordinates + date +
herbarium codes) would capture ~70% of the value.  A dedicated small
NER model gets to 90%+.

**Distribution**: pure geographic NER.  Off-the-shelf tools apply.

**Etymology**: usually one sentence.  Not worth the annotation cost
of a dedicated extractor.

**Biology / Notes**: mixed prose.  Semantic topic tagging (host,
substrate, ecology, phenology, comparative claims), not segment
extraction.  Different tool again.

## Practical implication for brat scope

Even if we build Materials-examined and Distribution extractors, they
wouldn't share a brat annotation pass with description / diagnosis.
Different segment ontologies → different `.conf` files → different
reviewer instructions → different validation metrics.  So brat
annotation stays scoped to description / diagnosis regardless.

The pipeline downstream of Claude's `llm_annotate_features` could
**grow additional passes** targeting other sections with their own
extractors.  That's independent of what brat sees.

## Bottom line

Don't broaden the current segment classifier.  Do consider
Materials-examined and Distribution as separate extractors — they're
where the next-largest scientific payoff sits (species distribution
maps, collector networks, temporal analysis of range shifts).
Different track, different tools, different training data.
