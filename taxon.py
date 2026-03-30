"""Represent Nomenclature paragraphs and matching treatment sections."""

import itertools
from typing import Any, Dict, Iterable, Iterator, List, Optional

from paragraph import Paragraph
from label import Label
from line import Line


def get_ingest_field(
    record: Dict[str, Any], *keys: str, default: Any = None
) -> Any:
    """
    Get a field from a taxa record's ingest data.

    Uses canonical ingest field names:
    - '_id' for document ID
    - 'url' for human URL
    - 'pdf_url' for PDF URL

    Args:
        record: Taxa record dict with 'ingest' field
        *keys: Field path (e.g., 'url', '_id', 'pdf_url')
        default: Value to return if field not found

    Returns:
        Field value or default

    Examples:
        get_ingest_field(record, '_id')      # Gets ingest._id
        get_ingest_field(record, 'url')      # Gets ingest.url
        get_ingest_field(record, 'pdf_url')  # Gets ingest.pdf_url
    """
    ingest = record.get("ingest")
    if ingest is None:
        return default

    result: Any = ingest
    for key in keys:
        if not isinstance(result, dict):
            return default
        result = result.get(key)
        if result is None:
            return default

    return result


# ---------------------------------------------------------------------------
# Label → row field mappings
# ---------------------------------------------------------------------------

# Maps YEDDA tag string → flat field name in as_row() output.
_LABEL_TO_FIELD: Dict[str, str] = {
    "Description": "description",
    "Diagnosis": "diagnosis",
    "Etymology": "etymology",
    "Distribution": "distribution",
    "Materials-examined": "materials_examined",
    "Type-designation": "type_designation",
    "Biology": "biology",
    "Notes": "notes",
    "Key": "key",
    "Figure-caption": "figure_captions",
}

# Maps YEDDA tag string → spans field name in as_row() output.
_LABEL_TO_SPANS_FIELD: Dict[str, str] = {
    "Nomenclature": "nomenclature_spans",
    "Description": "description_spans",
    "Diagnosis": "diagnosis_spans",
    "Etymology": "etymology_spans",
    "Distribution": "distribution_spans",
    "Materials-examined": "materials_examined_spans",
    "Type-designation": "type_designation_spans",
    "Biology": "biology_spans",
    "Notes": "notes_spans",
    "Figure-caption": "figure_caption_spans",
}


class Taxon(object):
    FIELDNAMES = [
        "serial_number",
        "filename",
        "human_url",
        "pdf_url",
        "label",
        "paragraph_number",
        "pdf_page",
        "pdf_label",
        "empirical_page_number",
        "body",
    ]
    # Maximum consecutive Misc-exposition blocks before treatment is abandoned.
    MISC_GAP_LIMIT: int = 4

    _nomenclatures: List[Paragraph]
    # All non-nomenclature paragraphs in document order (for dictionaries()).
    _section_paragraphs: List[Paragraph]
    # label string → paragraphs, for per-section field access in as_row().
    _sections: Dict[str, List[Paragraph]]
    _serial: int = 0

    def __init__(self) -> None:
        self.__class__._serial += 1
        self._serial = self.__class__._serial
        self.reset()

    def __repr__(self) -> str:
        return repr(list(self.dictionaries()))

    def reset(self) -> None:
        self._nomenclatures = []
        self._section_paragraphs = []
        self._sections = {}

    def add_nomenclature(self, pp: Paragraph) -> None:
        self._nomenclatures.append(pp)

    def add_section(self, label: str, pp: Paragraph) -> None:
        """Add a treatment-section paragraph under the given label string."""
        self._section_paragraphs.append(pp)
        self._sections.setdefault(label, []).append(pp)

    def has_nomenclature(self) -> bool:
        return bool(self._nomenclatures)

    def has_section(self) -> bool:
        """Return True if at least one treatment-section paragraph is present.
        """
        return bool(self._section_paragraphs)

    def has_description(self) -> bool:
        return bool(self._sections.get("Description"))

    def doc_id(self) -> Optional[str]:
        """Return doc_id from the first nomenclature paragraph, if any."""
        if self._nomenclatures:
            first_line = self._nomenclatures[0].first_line
            return first_line.doc_id if first_line else None
        return None

    def dictionaries(self) -> Iterator[Dict[str, str]]:
        for pp in itertools.chain(
            self._nomenclatures, self._section_paragraphs
        ):
            d = pp.as_dict()
            d["serial_number"] = str(self._serial)
            yield d

    def human_url(self) -> Optional[str]:
        """Return human_url from the first nomenclature paragraph, if any."""
        row = self.as_row()
        ingest = row.get("ingest")
        if isinstance(ingest, dict):
            return ingest.get("url")
        return None

    def as_row(self) -> Dict[str, Any]:
        """Convert this Taxon to a dictionary suitable for output.

        Flat section fields (None when absent):
          description, diagnosis, etymology, distribution,
          materials_examined, type_designation, biology, notes, key,
          figure_captions

        Span fields (empty list when absent):
          nomenclature_spans, description_spans, diagnosis_spans,
          etymology_spans, distribution_spans, materials_examined_spans,
          type_designation_spans, biology_spans, notes_spans,
          figure_caption_spans
        """

        def span_to_string_dict(span_dict: Dict[str, Any]) -> Dict[str, str]:
            """Convert span dict values to strings for MapType schema."""
            return {
                k: str(v) if v is not None else "None"
                for k, v in span_dict.items()
            }

        pp = self._nomenclatures[0]
        first_line = pp.first_line
        assert (
            first_line is not None
        ), "Nomenclature paragraph must have at least one line"

        # For synthetic stub nomenclatures, fall back to first section para.
        ingest = first_line.ingest
        attachment_name = first_line.attachment_name
        if ingest is None and self._section_paragraphs:
            sec_first_line = self._section_paragraphs[0].first_line
            if sec_first_line is not None:
                ingest = sec_first_line.ingest
                attachment_name = sec_first_line.attachment_name

        # Build flat section text fields.
        section_fields: Dict[str, Optional[str]] = {}
        for label_str, field_name in _LABEL_TO_FIELD.items():
            pps = self._sections.get(label_str, [])
            section_fields[field_name] = (
                "\n".join(str(p) for p in pps) if pps else None
            )

        # Build span fields.
        span_fields: Dict[str, List[Dict[str, str]]] = {}
        for label_str, spans_field in _LABEL_TO_SPANS_FIELD.items():
            if label_str == "Nomenclature":
                source_pps = self._nomenclatures
            else:
                source_pps = self._sections.get(label_str, [])
            span_fields[spans_field] = [
                span_to_string_dict(p.as_span().as_dict()) for p in source_pps
            ]

        retval: Dict[str, Any] = {
            "taxon": "\n".join(str(p) for p in self._nomenclatures),
            "ingest": ingest,
            "line_number": first_line.line_number,
            "paragraph_number": pp.paragraph_number,
            "pdf_page": pp.pdf_page,
            "pdf_label": pp.pdf_label,
            "empirical_page_number": (
                str(pp.empirical_page_number)
                if pp.empirical_page_number is not None
                else None
            ),
            "attachment_name": attachment_name,
        }
        retval.update(section_fields)
        retval.update(span_fields)
        return retval


# ---------------------------------------------------------------------------
# Treatment-section label set
# ---------------------------------------------------------------------------

# All label strings that keep a treatment open (do not increment the gap).
_TREATMENT_SECTION_LABELS: frozenset = frozenset(
    {
        "Description",
        "Diagnosis",
        "Etymology",
        "Distribution",
        "Materials-examined",
        "Type-designation",
        "Biology",
        "Notes",
        "Key",
        "Figure-caption",
    }
)


def group_paragraphs(paragraphs: Iterable[Paragraph]) -> Iterator[Taxon]:
    """Group annotated paragraphs into Taxon objects.

    State machine with two states:

    Look for Nomenclatures
      Nomenclature  → accumulate; stay in state
      treatment-section label → if no Nomenclature yet, create stub;
                                transition to Look for Descriptions
      Misc-exposition → increment misc_gap; if > MISC_GAP_LIMIT and
                        has a Nomenclature, reset the current taxon
      other → skip

    Look for Descriptions
      treatment-section label → add to current treatment; reset misc_gap
      Nomenclature → yield current treatment (if has sections); start new
      Misc-exposition → increment misc_gap; if > MISC_GAP_LIMIT, yield
                        (if complete) and reset
      document boundary → yield (if complete) and reset

    Yields complete Taxon objects (has_nomenclature() and has_section()).
    """
    nomenclature = Label("Nomenclature")
    misc_exposition = Label("Misc-exposition")

    state = "Look for Nomenclatures"
    taxon = Taxon()
    misc_gap = 0

    for pp in paragraphs:
        label = pp.top_label()
        label_str = str(label)

        if state == "Look for Nomenclatures":
            if label == nomenclature:
                taxon.add_nomenclature(pp)
                misc_gap = 0
                continue

            if label_str in _TREATMENT_SECTION_LABELS:
                if not taxon.has_nomenclature():
                    stub_line = Line("Nomen undetected")
                    stub_paragraph = Paragraph(
                        labels=[nomenclature],
                        lines=[stub_line],
                        paragraph_number=pp.paragraph_number,
                    )
                    taxon.add_nomenclature(stub_paragraph)
                state = "Look for Descriptions"
                # Fall through to Look for Descriptions handling below.

            elif label == misc_exposition:
                misc_gap += 1
                if (
                    misc_gap > Taxon.MISC_GAP_LIMIT
                    and taxon.has_nomenclature()
                ):
                    taxon.reset()
                    misc_gap = 0
                continue

            else:
                continue  # Unknown label — skip.

        if state == "Look for Descriptions":
            # Document boundary check.
            pp_doc_id = pp.first_line.doc_id if pp.first_line else None
            taxon_doc_id = taxon.doc_id()
            if pp_doc_id and taxon_doc_id and pp_doc_id != taxon_doc_id:
                if taxon.has_nomenclature() and taxon.has_section():
                    yield taxon
                taxon = Taxon()
                misc_gap = 0
                state = "Look for Nomenclatures"
                if label == nomenclature:
                    taxon.add_nomenclature(pp)
                continue

            if label_str in _TREATMENT_SECTION_LABELS:
                taxon.add_section(label_str, pp)
                misc_gap = 0
                continue

            if label == nomenclature:
                if taxon.has_section():
                    yield taxon
                taxon = Taxon()
                taxon.add_nomenclature(pp)
                misc_gap = 0
                state = "Look for Nomenclatures"
                continue

            if label == misc_exposition:
                misc_gap += 1
                if misc_gap > Taxon.MISC_GAP_LIMIT:
                    if taxon.has_nomenclature() and taxon.has_section():
                        yield taxon
                    taxon = Taxon()
                    misc_gap = 0
                    state = "Look for Nomenclatures"
                continue

    if taxon.has_nomenclature() and taxon.has_section():
        yield taxon
