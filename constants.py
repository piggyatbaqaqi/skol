# Matches PDF page markers emitted by ingestors/extract_plaintext.  Both
# forms are accepted so downstream readers (fileobj.FileObject.read_line,
# pdf_section_extractor._get_pdf_page_marker) find the marker regardless of
# whether the caller reads raw plaintext or a YEDDA-wrapped .ann attachment:
#
#     --- PDF Page 7 Label 7 ---
#     [@--- PDF Page 7 Label 7 ---#Page-header*]
#
# Capture-group indexes are preserved for callers: group(1) is the numeric
# PDF page, group(3) is the label token (may be None).
pdf_page_pattern = (
    r'^(?:\[@)?'
    r'---\s*PDF\s+Page\s+(\d+)\s*(Label\s+(\S+)\s+)?---'
    r'(?:\s*\#[A-Za-z-]+\*\])?'
    r'\s*$'
)
