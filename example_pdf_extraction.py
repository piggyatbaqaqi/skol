"""
Example usage of PDFSectionExtractor class.

This script demonstrates how to extract sections from PDF attachments
in CouchDB documents.
"""

from pdf_section_extractor import PDFSectionExtractor

# Initialize extractor (uses environment variables for credentials)
extractor = PDFSectionExtractor(verbosity=1)

# Example 1: Extract from a specific document
print("\n" + "="*70)
print("EXAMPLE 1: Extract from specific document")
print("="*70)

sections = extractor.extract_from_document(
    database='skol_dev',
    doc_id='00df9554e9834283b5e844c7a994ba5f',
    attachment_name='article.pdf'  # Optional: auto-detects if omitted
)

print(f"\nExtracted {len(sections)} sections")
print("\nFirst 3 sections:")
for i, section in enumerate(sections[:3], 1):
    print(f"{i}. {section[:60]}...")

# Example 2: Extract metadata
print("\n" + "="*70)
print("EXAMPLE 2: Extract metadata")
print("="*70)

metadata = extractor.extract_metadata(sections)

print(f"Title: {metadata['title']}")
print(f"Keywords: {', '.join(metadata['keywords'])}")
print(f"Abstract: {metadata['abstract'][:100] if metadata['abstract'] else 'Not found'}...")
print(f"Sections: {', '.join(metadata['sections_found'])}")

# Example 3: Search for specific content
print("\n" + "="*70)
print("EXAMPLE 3: Search sections")
print("="*70)

# Find all sections mentioning "ascospores"
matching = extractor.get_section_by_keyword(sections, 'ascospores')
print(f"\nFound {len(matching)} sections mentioning 'ascospores'")
if matching:
    print(f"First match: {matching[0][:100]}...")

# Example 4: Auto-detect PDF attachment
print("\n" + "="*70)
print("EXAMPLE 4: Auto-detect PDF")
print("="*70)

# List attachments first
attachments = extractor.list_attachments(
    database='skol_dev',
    doc_id='00df9554e9834283b5e844c7a994ba5f'
)
print(f"\nAvailable attachments:")
for name, info in attachments.items():
    content_type = info.get('content_type', 'unknown')
    size = info.get('length', 0)
    print(f"  - {name}: {content_type} ({size:,} bytes)")

# Auto-detect and extract
sections2 = extractor.extract_from_document(
    database='skol_dev',
    doc_id='00df9554e9834283b5e844c7a994ba5f'
    # attachment_name omitted - will auto-detect PDF
)
print(f"\nAuto-detected PDF and extracted {len(sections2)} sections")

print("\n" + "="*70)
print("EXAMPLES COMPLETE")
print("="*70)
