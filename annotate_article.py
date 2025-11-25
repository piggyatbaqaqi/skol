#!/usr/bin/env python3
"""Annotate article.txt with YEDA format labels."""

from line import Line
from paragraph import Paragraph
import re

def read_article(filename):
    """Read article and return lines."""
    with open(filename, 'r') as f:
        return [line.rstrip('\n') for line in f]

def is_likely_nomenclature(text):
    """
    Check if text contains actual nomenclatural statement.

    More restrictive than Paragraph.contains_nomenclature() to avoid
    false positives from year citations in running text.
    """
    # Must have a binomial (Genus species pattern)
    if not re.search(r'\b[A-Z][a-z]+\s+[a-z]+\b', text):
        return False

    # Check with Paragraph class
    pp = Paragraph()
    for line in text.split('\n'):
        pp.append(Line(line))

    if not pp.contains_nomenclature():
        return False

    # Additional checks to filter false positives:
    # Must have author names or nomenclatural indicators
    indicators = [
        r'\bgen\.\s*nov\.',  # new genus
        r'\bsp\.\s*nov\.',   # new species
        r'\bcomb\.\s*nov\.',  # new combination
        r'\bstat\.\s*nov\.',  # new status
        r'\bemend\.',         # emendation
        r'^\s*≡',             # basionym
        r'MycoBank',
        r'\bet\s+al\.\s*,?\s*\(',  # Author et al. (year)
        r'\([A-Z][a-z]+\.?\)',  # (Author) in name
    ]

    return any(re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
               for pattern in indicators)

def is_description_section(text):
    """Check if text is a taxonomic description section."""
    text_lower = text.lower()

    # Description section headers
    headers = [
        'key characters:', 'diagnosis:', 'description:', 'etymology:',
        'type species:', 'type genus:', 'emended description:',
        'morphology:', 'material examined:', 'specimens? examined:',
        'holotype:', 'habitat:', 'distribution:', 'remarks:',
    ]

    if any(h in text_lower for h in headers):
        return True

    # Figure captions describing morphology
    if re.search(r'^figs?\.\s+\d', text_lower, re.MULTILINE):
        if any(term in text_lower for term in ['spore', 'hypha', 'wall', 'myc']):
            return True

    # Detailed morphological descriptions (multiple measurements)
    measurements = len(re.findall(r'\d+[–-]\d+\s*[µμ]m', text))
    if measurements >= 3:
        morph_terms = ['spore', 'wall', 'layer', 'hypha', 'septum']
        if sum(1 for t in morph_terms if t in text_lower) >= 2:
            return True

    return False

def group_lines_into_paragraphs(lines):
    """
    Group lines into logical paragraphs.

    Uses indentation, line length, and content patterns as heuristics.
    """
    paragraphs = []
    current = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Empty line - end paragraph
        if not stripped:
            if current:
                paragraphs.append('\n'.join(current))
                current = []
            continue

        # Start new paragraph on these patterns:
        should_break = False

        # New section headers (all caps, short lines)
        if stripped.isupper() and len(stripped) < 50:
            should_break = True

        # Formal nomenclatural statements (starts with genus name or ≡)
        elif re.match(r'^[A-Z][a-z]+\s+[a-z]+', stripped):
            # Might be a nomenclatural statement
            if any(pattern in line for pattern in [', gen. nov.', ', sp. nov.', ', comb. nov.', 'MycoBank']):
                should_break = True

        # Basionym/synonym lines
        elif stripped.startswith('≡'):
            should_break = True

        # New paragraph after certain punctuation
        elif current and current[-1].rstrip().endswith(('.', '!', '?')):
            # Check if this line starts a new sentence
            if stripped[0].isupper() and not stripped.startswith('The '):
                # Probably new paragraph (with exceptions for "The" continuing)
                should_break = True

        # Section headers ending with colon
        elif ':' in line and len(stripped) < 80:
            if any(keyword in stripped.lower() for keyword in [
                'etymology', 'diagnosis', 'description', 'morphology',
                'type species', 'key characters', 'material examined'
            ]):
                should_break = True

        if should_break and current:
            paragraphs.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        paragraphs.append('\n'.join(current))

    return paragraphs

def classify_and_annotate(paragraphs):
    """Classify paragraphs and add YEDA annotations."""
    output_lines = []
    stats = {'Nomenclature': 0, 'Description': 0, 'Misc-exposition': 0}

    for para in paragraphs:
        if not para.strip():
            continue

        # Classify
        if is_likely_nomenclature(para):
            label = 'Nomenclature'
        elif is_description_section(para):
            label = 'Description'
        else:
            label = 'Misc-exposition'

        stats[label] += 1

        # Add YEDA annotation
        output_lines.append('[@ ' + para)
        output_lines.append(f'#{label}*]')

    return output_lines, stats

def main():
    # Read and process
    lines = read_article('test_data/article.txt')
    print(f"Read {len(lines)} lines from article.txt")

    paragraphs = group_lines_into_paragraphs(lines)
    print(f"Grouped into {len(paragraphs)} paragraphs")

    output_lines, stats = classify_and_annotate(paragraphs)

    # Write output
    with open('test_data/article_reference.txt', 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"\nClassification statistics:")
    for label in ['Nomenclature', 'Description', 'Misc-exposition']:
        print(f"  {label}: {stats[label]}")

    print(f"\nWrote {sum(stats.values())} annotated paragraphs to test_data/article_reference.txt")

if __name__ == '__main__':
    main()
