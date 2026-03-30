"""Tests for YEDDA ↔ brat standoff converters.

Covers yedda_to_brat() and brat_to_yedda() individually, plus round-trip
fidelity.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from yedda_to_brat import yedda_to_brat
from brat_to_yedda import brat_to_yedda


class TestYeddaToBrat(unittest.TestCase):
    """yedda_to_brat: YEDDA string → (plaintext, brat_ann)."""

    def test_single_block_plaintext(self) -> None:
        yedda = "[@Amanita muscaria#Nomenclature*]"
        plaintext, _ = yedda_to_brat(yedda)
        self.assertEqual(plaintext, "Amanita muscaria")

    def test_single_block_offsets(self) -> None:
        yedda = "[@Amanita muscaria#Nomenclature*]"
        _, ann = yedda_to_brat(yedda)
        self.assertIn("Nomenclature", ann)
        self.assertIn("Nomenclature 0 16", ann)  # start=0, end=16

    def test_single_block_ann_format(self) -> None:
        yedda = "[@Amanita muscaria#Nomenclature*]"
        _, ann = yedda_to_brat(yedda)
        # Must have a line starting with T1
        lines = [ln for ln in ann.splitlines() if ln.startswith("T1")]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], "T1\tNomenclature 0 16\tAmanita muscaria")

    def test_two_blocks_offsets(self) -> None:
        yedda = (
            "[@Amanita muscaria#Nomenclature*]\n\n"
            "[@Pileus red.#Description*]"
        )
        plaintext, ann = yedda_to_brat(yedda)
        # Plaintext: "Amanita muscaria\n\nPileus red."
        self.assertEqual(plaintext, "Amanita muscaria\n\nPileus red.")
        nom_start = 0
        nom_end = len("Amanita muscaria")
        desc_start = nom_end + 2  # two-char separator "\n\n"
        desc_end = desc_start + len("Pileus red.")
        self.assertIn(f"Nomenclature {nom_start} {nom_end}", ann)
        self.assertIn(f"Description {desc_start} {desc_end}", ann)

    def test_entity_numbers_are_sequential(self) -> None:
        yedda = (
            "[@Amanita muscaria#Nomenclature*]\n\n"
            "[@Pileus red.#Description*]\n\n"
            "[@Distribution. Found in Europe.#Misc-exposition*]"
        )
        _, ann = yedda_to_brat(yedda)
        lines = [ln for ln in ann.splitlines() if ln]
        ids = [int(ln.split("\t")[0][1:]) for ln in lines]
        self.assertEqual(ids, [1, 2, 3])

    def test_empty_yedda_returns_empty(self) -> None:
        plaintext, ann = yedda_to_brat("")
        self.assertEqual(plaintext, "")
        self.assertEqual(ann, "")

    def test_multiline_block(self) -> None:
        yedda = "[@Line one\nLine two#Description*]"
        plaintext, ann = yedda_to_brat(yedda)
        self.assertEqual(plaintext, "Line one\nLine two")
        # len("Line one\nLine two") == 17
        self.assertIn("Description 0 17", ann)

    def test_unicode_text(self) -> None:
        # Non-ASCII characters; offsets are character-based (not byte-based).
        yedda = "[@Mérat 1821#Nomenclature*]"
        plaintext, ann = yedda_to_brat(yedda)
        end = len("Mérat 1821")  # character count
        self.assertIn(f"Nomenclature 0 {end}", ann)
        self.assertEqual(plaintext, "Mérat 1821")


class TestBratToYedda(unittest.TestCase):
    """brat_to_yedda: (plaintext, brat_ann) → YEDDA string."""

    def test_single_entity(self) -> None:
        plaintext = "Amanita muscaria"
        ann = "T1\tNomenclature 0 16\tAmanita muscaria"
        yedda = brat_to_yedda(plaintext, ann)
        self.assertIn("[@Amanita muscaria#Nomenclature*]", yedda)

    def test_two_entities(self) -> None:
        plaintext = "Amanita muscaria\n\nPileus red."
        ann = (
            "T1\tNomenclature 0 16\tAmanita muscaria\n"
            "T2\tDescription 18 29\tPileus red."
        )
        yedda = brat_to_yedda(plaintext, ann)
        self.assertIn("[@Amanita muscaria#Nomenclature*]", yedda)
        self.assertIn("[@Pileus red.#Description*]", yedda)

    def test_blocks_appear_in_offset_order(self) -> None:
        plaintext = "Amanita muscaria\n\nPileus red."
        # Provide annotations out of order
        ann = (
            "T2\tDescription 18 29\tPileus red.\n"
            "T1\tNomenclature 0 16\tAmanita muscaria"
        )
        yedda = brat_to_yedda(plaintext, ann)
        nom_pos = yedda.index("Nomenclature")
        desc_pos = yedda.index("Description")
        self.assertLess(nom_pos, desc_pos)

    def test_empty_ann_returns_empty(self) -> None:
        yedda = brat_to_yedda("", "")
        self.assertEqual(yedda.strip(), "")

    def test_non_entity_lines_ignored(self) -> None:
        # brat .ann files may have relation (R) or attribute (A) lines
        plaintext = "Amanita muscaria"
        ann = (
            "T1\tNomenclature 0 16\tAmanita muscaria\n"
            "A1\tSomeAttr T1 value\n"
            "R1\tRelation Arg1:T1 Arg2:T1"
        )
        yedda = brat_to_yedda(plaintext, ann)
        self.assertIn("[@Amanita muscaria#Nomenclature*]", yedda)


class TestYeddaBratRoundTrip(unittest.TestCase):
    """Full round-trip: YEDDA → brat → YEDDA preserves content."""

    def test_single_block_roundtrip(self) -> None:
        original = "[@Amanita muscaria (Fr.) Mérat#Nomenclature*]"
        plaintext, ann = yedda_to_brat(original)
        recovered = brat_to_yedda(plaintext, ann)
        self.assertIn("Amanita muscaria (Fr.) Mérat", recovered)
        self.assertIn("Nomenclature", recovered)

    def test_two_block_roundtrip(self) -> None:
        original = (
            "[@Amanita muscaria (Fr.) Mérat#Nomenclature*]\n\n"
            "[@Pileus 5–10 cm, convex.#Description*]"
        )
        plaintext, ann = yedda_to_brat(original)
        recovered = brat_to_yedda(plaintext, ann)
        self.assertIn("Nomenclature", recovered)
        self.assertIn("Description", recovered)

    def test_all_new_tags_survive_roundtrip(self) -> None:
        blocks = [
            (
                "[@Differs from A. muscaria by smaller spores."
                "#Diagnosis*]",
                "Diagnosis",
            ),
            (
                "[@Distribution. Found across Europe.#Distribution*]",
                "Distribution",
            ),
            (
                "[@NY 12345 (holotype).#Materials-examined*]",
                "Materials-examined",
            ),
            (
                "[@Holotype: NY 12345.#Type-designation*]",
                "Type-designation",
            ),
            (
                "[@Saprotrophic on dead hardwood.#Biology*]",
                "Biology",
            ),
        ]
        for yedda, tag in blocks:
            with self.subTest(tag=tag):
                plaintext, ann = yedda_to_brat(yedda)
                recovered = brat_to_yedda(plaintext, ann)
                self.assertIn(tag, recovered)

    def test_multiblock_order_preserved(self) -> None:
        original = (
            "[@Amanita muscaria#Nomenclature*]\n\n"
            "[@Holotype: NY 12345.#Type-designation*]\n\n"
            "[@Pileus 5–10 cm.#Description*]"
        )
        plaintext, ann = yedda_to_brat(original)
        recovered = brat_to_yedda(plaintext, ann)
        nom_pos = recovered.index("Nomenclature")
        td_pos = recovered.index("Type-designation")
        desc_pos = recovered.index("Description")
        self.assertLess(nom_pos, td_pos)
        self.assertLess(td_pos, desc_pos)


if __name__ == "__main__":
    unittest.main()
