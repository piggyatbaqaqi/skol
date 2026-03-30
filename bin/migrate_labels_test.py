"""Tests for migrate_labels.py — Tier 1 automatic relabeling."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestors.yedda_tags import Tag
from migrate_labels import migrate_yedda, relabel_by_header, relabel_holotype


class TestRelabelHolotype(unittest.TestCase):
    """Short Holotype blocks → Type-designation; long → Materials-examined."""

    def test_single_line_is_type_designation(self) -> None:
        self.assertEqual(relabel_holotype("Holotype: DUKE 123456."), Tag.TYPE_DESIGNATION)

    def test_two_lines_is_type_designation(self) -> None:
        text = "Holotype: NY 12345.\nIsotype: BPI 678."
        self.assertEqual(relabel_holotype(text), Tag.TYPE_DESIGNATION)

    def test_three_lines_is_materials_examined(self) -> None:
        text = "Holotype: NY 12345.\nIsotype: BPI 678.\nParatype: K 999."
        self.assertEqual(relabel_holotype(text), Tag.MATERIALS_EXAMINED)

    def test_blank_lines_not_counted(self) -> None:
        text = "Holotype: NY 12345.\n\nIsotype: BPI 678."
        self.assertEqual(relabel_holotype(text), Tag.TYPE_DESIGNATION)

    def test_long_specimen_list_is_materials_examined(self) -> None:
        text = "\n".join([
            "Specimens examined: China, Yunnan.",
            "NY 12345 (holotype).",
            "BPI 678 (isotype).",
            "K 999 (paratype).",
        ])
        self.assertEqual(relabel_holotype(text), Tag.MATERIALS_EXAMINED)


class TestRelabelByHeader(unittest.TestCase):
    """Header keyword detection on first sentence."""

    def test_distribution_keyword(self) -> None:
        self.assertEqual(
            relabel_by_header("Distribution. Found across Europe."),
            Tag.DISTRIBUTION,
        )

    def test_habitat_keyword(self) -> None:
        self.assertEqual(
            relabel_by_header("Habitat. Grows on decaying oak logs."),
            Tag.DISTRIBUTION,
        )

    def test_range_keyword(self) -> None:
        self.assertEqual(
            relabel_by_header("Range: temperate zones."),
            Tag.DISTRIBUTION,
        )

    def test_diagnosis_keyword(self) -> None:
        self.assertEqual(
            relabel_by_header(
                "Diagnosis. Differs from A. muscaria by smaller spores."
            ),
            Tag.DIAGNOSIS,
        )

    def test_biology_keyword(self) -> None:
        self.assertEqual(
            relabel_by_header("Biology. Saprotrophic on dead hardwood."),
            Tag.BIOLOGY,
        )

    def test_ecology_keyword(self) -> None:
        self.assertEqual(
            relabel_by_header("Ecology. Found in mixed forests."),
            Tag.BIOLOGY,
        )

    def test_host_keyword(self) -> None:
        self.assertEqual(
            relabel_by_header("Host. Quercus robur."),
            Tag.BIOLOGY,
        )

    def test_no_keyword_returns_none(self) -> None:
        self.assertIsNone(
            relabel_by_header("Pileus 3–8 cm, convex to flat.")
        )

    def test_keyword_only_in_later_sentence_not_matched(self) -> None:
        # "Distribution" appears only after a full sentence — should not match
        self.assertIsNone(
            relabel_by_header("Basidiomata annual. Distribution is broad.")
        )

    def test_case_insensitive(self) -> None:
        self.assertEqual(
            relabel_by_header("DIAGNOSIS. Differs from X."),
            Tag.DIAGNOSIS,
        )


class TestMigrateYedda(unittest.TestCase):
    """Full YEDDA migration: apply all Tier 1 rules."""

    def test_holotype_short_becomes_type_designation(self) -> None:
        yedda = "[@Holotype: NY 12345.#Holotype*]"
        new_text, changes = migrate_yedda(yedda)
        self.assertIn("Type-designation", new_text)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["old_tag"], "Holotype")
        self.assertEqual(changes[0]["new_tag"], "Type-designation")

    def test_holotype_long_becomes_materials_examined(self) -> None:
        specimen_list = "\n".join(["NY 12345.", "BPI 678.", "K 999."])
        yedda = f"[@{specimen_list}#Holotype*]"
        new_text, changes = migrate_yedda(yedda)
        self.assertIn("Materials-examined", new_text)
        self.assertEqual(len(changes), 1)

    def test_header_keyword_relabels_misc_exposition(self) -> None:
        yedda = "[@Distribution. Found across Europe.#Misc-exposition*]"
        new_text, changes = migrate_yedda(yedda)
        self.assertIn("Distribution", new_text)
        self.assertEqual(changes[0]["old_tag"], "Misc-exposition")
        self.assertEqual(changes[0]["new_tag"], "Distribution")

    def test_header_keyword_relabels_description(self) -> None:
        yedda = "[@Ecology. Found in mixed forests.#Description*]"
        new_text, changes = migrate_yedda(yedda)
        self.assertIn("Biology", new_text)
        self.assertEqual(len(changes), 1)

    def test_no_rule_matches_leaves_block_unchanged(self) -> None:
        yedda = "[@Pileus convex.#Description*]"
        new_text, changes = migrate_yedda(yedda)
        self.assertIn("[@Pileus convex.#Description*]", new_text)
        self.assertEqual(changes, [])

    def test_already_correct_tag_not_reported_as_change(self) -> None:
        yedda = "[@Distribution. Found in Asia.#Distribution*]"
        _, changes = migrate_yedda(yedda)
        self.assertEqual(changes, [])

    def test_multiple_blocks_mixed(self) -> None:
        yedda = (
            "[@Amanita muscaria (Fr.) Mérat#Nomenclature*]\n\n"
            "[@Holotype: NY 12345.#Holotype*]\n\n"
            "[@Pileus 5–10 cm.#Description*]"
        )
        new_text, changes = migrate_yedda(yedda)
        self.assertEqual(len(changes), 1)
        self.assertIn("Nomenclature", new_text)
        self.assertIn("Type-designation", new_text)
        self.assertIn("Description", new_text)

    def test_change_record_includes_block_index(self) -> None:
        yedda = (
            "[@Amanita muscaria#Nomenclature*]\n\n"
            "[@Holotype: NY 12345.#Holotype*]"
        )
        _, changes = migrate_yedda(yedda)
        self.assertEqual(changes[0]["block_index"], 1)

    def test_empty_yedda_returns_empty(self) -> None:
        new_text, changes = migrate_yedda("")
        self.assertEqual(new_text, "")
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
