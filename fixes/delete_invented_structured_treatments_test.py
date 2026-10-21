"""Tests for fixes/delete_invented_structured_treatments.py.

Run with:
    python -m pytest fixes/delete_invented_structured_treatments_test.py -v
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Iterator, List


sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from delete_invented_structured_treatments import (  # noqa: E402
    delete_invented,
    is_invented,
    target_db_name,
)

REPLACEMENT = "�"
FEATURES = {"pileus": {"shape": ["convex"]}}


def doc(doc_id: str, description: Any,
        json_annotated: Any = None) -> Dict[str, Any]:
    return {"_id": doc_id, "_rev": "1-a", "treatment": f"T {doc_id}",
            "description": description,
            "json_annotated": FEATURES if json_annotated is None
            else json_annotated,
            "json_valid": True}


class FakeDB:
    """The slice of couchdb.Database the fix uses."""

    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = {d["_id"]: d for d in docs}
        self.deleted: List[str] = []

    def __iter__(self) -> Iterator[str]:
        return iter(list(self.docs))

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        return self.docs[doc_id]

    def delete(self, document: Dict[str, Any]) -> None:
        self.deleted.append(document["_id"])
        del self.docs[document["_id"]]


class TestIsInvented(unittest.TestCase):

    def test_untranslatable_descriptions_are_invented(self):
        for text in (None, "", "  \n", "... --", "2.5-4 × 0.5",
                     REPLACEMENT * 30 + "et al." + REPLACEMENT * 30):
            with self.subTest(text=text):
                self.assertTrue(is_invented(doc("t", text)))

    def test_missing_description_is_invented(self):
        record = doc("t", "x")
        del record["description"]
        self.assertTrue(is_invented(record))

    def test_real_description_is_kept(self):
        self.assertFalse(is_invented(doc("t", "Pileus 40-60 mm, convex.")))

    def test_partly_garbled_description_is_kept(self):
        text = "Basidiomes" + REPLACEMENT * 5 + " separable, gelatinous"
        self.assertFalse(is_invented(doc("t", text)))

    def test_judged_on_description_not_on_the_json(self):
        """An untranslatable description is invented whatever the model
        produced; the pipeline now never saves such records at all."""
        self.assertTrue(is_invented(doc("t", "", json_annotated={})))
        self.assertFalse(is_invented(doc("t", "Pileus convex.",
                                         json_annotated={})))


class TestTargetDbName(unittest.TestCase):

    CONFIG = {"treatments_structured_db_name":
              "skol_exp_production_v4_1_03_00_treatments_structured",
              "dest_db": "skol_treatments_full_dev"}

    def test_explicit_db_wins(self):
        self.assertEqual(
            target_db_name("scratch", "production_v4_1", self.CONFIG),
            "scratch")

    def test_experiment_uses_its_structured_db(self):
        self.assertEqual(
            target_db_name(None, "production_v4_1", self.CONFIG),
            "skol_exp_production_v4_1_03_00_treatments_structured")

    def test_refuses_to_default(self):
        with self.assertRaises(ValueError):
            target_db_name(None, None, self.CONFIG)


class TestDeleteInvented(unittest.TestCase):

    def run_fix(self, db: FakeDB, **kwargs: Any) -> Dict[str, int]:
        with contextlib.redirect_stdout(io.StringIO()):
            return delete_invented(db, **kwargs)

    def test_deletes_only_invented(self):
        db = FakeDB([doc("empty", ""), doc("real", "Pileus convex."),
                     doc("noise", REPLACEMENT * 20)])
        counts = self.run_fix(db)
        self.assertEqual(sorted(db.deleted), ["empty", "noise"])
        self.assertEqual(counts, {"scanned": 3, "invented": 2,
                                  "deleted": 2})

    def test_skips_design_docs(self):
        db = FakeDB([doc("_design/views", None), doc("real", "Pileus.")])
        counts = self.run_fix(db)
        self.assertEqual(db.deleted, [])
        self.assertEqual(counts["scanned"], 1)

    def test_dry_run_deletes_nothing_but_counts(self):
        db = FakeDB([doc("empty", ""), doc("real", "Pileus convex.")])
        counts = self.run_fix(db, dry_run=True)
        self.assertEqual(db.deleted, [])
        self.assertEqual(counts["deleted"], 1)

    def test_limit_caps_deletions(self):
        db = FakeDB([doc(f"e{i}", "") for i in range(5)])
        counts = self.run_fix(db, limit=2)
        self.assertEqual(len(db.deleted), 2)
        self.assertEqual(counts["deleted"], 2)

    def test_lists_each_deletion_at_normal_verbosity(self):
        db = FakeDB([doc("taxon_empty", ""), doc("taxon_keep", "Pileus.")])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            delete_invented(db, dry_run=True, verbosity=1)
        self.assertIn("taxon_empty", out.getvalue())
        self.assertNotIn("taxon_keep", out.getvalue())


if __name__ == "__main__":
    unittest.main()
