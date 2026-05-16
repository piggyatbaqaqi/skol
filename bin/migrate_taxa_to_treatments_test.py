"""Tests for bin/migrate_taxa_to_treatments.py.

Focused on the pure-logic helpers: DB-name transformation, document
field rewrites, experiment-doc database-mapping updates.  No CouchDB
access — real-DB integration runs separately via the CLI in --dry-run mode.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from migrate_taxa_to_treatments import (  # type: ignore[import]  # noqa: E402
    discover_source_dbs,
    is_taxa_db_name,
    rename_db_name,
    rename_experiment_databases,
    rename_taxon_field,
)


# ---------------------------------------------------------------------------
# rename_db_name
# ---------------------------------------------------------------------------


class TestRenameDbName:
    def test_dev_main(self) -> None:
        assert rename_db_name("skol_taxa_dev") == "skol_treatments_dev"

    def test_full_variant(self) -> None:
        assert (
            rename_db_name("skol_taxa_full_dev")
            == "skol_treatments_full_dev"
        )

    def test_experiment_specific(self) -> None:
        assert (
            rename_db_name("skol_taxa_taxpub_v1_dev")
            == "skol_treatments_taxpub_v1_dev"
        )

    def test_suffix_form(self) -> None:
        """Experiment DBs name taxa at the end: skol_exp_X_taxa."""
        assert (
            rename_db_name("skol_exp_hand_annotated_taxa")
            == "skol_exp_hand_annotated_treatments"
        )
        assert (
            rename_db_name("skol_exp_jats_v1_taxa")
            == "skol_exp_jats_v1_treatments"
        )

    def test_no_taxa_component(self) -> None:
        """If 'taxa' isn't an underscore-delimited word, return unchanged."""
        assert rename_db_name("skol_training") == "skol_training"
        assert rename_db_name("skol_taxon_clusters") == "skol_taxon_clusters"


# ---------------------------------------------------------------------------
# is_taxa_db_name / discover_source_dbs
# ---------------------------------------------------------------------------


class TestIsTaxaDbName:
    def test_underscore_delimited(self) -> None:
        assert is_taxa_db_name("skol_taxa_dev") is True
        assert is_taxa_db_name("skol_exp_x_taxa") is True

    def test_partial_match_rejected(self) -> None:
        """'taxon' or 'taxonomy' should not register as a taxa DB."""
        assert is_taxa_db_name("skol_taxon_clusters") is False
        assert is_taxa_db_name("skol_taxonomy") is False


class TestDiscoverSourceDbs:
    def test_includes_taxa_dbs(self) -> None:
        all_dbs = [
            "_users",
            "skol_taxa_dev",
            "skol_taxa_full_dev",
            "skol_training",
            "skol_exp_jats_v1_taxa",
        ]
        assert discover_source_dbs(all_dbs) == [
            "skol_exp_jats_v1_taxa",
            "skol_taxa_dev",
            "skol_taxa_full_dev",
        ]

    def test_excludes_system_dbs(self) -> None:
        """Skip CouchDB internal DBs (those starting with underscore)."""
        assert discover_source_dbs(["_users", "_replicator"]) == []

    def test_excludes_known_artifacts(self) -> None:
        """skol_taxa_migration_dev is a dedup history table, not a taxa
        store; the default denylist must exclude it."""
        result = discover_source_dbs([
            "skol_taxa_dev",
            "skol_taxa_migration_dev",
        ])
        assert "skol_taxa_migration_dev" not in result
        assert "skol_taxa_dev" in result


# ---------------------------------------------------------------------------
# rename_taxon_field (per-doc rewrite)
# ---------------------------------------------------------------------------


class TestRenameTaxonField:
    def test_taxon_renamed_to_treatment(self) -> None:
        doc = {"_id": "x", "_rev": "1-abc", "taxon": "Genus species"}
        new, changed = rename_taxon_field(doc)
        assert changed is True
        assert new["treatment"] == "Genus species"
        assert "taxon" not in new

    def test_already_renamed_idempotent(self) -> None:
        doc = {"_id": "x", "treatment": "Genus species"}
        new, changed = rename_taxon_field(doc)
        assert changed is False
        assert new is doc  # No mutation when no change.

    def test_neither_field_present_no_op(self) -> None:
        doc = {"_id": "x", "other": "field"}
        new, changed = rename_taxon_field(doc)
        assert changed is False
        assert new is doc

    def test_both_fields_present_raises(self) -> None:
        """A doc carrying *both* 'taxon' and 'treatment' is suspicious data
        divergence — refuse to silently pick one."""
        doc = {"_id": "x", "taxon": "old", "treatment": "new"}
        with pytest.raises(ValueError, match="both"):
            rename_taxon_field(doc)

    def test_other_fields_preserved(self) -> None:
        doc = {
            "_id": "x",
            "_rev": "1-abc",
            "taxon": "Genus species",
            "ingest": {"_id": "doc1"},
            "description": "long desc",
        }
        new, _ = rename_taxon_field(doc)
        assert new["_id"] == "x"
        assert new["_rev"] == "1-abc"
        assert new["ingest"] == {"_id": "doc1"}
        assert new["description"] == "long desc"


# ---------------------------------------------------------------------------
# rename_experiment_databases
# ---------------------------------------------------------------------------


class TestRenameExperimentDatabases:
    def test_taxa_field_renamed_and_value_transformed(self) -> None:
        doc = {
            "_id": "production",
            "databases": {
                "ingest": "skol_dev",
                "taxa": "skol_taxa_dev",
            },
        }
        new, changed = rename_experiment_databases(doc)
        assert changed is True
        assert new["databases"] == {
            "ingest": "skol_dev",
            "treatments": "skol_treatments_dev",
        }

    def test_taxa_full_field_renamed_and_value_transformed(self) -> None:
        doc = {
            "_id": "x",
            "databases": {
                "taxa": "skol_taxa_dev",
                "taxa_full": "skol_taxa_full_dev",
            },
        }
        new, _ = rename_experiment_databases(doc)
        assert new["databases"] == {
            "treatments": "skol_treatments_dev",
            "treatments_full": "skol_treatments_full_dev",
        }

    def test_non_local_value_still_transformed(self) -> None:
        """Even when the referenced DB does not exist on this server
        (e.g. a prod-only DB seen in a dev experiment doc), the value
        transformation must run so the doc is prod-ready when later
        replicated.  3-dev validates the script; 3-prod uses the same
        script."""
        doc = {
            "_id": "hand_annotated",
            "databases": {"taxa": "skol_exp_hand_annotated_taxa"},
        }
        new, _ = rename_experiment_databases(doc)
        assert new["databases"]["treatments"] == "skol_exp_hand_annotated_treatments"

    def test_no_databases_field_no_op(self) -> None:
        doc = {"_id": "x", "other": "field"}
        new, changed = rename_experiment_databases(doc)
        assert changed is False
        assert new is doc

    def test_already_migrated_no_op(self) -> None:
        doc = {
            "_id": "x",
            "databases": {"treatments": "skol_treatments_dev"},
        }
        new, changed = rename_experiment_databases(doc)
        assert changed is False
        assert new is doc

    def test_unrelated_databases_preserved(self) -> None:
        doc = {
            "_id": "x",
            "databases": {
                "ingest": "skol_dev",
                "training": "skol_training",
                "taxa": "skol_taxa_dev",
                "annotations": "skol_ann_reviewed",
            },
        }
        new, _ = rename_experiment_databases(doc)
        assert new["databases"]["ingest"] == "skol_dev"
        assert new["databases"]["training"] == "skol_training"
        assert new["databases"]["annotations"] == "skol_ann_reviewed"

    def test_non_dict_databases_field_no_op(self) -> None:
        """Defensive: handle malformed experiment docs gracefully."""
        doc = {"_id": "x", "databases": "not a dict"}
        new, changed = rename_experiment_databases(doc)
        assert changed is False
        assert new is doc
