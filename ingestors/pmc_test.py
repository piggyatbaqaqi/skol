"""
TDD tests for the PmcBiocIngestor class.

Tests article discovery via E-utilities esearch, BioC JSON retrieval,
DOI-based enrichment of existing documents, metadata extraction, and
skip-existing logic.
"""

import json
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call
from uuid import uuid5, NAMESPACE_URL

from .pmc import PmcBiocIngestor


def _make_ingestor(**overrides: Any) -> PmcBiocIngestor:
    """Create a PmcBiocIngestor with mocked dependencies."""
    db = overrides.pop('db', MagicMock())
    robot_parser = overrides.pop('robot_parser', MagicMock())
    robot_parser.can_fetch.return_value = True
    defaults: Dict[str, Any] = dict(
        db=db,
        user_agent='test-agent',
        robot_parser=robot_parser,
        verbosity=0,
        journal_search_term='"Fungal Syst Evol"[journal]',
    )
    defaults.update(overrides)
    return PmcBiocIngestor(**defaults)


def _make_esearch_response(
    pmcids: List[str],
    count: int = 0,
    retstart: int = 0,
    retmax: int = 500,
) -> Dict[str, Any]:
    """Build a mock E-utilities esearch JSON response."""
    if count == 0:
        count = len(pmcids)
    return {
        "header": {"type": "esearch", "version": "0.3"},
        "esearchresult": {
            "count": str(count),
            "retmax": str(retmax),
            "retstart": str(retstart),
            "idlist": pmcids,
            "translationset": [],
            "querytranslation": '"Fungal Syst Evol"[Journal]',
        },
    }


def _make_bioc_json(
    pmcid: str = "PMC1234567",
    pmid: str = "12345678",
    doi: str = "10.1234/test.2025.001",
    title: str = "A new species of Testomyces",
    authors: Optional[List[Dict[str, str]]] = None,
    license: str = "CC BY",
) -> List[Dict[str, Any]]:
    """Build a minimal BioC JSON response matching the real API structure."""
    if authors is None:
        authors_infons = {
            "name_0": "surname:Smith;given-names:Alice",
            "name_1": "surname:Jones;given-names:Bob",
        }
    else:
        authors_infons = {
            f"name_{i}": f"surname:{a['surname']};given-names:{a['given_names']}"
            for i, a in enumerate(authors)
        }

    front_passage: Dict[str, Any] = {
        "bioctype": "BioCPassage",
        "offset": 0,
        "infons": {
            "article-id_doi": doi,
            "article-id_pmc": pmcid,
            "article-id_pmid": pmid,
            "section_type": "TITLE",
            "type": "front",
            "year": "2025",
            "volume": "10",
            "issue": "1",
            **authors_infons,
        },
        "text": title,
        "sentences": [],
        "annotations": [],
        "relations": [],
    }
    abstract_passage: Dict[str, Any] = {
        "bioctype": "BioCPassage",
        "offset": len(title) + 1,
        "infons": {
            "section_type": "ABSTRACT",
            "type": "abstract",
        },
        "text": "This is the abstract text.",
        "sentences": [],
        "annotations": [],
        "relations": [],
    }
    return [
        {
            "bioctype": "BioCCollection",
            "source": "PMC",
            "date": "20250301",
            "key": "pmc.key",
            "version": "1.0",
            "infons": {},
            "documents": [
                {
                    "bioctype": "BioCDocument",
                    "id": pmcid,
                    "infons": {"license": license},
                    "passages": [front_passage, abstract_passage],
                    "relations": [],
                }
            ],
        }
    ]


def _make_mock_response(
    status_code: int = 200,
    json_data: Any = None,
    text: str = "",
) -> MagicMock:
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    if json_data is not None:
        resp.json.return_value = json_data
        resp.text = json.dumps(json_data)
    else:
        resp.text = text
        resp.json.side_effect = ValueError("No JSON")
    return resp


class TestDiscoverPmcids(unittest.TestCase):
    """Test article discovery via E-utilities esearch."""

    def test_discover_pmcids(self) -> None:
        """Single-page esearch returns all PMCIDs."""
        ing = _make_ingestor()
        esearch_resp = _make_esearch_response(["1111111", "2222222", "3333333"])
        mock_resp = _make_mock_response(json_data=esearch_resp)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        pmcids = ing._discover_pmcids()

        self.assertEqual(pmcids, ["1111111", "2222222", "3333333"])

    def test_discover_pmcids_pagination(self) -> None:
        """Multi-page esearch fetches all pages via retstart."""
        ing = _make_ingestor()
        page1 = _make_esearch_response(
            ["1111111", "2222222"], count=3, retmax=2, retstart=0,
        )
        page2 = _make_esearch_response(
            ["3333333"], count=3, retmax=2, retstart=2,
        )
        resp1 = _make_mock_response(json_data=page1)
        resp2 = _make_mock_response(json_data=page2)
        ing.http_client.get = MagicMock(side_effect=[resp1, resp2])

        pmcids = ing._discover_pmcids()

        self.assertEqual(pmcids, ["1111111", "2222222", "3333333"])
        self.assertEqual(ing.http_client.get.call_count, 2)

    def test_discover_respects_max_articles(self) -> None:
        """max_articles limits the number of PMCIDs returned."""
        ing = _make_ingestor(max_articles=2)
        esearch_resp = _make_esearch_response(
            ["1111111", "2222222", "3333333"],
        )
        mock_resp = _make_mock_response(json_data=esearch_resp)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        pmcids = ing._discover_pmcids()

        self.assertEqual(len(pmcids), 2)


class TestFetchBiocJson(unittest.TestCase):
    """Test BioC JSON retrieval from pmcoa.cgi."""

    def test_fetch_bioc_json_success(self) -> None:
        """Successful BioC JSON response is parsed and returned."""
        ing = _make_ingestor()
        bioc = _make_bioc_json(pmcid="PMC1234567")
        mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        result = ing._fetch_bioc_json("1234567")

        self.assertIsNotNone(result)
        self.assertEqual(result[0]["documents"][0]["id"], "PMC1234567")

    def test_fetch_bioc_json_not_in_oa(self) -> None:
        """Non-200 response (article not in OA subset) returns None."""
        ing = _make_ingestor()
        mock_resp = _make_mock_response(status_code=404, text="No result")
        ing.http_client.get = MagicMock(return_value=mock_resp)

        result = ing._fetch_bioc_json("9999999")

        self.assertIsNone(result)

    def test_fetch_bioc_json_html_error(self) -> None:
        """HTML error response (not JSON) returns None."""
        ing = _make_ingestor()
        mock_resp = _make_mock_response(status_code=200, text="<html>Error</html>")
        ing.http_client.get = MagicMock(return_value=mock_resp)

        result = ing._fetch_bioc_json("9999999")

        self.assertIsNone(result)


class TestMetadataExtraction(unittest.TestCase):
    """Test extraction of metadata from BioC JSON."""

    def test_extract_title(self) -> None:
        """Title is extracted from the TITLE passage."""
        ing = _make_ingestor()
        bioc = _make_bioc_json(title="Fungi of the World")

        metadata = ing._extract_metadata(bioc)

        self.assertEqual(metadata["title"], "Fungi of the World")

    def test_extract_doi(self) -> None:
        """DOI is extracted from front passage infons."""
        ing = _make_ingestor()
        bioc = _make_bioc_json(doi="10.9999/fungi.2025.42")

        metadata = ing._extract_metadata(bioc)

        self.assertEqual(metadata["doi"], "10.9999/fungi.2025.42")

    def test_extract_pmid(self) -> None:
        """PMID is extracted from front passage infons."""
        ing = _make_ingestor()
        bioc = _make_bioc_json(pmid="87654321")

        metadata = ing._extract_metadata(bioc)

        self.assertEqual(metadata["pmid"], "87654321")

    def test_extract_pmcid(self) -> None:
        """PMCID is extracted from front passage infons."""
        ing = _make_ingestor()
        bioc = _make_bioc_json(pmcid="PMC7777777")

        metadata = ing._extract_metadata(bioc)

        self.assertEqual(metadata["pmcid"], "PMC7777777")

    def test_extract_license(self) -> None:
        """License is extracted from document infons."""
        ing = _make_ingestor()
        bioc = _make_bioc_json(license="CC BY-SA 4.0")

        metadata = ing._extract_metadata(bioc)

        self.assertEqual(metadata["license"], "CC BY-SA 4.0")

    def test_extract_authors(self) -> None:
        """Authors are extracted from name_N infons in order."""
        ing = _make_ingestor()
        bioc = _make_bioc_json(
            authors=[
                {"surname": "Yarroll", "given_names": "Piggy"},
                {"surname": "Block", "given_names": "Gregory"},
            ],
        )

        metadata = ing._extract_metadata(bioc)

        self.assertEqual(len(metadata["authors"]), 2)
        self.assertEqual(metadata["authors"][0]["surname"], "Yarroll")
        self.assertEqual(metadata["authors"][0]["given_names"], "Piggy")
        self.assertEqual(metadata["authors"][1]["surname"], "Block")


class TestDocumentIdStability(unittest.TestCase):
    """Test that document IDs are deterministic."""

    def test_document_id_from_pmcid(self) -> None:
        """Same PMCID always produces the same document ID."""
        ing = _make_ingestor()

        id1 = ing._make_doc_id("1234567")
        id2 = ing._make_doc_id("1234567")

        self.assertEqual(id1, id2)
        expected = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        self.assertEqual(id1, expected)

    def test_different_pmcids_produce_different_ids(self) -> None:
        """Different PMCIDs produce different document IDs."""
        ing = _make_ingestor()

        id1 = ing._make_doc_id("1111111")
        id2 = ing._make_doc_id("2222222")

        self.assertNotEqual(id1, id2)


class TestSkipExisting(unittest.TestCase):
    """Test skip-existing logic for documents already in CouchDB."""

    def test_skip_existing_with_bioc_json(self) -> None:
        """Existing document with bioc_json field is skipped."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "bioc_json": [{"documents": []}],
            "_attachments": {"article.pdf": {}},
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(return_value=existing_doc)
        ing = _make_ingestor(db=db)
        bioc = _make_bioc_json(pmcid="PMC1234567")
        mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing._ingest_article("1234567")

        # Should not save — document already has bioc_json
        db.save.assert_not_called()


class TestEnrichExisting(unittest.TestCase):
    """Test DOI-based enrichment of existing documents from other sources."""

    def test_enrich_existing_by_doi(self) -> None:
        """Existing Crossref doc found by DOI gets bioc_json added."""
        db = MagicMock()
        pmc_doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        # PMC document doesn't exist yet
        db.__contains__ = MagicMock(return_value=False)

        # Existing Crossref document found by DOI
        crossref_doc = {
            "_id": "crossref_doc_id_abc",
            "_rev": "1-abc",
            "title": "Original Crossref Title",
            "authors": [{"name": "Smith, A."}],
            "doi": "10.1234/test.2025.001",
            "source": "crossref",
        }
        db.save = MagicMock(return_value=("crossref_doc_id_abc", "2-def"))
        ing = _make_ingestor(db=db)

        # Mock DOI search to find existing doc
        ing._find_existing_by_doi = MagicMock(return_value=crossref_doc)

        bioc = _make_bioc_json(
            pmcid="PMC1234567",
            doi="10.1234/test.2025.001",
        )
        mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing._ingest_article("1234567")

        # Should save with bioc_json added to existing doc
        db.save.assert_called()
        saved_doc = db.save.call_args[0][0]
        self.assertIn("bioc_json", saved_doc)
        self.assertEqual(saved_doc["_id"], "crossref_doc_id_abc")

    def test_enrich_preserves_existing_metadata(self) -> None:
        """Enrichment does not overwrite title or authors."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        crossref_doc = {
            "_id": "crossref_doc_id_abc",
            "_rev": "1-abc",
            "title": "Original Crossref Title",
            "authors": [{"name": "Smith, A."}],
            "doi": "10.1234/test.2025.001",
            "source": "crossref",
        }
        db.save = MagicMock(return_value=("crossref_doc_id_abc", "2-def"))
        ing = _make_ingestor(db=db)
        ing._find_existing_by_doi = MagicMock(return_value=crossref_doc)

        bioc = _make_bioc_json(
            pmcid="PMC1234567",
            doi="10.1234/test.2025.001",
            title="Different BioC Title",
        )
        mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing._ingest_article("1234567")

        saved_doc = db.save.call_args[0][0]
        self.assertEqual(saved_doc["title"], "Original Crossref Title")
        self.assertEqual(saved_doc["authors"], [{"name": "Smith, A."}])
        self.assertEqual(saved_doc["pmcid"], "PMC1234567")

    def test_enrich_skips_pdf_if_attached(self) -> None:
        """Existing doc with article.pdf does not re-download PDF."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        crossref_doc = {
            "_id": "crossref_doc_id_abc",
            "_rev": "1-abc",
            "doi": "10.1234/test.2025.001",
            "source": "crossref",
            "_attachments": {"article.pdf": {"content_type": "application/pdf"}},
        }
        db.save = MagicMock(return_value=("crossref_doc_id_abc", "2-def"))
        ing = _make_ingestor(db=db)
        ing._find_existing_by_doi = MagicMock(return_value=crossref_doc)

        bioc = _make_bioc_json(
            pmcid="PMC1234567",
            doi="10.1234/test.2025.001",
        )
        bioc_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=bioc_resp)

        ing._ingest_article("1234567")

        # PDF should not be re-downloaded — only bioc_json added
        saved_doc = db.save.call_args[0][0]
        self.assertIn("bioc_json", saved_doc)
        # Only one call to http_client.get (for BioC JSON, not PDF)
        self.assertEqual(ing.http_client.get.call_count, 1)


class TestDryRun(unittest.TestCase):
    """Test dry_run flag prevents writes but allows discovery."""

    def test_dry_run_does_not_save(self) -> None:
        """dry_run skips BioC fetch and db.save for new articles."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        ing = _make_ingestor(db=db, dry_run=True)
        ing.http_client.get = MagicMock()

        ing._ingest_article("1234567")

        ing.http_client.get.assert_not_called()
        db.save.assert_not_called()

    def test_dry_run_does_not_save_enrichment(self) -> None:
        """dry_run skips enrichment of existing documents."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        ing = _make_ingestor(db=db, dry_run=True)
        ing._find_existing_by_doi = MagicMock(return_value={"_id": "x"})
        ing.http_client.get = MagicMock()

        ing._ingest_article("1234567")

        db.save.assert_not_called()

    def test_dry_run_still_discovers(self) -> None:
        """dry_run still discovers PMCIDs via esearch."""
        ing = _make_ingestor(dry_run=True)
        esearch_resp = _make_esearch_response(["1111111", "2222222"])
        mock_resp = _make_mock_response(json_data=esearch_resp)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        pmcids = ing._discover_pmcids()

        self.assertEqual(pmcids, ["1111111", "2222222"])


class TestForceFlag(unittest.TestCase):
    """Test force flag overrides skip-existing logic."""

    def test_force_reprocesses_existing(self) -> None:
        """force=True re-fetches and saves even if bioc_json exists."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "_rev": "1-abc",
            "bioc_json": [{"old": "data"}],
            "source": "pmc",
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(return_value=existing_doc)
        db.save = MagicMock(return_value=(doc_id, "2-def"))

        ing = _make_ingestor(db=db, force=True)
        bioc = _make_bioc_json(pmcid="PMC1234567")
        mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing._ingest_article("1234567")

        db.save.assert_called()
        saved_doc = db.save.call_args[0][0]
        self.assertEqual(saved_doc["_id"], doc_id)
        # bioc_json should be the new data, not the old
        self.assertNotEqual(saved_doc["bioc_json"], [{"old": "data"}])

    def test_without_force_skips_existing(self) -> None:
        """Without force, existing doc with bioc_json is skipped (baseline)."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "bioc_json": [{"documents": []}],
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(return_value=existing_doc)
        ing = _make_ingestor(db=db, force=False)

        ing._ingest_article("1234567")

        db.save.assert_not_called()


class TestLimitFlag(unittest.TestCase):
    """Test limit flag caps articles processed."""

    def test_limit_caps_ingested_articles(self) -> None:
        """limit=1 processes only 1 article even though 3 are discovered."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        db.save = MagicMock(return_value=("id", "rev"))

        ing = _make_ingestor(db=db, limit=1)
        esearch_resp = _make_esearch_response(["1111111", "2222222", "3333333"])
        bioc = _make_bioc_json()

        esearch_mock_resp = _make_mock_response(json_data=esearch_resp)
        bioc_mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(
            side_effect=[esearch_mock_resp, bioc_mock_resp],
        )

        ing.ingest()

        self.assertEqual(db.save.call_count, 1)

    def test_no_limit_processes_all(self) -> None:
        """Without limit, all discovered articles are processed."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        db.save = MagicMock(return_value=("id", "rev"))

        ing = _make_ingestor(db=db)
        esearch_resp = _make_esearch_response(["1111111", "2222222"])
        bioc = _make_bioc_json()

        esearch_mock_resp = _make_mock_response(json_data=esearch_resp)
        bioc_mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(
            side_effect=[esearch_mock_resp, bioc_mock_resp, bioc_mock_resp],
        )

        ing.ingest()

        self.assertEqual(db.save.call_count, 2)


class TestDocIdsFlag(unittest.TestCase):
    """Test pmcids flag bypasses discovery."""

    def test_pmcids_skips_discovery(self) -> None:
        """pmcids processes specified PMCIDs without esearch."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        db.save = MagicMock(return_value=("id", "rev"))

        ing = _make_ingestor(db=db, pmcids=["1234567"])
        bioc = _make_bioc_json(pmcid="PMC1234567")
        mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing.ingest()

        # Only 1 GET call (bioc fetch, no esearch)
        self.assertEqual(ing.http_client.get.call_count, 1)
        db.save.assert_called()

    def test_pmcids_processes_all_specified(self) -> None:
        """All specified pmcids are processed."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        db.save = MagicMock(return_value=("id", "rev"))

        ing = _make_ingestor(db=db, pmcids=["1111111", "2222222"])
        bioc = _make_bioc_json()
        mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing.ingest()

        self.assertEqual(db.save.call_count, 2)


class TestBiocJsonAvailable(unittest.TestCase):
    """Test bioc_json_available tracking on fetch success/failure."""

    def test_bioc_json_available_true_on_success(self) -> None:
        """Successful BioC JSON fetch sets bioc_json_available=True."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        db.save = MagicMock(return_value=("id", "rev"))
        ing = _make_ingestor(db=db)

        bioc = _make_bioc_json(pmcid="PMC1234567")
        mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing._ingest_article("1234567")

        saved_doc = db.save.call_args[0][0]
        self.assertIs(saved_doc["bioc_json_available"], True)

    def test_bioc_json_available_false_on_http_error(self) -> None:
        """Failed BioC JSON fetch (HTTP error) sets bioc_json_available=False."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        db.save = MagicMock(return_value=("id", "rev"))
        ing = _make_ingestor(db=db)

        mock_resp = _make_mock_response(status_code=404, text="Not found")
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing._ingest_article("1234567")

        db.save.assert_called()
        saved_doc = db.save.call_args[0][0]
        self.assertIs(saved_doc["bioc_json_available"], False)
        self.assertNotIn("bioc_json", saved_doc)

    def test_bioc_json_available_false_on_json_error(self) -> None:
        """Failed BioC JSON fetch (invalid JSON) sets bioc_json_available=False."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        db.save = MagicMock(return_value=("id", "rev"))
        ing = _make_ingestor(db=db)

        mock_resp = _make_mock_response(
            status_code=200, text="<html>Error</html>",
        )
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing._ingest_article("1234567")

        db.save.assert_called()
        saved_doc = db.save.call_args[0][0]
        self.assertIs(saved_doc["bioc_json_available"], False)

    def test_skip_when_bioc_json_available_false(self) -> None:
        """Article with bioc_json_available=False is skipped by default."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "bioc_json_available": False,
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(return_value=existing_doc)
        ing = _make_ingestor(db=db)
        ing.http_client.get = MagicMock()

        ing._ingest_article("1234567")

        ing.http_client.get.assert_not_called()
        db.save.assert_not_called()

    def test_recheck_retries_when_bioc_json_available_false(self) -> None:
        """recheck_bioc_json=True retries even if bioc_json_available=False."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "_rev": "1-abc",
            "bioc_json_available": False,
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(return_value=existing_doc)
        db.save = MagicMock(return_value=(doc_id, "2-def"))

        ing = _make_ingestor(db=db, recheck_bioc_json=True)
        bioc = _make_bioc_json(pmcid="PMC1234567")
        mock_resp = _make_mock_response(json_data=bioc)
        ing.http_client.get = MagicMock(return_value=mock_resp)

        ing._ingest_article("1234567")

        ing.http_client.get.assert_called()
        db.save.assert_called()
        saved_doc = db.save.call_args[0][0]
        self.assertIs(saved_doc["bioc_json_available"], True)
        self.assertIn("bioc_json", saved_doc)

    def test_download_bioc_json_false_skips_fetch(self) -> None:
        """download_bioc_json=False skips BioC JSON fetch entirely."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        ing = _make_ingestor(db=db, download_bioc_json=False)
        ing.http_client.get = MagicMock()

        ing._ingest_article("1234567")

        ing.http_client.get.assert_not_called()
        db.save.assert_not_called()


class TestRateLimitConfig(unittest.TestCase):
    """Test rate limit configuration based on API key presence."""

    def setUp(self) -> None:
        import bin.env_config as env_config_module
        self._env_config_module = env_config_module
        self._orig_cache = env_config_module._skol_env_cache

    def tearDown(self) -> None:
        self._env_config_module._skol_env_cache = self._orig_cache

    def test_rate_limit_with_api_key(self) -> None:
        """With API key in .skol_env, rate limit allows 10 rps."""
        self._env_config_module._skol_env_cache = {
            'NCBI_API_KEY': 'test_key_12345',
        }
        ing = _make_ingestor(rate_limit_min_ms=100)
        self.assertEqual(ing.ncbi_api_key, "test_key_12345")

    def test_rate_limit_without_api_key(self) -> None:
        """Without API key, default rate limit is conservative."""
        self._env_config_module._skol_env_cache = {}
        ing = _make_ingestor()
        self.assertIsNone(ing.ncbi_api_key)


if __name__ == "__main__":
    unittest.main()
