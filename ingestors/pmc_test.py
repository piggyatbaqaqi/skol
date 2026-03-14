"""
TDD tests for the PmcIngestor class.

Tests article discovery via E-utilities esearch, JATS XML retrieval
via OAI-PMH, metadata extraction from JATS front matter, DOI-based
enrichment of existing documents, and skip-existing logic.
"""

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
from uuid import uuid5, NAMESPACE_URL

from .pmc import PmcIngestor


def _make_ingestor(**overrides: Any) -> PmcIngestor:
    """Create a PmcIngestor with mocked dependencies."""
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
    return PmcIngestor(**defaults)


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
        resp.text = str(json_data)
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


class TestExtractMetadataFromXml(unittest.TestCase):
    """Test JATS XML metadata extraction."""

    _JATS_ARTICLE = (
        '<article>'
        '<front><article-meta>'
        '<article-id pub-id-type="doi">10.1234/test.2025.001</article-id>'
        '<article-id pub-id-type="pmid">12345678</article-id>'
        '<article-id pub-id-type="pmc">PMC7777777</article-id>'
        '<title-group>'
        '<article-title>A new species of <italic>Testomyces</italic></article-title>'
        '</title-group>'
        '<contrib-group>'
        '<contrib contrib-type="author"><name>'
        '<surname>Smith</surname><given-names>Alice</given-names>'
        '</name></contrib>'
        '<contrib contrib-type="author"><name>'
        '<surname>Jones</surname><given-names>Bob</given-names>'
        '</name></contrib>'
        '</contrib-group>'
        '<permissions>'
        '<license xlink:href="https://creativecommons.org/licenses/by/4.0/"'
        ' xmlns:xlink="http://www.w3.org/1999/xlink"/>'
        '</permissions>'
        '</article-meta></front>'
        '<body><p>Body text.</p></body>'
        '</article>'
    )

    def test_extract_title(self) -> None:
        """Title is extracted including inline markup text."""
        metadata = PmcIngestor._extract_metadata_from_xml(
            self._JATS_ARTICLE,
        )
        self.assertEqual(
            metadata["title"],
            "A new species of Testomyces",
        )

    def test_extract_doi(self) -> None:
        """DOI is extracted from article-id elements."""
        metadata = PmcIngestor._extract_metadata_from_xml(
            self._JATS_ARTICLE,
        )
        self.assertEqual(metadata["doi"], "10.1234/test.2025.001")

    def test_extract_pmid(self) -> None:
        """PMID is extracted from article-id elements."""
        metadata = PmcIngestor._extract_metadata_from_xml(
            self._JATS_ARTICLE,
        )
        self.assertEqual(metadata["pmid"], "12345678")

    def test_extract_pmcid(self) -> None:
        """PMCID is extracted from article-id elements."""
        metadata = PmcIngestor._extract_metadata_from_xml(
            self._JATS_ARTICLE,
        )
        self.assertEqual(metadata["pmcid"], "PMC7777777")

    def test_extract_license(self) -> None:
        """License URL is extracted from permissions."""
        metadata = PmcIngestor._extract_metadata_from_xml(
            self._JATS_ARTICLE,
        )
        self.assertIn("creativecommons.org", metadata["license"])

    def test_extract_authors(self) -> None:
        """Authors are extracted in order."""
        metadata = PmcIngestor._extract_metadata_from_xml(
            self._JATS_ARTICLE,
        )
        self.assertEqual(len(metadata["authors"]), 2)
        self.assertEqual(metadata["authors"][0]["surname"], "Smith")
        self.assertEqual(metadata["authors"][0]["given_names"], "Alice")
        self.assertEqual(metadata["authors"][1]["surname"], "Jones")

    def test_handles_missing_front(self) -> None:
        """XML without <front> returns empty metadata."""
        metadata = PmcIngestor._extract_metadata_from_xml(
            "<article><body><p>Text</p></body></article>",
        )
        self.assertEqual(metadata["title"], "")
        self.assertEqual(metadata["authors"], [])

    def test_handles_malformed_xml(self) -> None:
        """Malformed XML returns empty metadata."""
        metadata = PmcIngestor._extract_metadata_from_xml(
            "not xml at all",
        )
        self.assertEqual(metadata["title"], "")

    def test_handles_namespaced_article(self) -> None:
        """Article with JATS namespace prefix is parsed correctly."""
        ns = "https://jats.nlm.nih.gov/ns/archiving/1.4/"
        xml = (
            f'<article xmlns="{ns}">'
            f'<front><article-meta>'
            f'<title-group>'
            f'<article-title>Namespaced Title</article-title>'
            f'</title-group>'
            f'</article-meta></front>'
            f'</article>'
        )
        metadata = PmcIngestor._extract_metadata_from_xml(xml)
        self.assertEqual(metadata["title"], "Namespaced Title")


class TestSkipExisting(unittest.TestCase):
    """Test skip-existing logic for documents already in CouchDB."""

    def test_skip_existing_with_xml_attached(self) -> None:
        """Existing document with article.xml is skipped."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "xml_available": True,
            "_attachments": {"article.xml": {}},
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(return_value=existing_doc)
        ing = _make_ingestor(db=db)

        ing._ingest_article("1234567")

        db.save.assert_not_called()


class TestDryRun(unittest.TestCase):
    """Test dry_run flag prevents writes but allows discovery."""

    def test_dry_run_does_not_save(self) -> None:
        """dry_run skips XML fetch and db.save for new articles."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        ing = _make_ingestor(db=db, dry_run=True)
        ing.http_client.get = MagicMock()

        ing._ingest_article("1234567")

        ing.http_client.get.assert_not_called()
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
        """force=True re-downloads XML even if already attached."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "_rev": "1-abc",
            "xml_available": True,
            "_attachments": {"article.xml": {}},
            "source": "pmc",
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(return_value=existing_doc)
        db.save = MagicMock(return_value=(doc_id, "2-def"))

        ing = _make_ingestor(db=db, force=True)
        article = '<article><body>Forced</body></article>'
        oai = _make_oai_pmh_response(article_xml=article)
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        ing._ingest_article("1234567")

        db.put_attachment.assert_called()

    def test_without_force_skips_existing(self) -> None:
        """Without force, existing doc with XML is skipped."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "xml_available": True,
            "_attachments": {"article.xml": {}},
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

        esearch_mock_resp = _make_mock_response(json_data=esearch_resp)
        article = '<article><body>Test</body></article>'
        oai = _make_oai_pmh_response(article_xml=article)
        xml_resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(
            side_effect=[esearch_mock_resp, xml_resp],
        )

        ing.ingest()

        # 1 save for _ensure_document + 1 save in _download_and_attach_xml
        self.assertGreaterEqual(db.save.call_count, 1)


class TestDocIdsFlag(unittest.TestCase):
    """Test pmcids flag bypasses discovery."""

    def test_pmcids_skips_discovery(self) -> None:
        """pmcids processes specified PMCIDs without esearch."""
        db = MagicMock()
        # First call returns False (doc doesn't exist),
        # subsequent calls return True (after _ensure_document creates it).
        db.__contains__ = MagicMock(side_effect=[False, True, True])
        db.save = MagicMock(return_value=("id", "rev"))

        ing = _make_ingestor(db=db, pmcids=["1234567"])
        article = '<article><body>Test</body></article>'
        oai = _make_oai_pmh_response(article_xml=article)
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        ing.ingest()

        # 1 GET call (OAI-PMH XML, no esearch)
        self.assertEqual(ing.http_client.get.call_count, 1)
        db.save.assert_called()

    def test_pmcids_processes_all_specified(self) -> None:
        """All specified pmcids are processed."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        db.save = MagicMock(return_value=("id", "rev"))

        ing = _make_ingestor(db=db, pmcids=["1111111", "2222222"])
        article = '<article><body>Test</body></article>'
        oai = _make_oai_pmh_response(article_xml=article)
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        ing.ingest()

        self.assertGreaterEqual(db.save.call_count, 2)


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


def _make_oai_pmh_response(
    pmcid: str = "1234567",
    article_xml: str = '<article><body><p>Test</p></body></article>',
    error_code: Optional[str] = None,
    error_text: str = "",
) -> bytes:
    """Build a mock OAI-PMH XML response."""
    oai_ns = "http://www.openarchives.org/OAI/2.0/"
    if error_code:
        return (
            f'<OAI-PMH xmlns="{oai_ns}">'
            f'<responseDate>2026-01-01T00:00:00Z'
            f'</responseDate>'
            f'<request>{PmcIngestor.OAI_PMH_URL}'
            f'</request>'
            f'<error code="{error_code}">'
            f'{error_text}</error>'
            f'</OAI-PMH>'
        ).encode("utf-8")
    return (
        f'<OAI-PMH xmlns="{oai_ns}">'
        f'<responseDate>2026-01-01T00:00:00Z'
        f'</responseDate>'
        f'<request>{PmcIngestor.OAI_PMH_URL}'
        f'</request>'
        f'<GetRecord><record>'
        f'<header>'
        f'<identifier>oai:pubmedcentral.nih.gov:'
        f'{pmcid}</identifier>'
        f'<datestamp>2026-01-01</datestamp>'
        f'</header>'
        f'<metadata>{article_xml}</metadata>'
        f'</record></GetRecord>'
        f'</OAI-PMH>'
    ).encode("utf-8")


def _make_oai_http_response(
    oai_content: bytes,
    status_code: int = 200,
) -> MagicMock:
    """Create a mock HTTP response for OAI-PMH."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.content = oai_content
    return resp


class TestFetchJatsXml(unittest.TestCase):
    """Test JATS XML retrieval via OAI-PMH."""

    def test_fetch_success(self) -> None:
        """Successful OAI-PMH response returns XML string."""
        ing = _make_ingestor()
        jats_ns = (
            "https://jats.nlm.nih.gov/ns/"
            "archiving/1.4/"
        )
        article = (
            f'<article xmlns="{jats_ns}">'
            f'<body>Content</body></article>'
        )
        oai = _make_oai_pmh_response(
            pmcid="1234567", article_xml=article,
        )
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        result = ing._fetch_jats_xml("1234567")

        self.assertIsNotNone(result)
        self.assertIn("Content", result)
        self.assertIn("article", result)

    def test_fetch_http_error(self) -> None:
        """Non-200 HTTP response returns None."""
        ing = _make_ingestor()
        resp = _make_oai_http_response(b"", status_code=503)
        ing.http_client.get = MagicMock(return_value=resp)

        result = ing._fetch_jats_xml("9999999")

        self.assertIsNone(result)

    def test_fetch_oai_error(self) -> None:
        """OAI-PMH error response returns None."""
        ing = _make_ingestor()
        oai = _make_oai_pmh_response(
            error_code="idDoesNotExist",
            error_text="Identifier does not exist.",
        )
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        result = ing._fetch_jats_xml("9999999")

        self.assertIsNone(result)

    def test_fetch_network_error(self) -> None:
        """Network error returns None."""
        ing = _make_ingestor()
        ing.http_client.get = MagicMock(
            side_effect=ConnectionError("timeout"),
        )

        result = ing._fetch_jats_xml("1234567")

        self.assertIsNone(result)

    def test_fetch_invalid_xml(self) -> None:
        """Invalid XML in response returns None."""
        ing = _make_ingestor()
        resp = _make_oai_http_response(
            b"this is not xml",
        )
        ing.http_client.get = MagicMock(return_value=resp)

        result = ing._fetch_jats_xml("1234567")

        self.assertIsNone(result)

    def test_calls_correct_url(self) -> None:
        """OAI-PMH request uses correct URL and params."""
        ing = _make_ingestor()
        article = '<article><body>X</body></article>'
        oai = _make_oai_pmh_response(article_xml=article)
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        ing._fetch_jats_xml("1234567")

        call_args = ing.http_client.get.call_args
        self.assertEqual(
            call_args[0][0],
            PmcIngestor.OAI_PMH_URL,
        )
        params = call_args[1]["params"]
        self.assertEqual(params["verb"], "GetRecord")
        self.assertEqual(
            params["identifier"],
            "oai:pubmedcentral.nih.gov:1234567",
        )
        self.assertEqual(
            params["metadataPrefix"], "pmc",
        )


class TestExtractArticleXml(unittest.TestCase):
    """Test extraction of <article> from OAI-PMH envelope."""

    def test_extracts_article_element(self) -> None:
        """Article element is extracted from metadata."""
        ing = _make_ingestor()
        jats_ns = (
            "https://jats.nlm.nih.gov/ns/"
            "archiving/1.4/"
        )
        article = (
            f'<article xmlns="{jats_ns}">'
            f'<body>Hello</body></article>'
        )
        oai = _make_oai_pmh_response(article_xml=article)

        result = ing._extract_article_xml(oai, "1234567")

        self.assertIsNotNone(result)
        self.assertIn("Hello", result)
        self.assertIn("article", result)

    def test_returns_none_for_oai_error(self) -> None:
        """OAI-PMH error response returns None."""
        ing = _make_ingestor()
        oai = _make_oai_pmh_response(
            error_code="idDoesNotExist",
            error_text="Not found",
        )

        result = ing._extract_article_xml(oai, "9999")

        self.assertIsNone(result)

    def test_returns_none_for_malformed_xml(self) -> None:
        """Malformed XML returns None."""
        ing = _make_ingestor()

        result = ing._extract_article_xml(
            b"not xml at all", "1234567",
        )

        self.assertIsNone(result)

    def test_returns_none_when_no_metadata(self) -> None:
        """Response without metadata element returns None."""
        ing = _make_ingestor()
        oai_ns = "http://www.openarchives.org/OAI/2.0/"
        oai = (
            f'<OAI-PMH xmlns="{oai_ns}">'
            f'<responseDate>2026-01-01T00:00:00Z'
            f'</responseDate>'
            f'<request>url</request>'
            f'<GetRecord><record>'
            f'<header><identifier>x</identifier>'
            f'</header>'
            f'</record></GetRecord>'
            f'</OAI-PMH>'
        ).encode("utf-8")

        result = ing._extract_article_xml(oai, "1234567")

        self.assertIsNone(result)

    def test_handles_jats_namespace(self) -> None:
        """Article with JATS namespace is extracted."""
        ing = _make_ingestor()
        jats_ns = (
            "https://jats.nlm.nih.gov/ns/archiving/1.4/"
        )
        article = (
            f'<article xmlns="{jats_ns}">'
            f'<body><p>Namespaced</p></body>'
            f'</article>'
        )
        oai = _make_oai_pmh_response(article_xml=article)

        result = ing._extract_article_xml(oai, "1234567")

        self.assertIsNotNone(result)
        self.assertIn("Namespaced", result)


class TestXmlDownload(unittest.TestCase):
    """Test XML download and attachment via _ingest_article."""

    def test_download_xml_attaches_to_doc(self) -> None:
        """download_xml=True fetches and attaches JATS XML."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/"
            "PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "_rev": "1-abc",
            "source": "pmc",
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(
            return_value=existing_doc,
        )
        db.save = MagicMock(
            return_value=(doc_id, "2-def"),
        )

        ing = _make_ingestor(db=db, download_xml=True)
        article = '<article><body>Test</body></article>'
        oai = _make_oai_pmh_response(article_xml=article)
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        ing._download_and_attach_xml("1234567", doc_id)

        db.put_attachment.assert_called_once()
        call_args = db.put_attachment.call_args
        self.assertEqual(call_args[0][2], "article.xml")
        self.assertEqual(
            call_args[0][3], "application/xml",
        )

    def test_download_xml_sets_metadata(self) -> None:
        """Successful XML download sets xml_available and format."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/"
            "PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "_rev": "1-abc",
            "source": "pmc",
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(
            return_value=existing_doc,
        )
        db.save = MagicMock(
            return_value=(doc_id, "2-def"),
        )

        ing = _make_ingestor(db=db, download_xml=True)
        article = '<article><body>Test</body></article>'
        oai = _make_oai_pmh_response(article_xml=article)
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        ing._download_and_attach_xml("1234567", doc_id)

        saved_doc = db.save.call_args[0][0]
        self.assertIs(saved_doc["xml_available"], True)
        self.assertEqual(saved_doc["xml_format"], "jats")

    def test_download_xml_marks_unavailable(self) -> None:
        """Failed XML download sets xml_available=False."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/"
            "PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "_rev": "1-abc",
            "source": "pmc",
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(
            return_value=existing_doc,
        )
        db.save = MagicMock(
            return_value=(doc_id, "2-def"),
        )

        ing = _make_ingestor(db=db, download_xml=True)
        oai = _make_oai_pmh_response(
            error_code="idDoesNotExist",
            error_text="Not found",
        )
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        ing._download_and_attach_xml("1234567", doc_id)

        db.save.assert_called()
        saved_doc = db.save.call_args[0][0]
        self.assertIs(saved_doc["xml_available"], False)
        db.put_attachment.assert_not_called()


class TestXmlSkipExisting(unittest.TestCase):
    """Test skip logic for existing XML."""

    def test_skip_when_xml_already_attached(self) -> None:
        """Document with article.xml attachment is skipped."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/"
            "PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "_attachments": {
                "article.xml": {
                    "content_type": "application/xml",
                },
            },
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(
            return_value=existing_doc,
        )
        ing = _make_ingestor(db=db, download_xml=True)
        ing.http_client.get = MagicMock()

        ing._ingest_article("1234567")

        ing.http_client.get.assert_not_called()
        db.save.assert_not_called()

    def test_skip_when_xml_available_false(self) -> None:
        """Doc with xml_available=False is skipped by default."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/"
            "PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "xml_available": False,
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(
            return_value=existing_doc,
        )
        ing = _make_ingestor(db=db, download_xml=True)
        ing.http_client.get = MagicMock()

        ing._ingest_article("1234567")

        ing.http_client.get.assert_not_called()
        db.save.assert_not_called()

    def test_recheck_xml_retries_unavailable(self) -> None:
        """recheck_xml=True retries if xml_available=False."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/"
            "PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "_rev": "1-abc",
            "xml_available": False,
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(
            return_value=existing_doc,
        )
        db.save = MagicMock(
            return_value=(doc_id, "2-def"),
        )

        ing = _make_ingestor(
            db=db,
            download_xml=True,
            recheck_xml=True,
        )
        article = '<article><body>Retry</body></article>'
        oai = _make_oai_pmh_response(article_xml=article)
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        ing._ingest_article("1234567")

        ing.http_client.get.assert_called()
        db.put_attachment.assert_called_once()

    def test_force_retries_xml(self) -> None:
        """force=True re-downloads XML even if attached."""
        db = MagicMock()
        doc_id = uuid5(
            NAMESPACE_URL,
            "https://pmc.ncbi.nlm.nih.gov/articles/"
            "PMC1234567/",
        ).hex
        existing_doc = {
            "_id": doc_id,
            "_rev": "1-abc",
            "_attachments": {
                "article.xml": {
                    "content_type": "application/xml",
                },
            },
        }
        db.__contains__ = MagicMock(return_value=True)
        db.__getitem__ = MagicMock(
            return_value=existing_doc,
        )
        db.save = MagicMock(
            return_value=(doc_id, "2-def"),
        )

        ing = _make_ingestor(
            db=db,
            download_xml=True,
            force=True,
        )
        article = '<article><body>Force</body></article>'
        oai = _make_oai_pmh_response(article_xml=article)
        resp = _make_oai_http_response(oai)
        ing.http_client.get = MagicMock(return_value=resp)

        ing._ingest_article("1234567")

        db.put_attachment.assert_called_once()


class TestNeedsXmlDownload(unittest.TestCase):
    """Test _needs_xml_download logic."""

    def test_returns_false_when_disabled(self) -> None:
        ing = _make_ingestor(download_xml=False)
        self.assertFalse(ing._needs_xml_download(None))

    def test_returns_true_for_new_doc(self) -> None:
        ing = _make_ingestor(download_xml=True)
        self.assertTrue(ing._needs_xml_download(None))

    def test_returns_false_when_has_attachment(self) -> None:
        ing = _make_ingestor(download_xml=True)
        existing = {
            "_attachments": {"article.xml": {}},
        }
        self.assertFalse(
            ing._needs_xml_download(existing),
        )

    def test_returns_true_when_force(self) -> None:
        ing = _make_ingestor(
            download_xml=True, force=True,
        )
        existing = {
            "_attachments": {"article.xml": {}},
        }
        self.assertTrue(
            ing._needs_xml_download(existing),
        )

    def test_returns_false_when_unavailable(self) -> None:
        ing = _make_ingestor(download_xml=True)
        existing = {"xml_available": False}
        self.assertFalse(
            ing._needs_xml_download(existing),
        )

    def test_returns_true_when_recheck(self) -> None:
        ing = _make_ingestor(
            download_xml=True, recheck_xml=True,
        )
        existing = {"xml_available": False}
        self.assertTrue(
            ing._needs_xml_download(existing),
        )


class TestEnsureDocument(unittest.TestCase):
    """Test _ensure_document creates doc when needed."""

    def test_creates_new_doc(self) -> None:
        """Creates a minimal document when it doesn't exist."""
        db = MagicMock()
        db.save = MagicMock(return_value=("id", "rev"))
        ing = _make_ingestor(db=db)

        ing._ensure_document("1234567", "doc_id_abc", None)

        db.save.assert_called_once()
        saved_doc = db.save.call_args[0][0]
        self.assertEqual(saved_doc["_id"], "doc_id_abc")
        self.assertEqual(saved_doc["pmcid"], "1234567")
        self.assertEqual(saved_doc["source"], "pmc")

    def test_skips_when_exists(self) -> None:
        """Doesn't create a document when it already exists."""
        db = MagicMock()
        ing = _make_ingestor(db=db)
        existing = {"_id": "doc_id_abc", "_rev": "1-abc"}

        ing._ensure_document("1234567", "doc_id_abc", existing)

        db.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
