"""
Ingestor for PubMed Central articles via NCBI E-utilities, BioC JSON API,
and the OAI-PMH service.

Discovers articles using E-utilities esearch, retrieves full text as BioC JSON
from the PMC Open Access subset, and enriches existing documents (e.g., from
Crossref) with PMC full text when a DOI match is found.

Optionally downloads JATS XML full text via the PMC OAI-PMH service.
"""

import json
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, Dict, List, Optional
from uuid import uuid5, NAMESPACE_URL

from bin.env_config import get_env_config
from .ingestor import Ingestor
from .timestamps import set_timestamps


class PmcBiocIngestor(Ingestor):
    """
    Ingestor for PubMed Central articles using BioC JSON format.

    Discovers articles via E-utilities esearch with a journal search term,
    retrieves BioC JSON from the PMC Open Access subset, and stores
    documents in CouchDB. When a DOI match is found with an existing
    document (e.g., from Crossref), the existing document is enriched
    with BioC JSON data rather than creating a duplicate.

    Optionally downloads JATS XML full text via the PMC OAI-PMH service.
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BIOC_URL = (
        "https://www.ncbi.nlm.nih.gov"
        "/research/bionlp/RESTful/pmcoa.cgi/BioC_json"
    )
    OAI_PMH_URL = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
    PMC_ARTICLE_URL_TEMPLATE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC{}/"

    def __init__(
        self,
        journal_search_term: str,
        max_articles: Optional[int] = None,
        dry_run: Optional[bool] = None,
        force: Optional[bool] = None,
        limit: Optional[int] = None,
        pmcids: Optional[List[str]] = None,
        skip_existing: Optional[bool] = None,
        incremental: Optional[bool] = None,
        download_bioc_json: bool = True,
        recheck_bioc_json: bool = False,
        download_xml: bool = False,
        recheck_xml: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.journal_search_term = journal_search_term
        self.max_articles = max_articles
        config = get_env_config()
        self.ncbi_api_key = config.get('ncbi_api_key')
        self.dry_run = dry_run if dry_run is not None else config.get('dry_run', False)
        self.force = force if force is not None else config.get('force', False)
        self.limit = limit if limit is not None else config.get('limit')
        self.pmcids = pmcids if pmcids is not None else config.get('pmcids')
        self.skip_existing = (
            skip_existing if skip_existing is not None
            else config.get('skip_existing', False)
        )
        self.incremental = (
            incremental if incremental is not None
            else config.get('incremental', False)
        )
        self.download_bioc_json = download_bioc_json
        self.recheck_bioc_json = recheck_bioc_json
        self.download_xml = download_xml
        self.recheck_xml = recheck_xml

    def ingest(self) -> None:
        """Discover and ingest articles from PMC."""
        pmcids = (self.pmcids if self.pmcids is not None
                  else self._discover_pmcids())

        if self.verbosity >= 2:
            print(f"Found {len(pmcids)} article(s) to process")

        processed = 0
        for pmcid in pmcids:
            if self.limit is not None and processed >= self.limit:
                if self.verbosity >= 2:
                    print(f"Reached limit of {self.limit}")
                break
            self._ingest_article(pmcid)
            processed += 1

    def _discover_pmcids(self) -> List[str]:
        """Discover PMCIDs via E-utilities esearch, handling pagination.

        Restricts to the Open Access subset since BioC JSON is only
        available for OA articles.
        """
        all_ids: List[str] = []
        retstart = 0
        # The BioC JSON API (pmcoa.cgi) only serves OA articles,
        # so restrict discovery to match.
        term = f'{self.journal_search_term} AND "open access"[filter]'

        while True:
            params: Dict[str, Any] = {
                "db": "pmc",
                "term": term,
                "retmode": "json",
                "retmax": 500,
                "retstart": retstart,
            }
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            resp = self.http_client.get(self.ESEARCH_URL, params=params)
            data = resp.json()
            result = data["esearchresult"]

            all_ids.extend(result["idlist"])

            count = int(result["count"])
            retstart += int(result["retmax"])

            if retstart >= count:
                break

        if self.max_articles is not None:
            all_ids = all_ids[:self.max_articles]

        return all_ids

    def _fetch_bioc_json(self, pmcid: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch BioC JSON for a PMC article.

        Args:
            pmcid: Numeric PMCID (without 'PMC' prefix).

        Returns:
            Parsed BioC JSON list, or None if unavailable.
        """
        url = f"{self.BIOC_URL}/PMC{pmcid}/unicode"
        resp = self.http_client.get(url)

        if not resp.ok:
            return None

        try:
            return resp.json()
        except (ValueError, json.JSONDecodeError):
            return None

    def _fetch_jats_xml(self, pmcid: str) -> Optional[str]:
        """Fetch JATS XML for a PMC article via the OAI-PMH service.

        Requests the full-text JATS XML using the ``pmc`` metadata
        prefix and extracts the ``<article>`` element from the
        OAI-PMH envelope.

        Args:
            pmcid: Numeric PMCID (without 'PMC' prefix).

        Returns:
            JATS XML string (the ``<article>`` element), or None
            if unavailable.
        """
        params = {
            "verb": "GetRecord",
            "identifier": f"oai:pubmedcentral.nih.gov:{pmcid}",
            "metadataPrefix": "pmc",
        }
        try:
            resp = self.http_client.get(
                self.OAI_PMH_URL, params=params,
            )
        except Exception as exc:
            if self.verbosity >= 1:
                print(f"  OAI-PMH error for PMC{pmcid}: {exc}")
            return None

        if not resp.ok:
            if self.verbosity >= 1:
                print(f"  OAI-PMH unavailable for PMC{pmcid} "
                      f"(HTTP {resp.status_code})")
            return None

        return self._extract_article_xml(resp.content, pmcid)

    def _extract_article_xml(
        self, oai_content: bytes, pmcid: str,
    ) -> Optional[str]:
        """Extract the JATS <article> element from an OAI-PMH response.

        Args:
            oai_content: Raw OAI-PMH XML response bytes.
            pmcid: PMCID for error messages.

        Returns:
            JATS XML string, or None if not found or on error.
        """
        try:
            root = ET.fromstring(oai_content)
        except ET.ParseError as exc:
            if self.verbosity >= 1:
                print(f"  OAI-PMH XML parse error for PMC{pmcid}: "
                      f"{exc}")
            return None

        oai_ns = "http://www.openarchives.org/OAI/2.0/"

        # Check for OAI-PMH error
        error = root.find(f"{{{oai_ns}}}error")
        if error is not None:
            code = error.get("code", "unknown")
            if self.verbosity >= 1:
                print(f"  OAI-PMH error for PMC{pmcid}: "
                      f"{code} — {error.text}")
            return None

        # Find the <article> element inside <metadata>
        metadata = root.find(
            f".//{{{oai_ns}}}metadata"
        )
        if metadata is None:
            if self.verbosity >= 1:
                print(f"  No metadata in OAI-PMH response "
                      f"for PMC{pmcid}")
            return None

        # The <article> element may be in any JATS namespace
        article = None
        for child in metadata:
            local_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local_name == "article":
                article = child
                break

        if article is None:
            if self.verbosity >= 1:
                print(f"  No <article> element in OAI-PMH "
                      f"response for PMC{pmcid}")
            return None

        return ET.tostring(
            article, encoding="unicode", xml_declaration=False,
        )

    def _extract_metadata(self, bioc: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata from BioC JSON.

        Returns dict with keys: title, doi, pmid, pmcid, license, authors.
        """
        doc = bioc[0]["documents"][0]

        # Find the front (title) passage
        front = None
        for passage in doc["passages"]:
            if passage.get("infons", {}).get("type") == "front":
                front = passage
                break

        infons = front["infons"] if front else {}

        # Extract authors from name_N infons
        authors: List[Dict[str, str]] = []
        i = 0
        while f"name_{i}" in infons:
            name_str = infons[f"name_{i}"]
            parts: Dict[str, str] = {}
            for part in name_str.split(";"):
                key, _, value = part.partition(":")
                parts[key] = value
            authors.append({
                "surname": parts.get("surname", ""),
                "given_names": parts.get("given-names", ""),
            })
            i += 1

        return {
            "title": front["text"] if front else "",
            "doi": infons.get("article-id_doi", ""),
            "pmid": infons.get("article-id_pmid", ""),
            "pmcid": infons.get("article-id_pmc", ""),
            "license": doc.get("infons", {}).get("license", ""),
            "authors": authors,
        }

    def _make_doc_id(self, pmcid: str) -> str:
        """Create a deterministic document ID from a PMCID."""
        url = self.PMC_ARTICLE_URL_TEMPLATE.format(pmcid)
        return uuid5(NAMESPACE_URL, url).hex

    def _find_existing_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Find an existing document by DOI in CouchDB.

        Args:
            doi: DOI string to search for.

        Returns:
            Existing document dict, or None if not found.
        """
        # TODO: implement DOI lookup via CouchDB view
        return None

    def _needs_bioc_download(
        self, existing: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if BioC JSON needs to be downloaded for a document."""
        if not self.download_bioc_json:
            return False
        if existing is None:
            return True
        if self.force:
            return True
        if "bioc_json" in existing:
            return False
        if (not self.recheck_bioc_json
                and existing.get("bioc_json_available") is False):
            return False
        return True

    def _needs_xml_download(
        self, existing: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if JATS XML needs to be downloaded for a document."""
        if not self.download_xml:
            return False
        if existing is None:
            return True
        if self.force:
            return True
        attachments = existing.get("_attachments", {})
        if "article.xml" in attachments:
            return False
        if (not self.recheck_xml
                and existing.get("xml_available") is False):
            return False
        return True

    def _process_bioc_json(
        self,
        pmcid: str,
        doc_id: str,
        existing: Optional[Dict[str, Any]],
    ) -> None:
        """Fetch and store BioC JSON for a PMC article.

        Creates or updates the CouchDB document with BioC JSON data.
        If a DOI-matched document exists from another source (e.g.,
        Crossref), enriches it instead of creating a duplicate.
        """
        if self.verbosity >= 3:
            print(f"Fetching BioC JSON for PMC{pmcid}")

        bioc = self._fetch_bioc_json(pmcid)
        if bioc is None:
            if self.verbosity >= 1:
                print(f"  BioC JSON unavailable for PMC{pmcid}")
            if existing is not None:
                existing["bioc_json_available"] = False
                self.db.save(existing)
            else:
                doc: Dict[str, Any] = {
                    "_id": doc_id,
                    "bioc_json_available": False,
                    "source": "pmc",
                }
                set_timestamps(doc, is_new=True)
                self.db.save(doc)
            return

        metadata = self._extract_metadata(bioc)
        doi = metadata.get("doi", "")

        # Check for existing doc by DOI (e.g., from Crossref)
        existing_doc = (
            self._find_existing_by_doi(doi) if doi else None
        )

        if existing_doc is not None:
            # Enrich existing document — preserve its title/authors
            existing_doc["bioc_json"] = bioc
            existing_doc["pmcid"] = metadata["pmcid"]
            existing_doc["bioc_json_available"] = True
            self.db.save(existing_doc)
            if self.verbosity >= 2:
                print(f"Enriched existing doc with BioC JSON: "
                      f"PMC{pmcid}")
        elif existing is not None:
            # Update existing PMC document (force reprocess)
            existing["bioc_json"] = bioc
            existing["bioc_json_available"] = True
            set_timestamps(existing)
            self.db.save(existing)
            if self.verbosity >= 2:
                print(f"Updated BioC JSON: PMC{pmcid}")
        else:
            # Create new document
            doc = {
                "_id": doc_id,
                "bioc_json": bioc,
                "bioc_json_available": True,
                "title": metadata["title"],
                "doi": doi,
                "pmid": metadata["pmid"],
                "pmcid": metadata["pmcid"],
                "license": metadata["license"],
                "authors": metadata["authors"],
                "source": "pmc",
            }
            set_timestamps(doc, is_new=True)
            self.db.save(doc)
            if self.verbosity >= 2:
                print(f"Added: PMC{pmcid} — "
                      f"{metadata['title'][:60]}")

    def _download_and_attach_xml(
        self, pmcid: str, doc_id: str,
    ) -> None:
        """Download and attach JATS XML for a PMC article.

        Fetches JATS XML via the OAI-PMH service and stores it as
        an ``article.xml`` CouchDB attachment. Tracks
        ``xml_available`` to avoid retrying known-unavailable
        articles.
        """
        if doc_id not in self.db:
            if self.verbosity >= 1:
                print(f"  Cannot attach XML for PMC{pmcid}: "
                      "document not found")
            return

        if self.verbosity >= 3:
            print(f"Fetching JATS XML for PMC{pmcid}")

        xml_string = self._fetch_jats_xml(pmcid)
        if xml_string is None:
            doc = self.db[doc_id]
            doc["xml_available"] = False
            self.db.save(doc)
            return

        doc = self.db[doc_id]
        doc["xml_available"] = True
        doc["xml_format"] = "jats"
        self.db.save(doc)
        doc = self.db[doc_id]
        self.db.put_attachment(
            doc,
            BytesIO(xml_string.encode("utf-8")),
            "article.xml",
            "application/xml",
        )
        if self.verbosity >= 2:
            print(f"  Attached JATS XML for PMC{pmcid}")

    def _ingest_article(self, pmcid: str) -> None:
        """Ingest a single article by PMCID.

        Determines what work is needed (BioC JSON, XML, or both)
        and processes accordingly. Skips if the document already
        has the requested data (unless force is set).
        """
        doc_id = self._make_doc_id(pmcid)

        # Check for existing document
        existing = (
            self.db[doc_id] if doc_id in self.db else None
        )

        # Determine what work is needed
        needs_bioc = self._needs_bioc_download(existing)
        needs_xml = self._needs_xml_download(existing)

        if not needs_bioc and not needs_xml:
            if self.verbosity >= 2:
                print(f"Skipping PMC{pmcid} "
                      "(already complete)")
            return

        if self.dry_run:
            if self.verbosity >= 2:
                print(f"Dry run: PMC{pmcid}")
            return

        if needs_bioc:
            self._process_bioc_json(pmcid, doc_id, existing)

        if needs_xml:
            self._download_and_attach_xml(pmcid, doc_id)
