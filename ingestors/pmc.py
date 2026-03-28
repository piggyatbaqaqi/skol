"""
Ingestor for PubMed Central articles via NCBI E-utilities and OAI-PMH.

Discovers articles using E-utilities esearch, retrieves JATS XML full text
via the PMC OAI-PMH service, and enriches existing documents (e.g., from
Crossref) with PMC full text when a DOI match is found.

Optionally downloads plaintext via E-utilities efetch.
"""

import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, Dict, List, Optional
from uuid import uuid5, NAMESPACE_URL

from bin.env_config import get_env_config
from .ingestor import Ingestor
from .timestamps import set_timestamps

# Keep old name available for backwards compatibility with existing code
# that references PmcBiocIngestor (e.g., publications.py configs).
# Will be assigned after class definition.


class PmcIngestor(Ingestor):
    """
    Ingestor for PubMed Central articles using JATS XML.

    Discovers articles via E-utilities esearch with a journal search term,
    retrieves JATS XML from PMC via OAI-PMH, and stores documents in
    CouchDB. When a DOI match is found with an existing document (e.g.,
    from Crossref), the existing document is enriched with PMC data
    rather than creating a duplicate.
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
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
        download_xml: bool = True,
        recheck_xml: bool = False,
        download_text: bool = False,
        recheck_text: bool = False,
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
        self.download_xml = download_xml
        self.recheck_xml = recheck_xml
        self.download_text = download_text
        self.recheck_text = recheck_text

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

        Restricts to the Open Access subset since OAI-PMH full text
        is only available for OA articles.
        """
        all_ids: List[str] = []
        retstart = 0
        # OAI-PMH full text is only available for OA articles.
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

    @staticmethod
    def _extract_metadata_from_xml(
        xml_string: str,
    ) -> Dict[str, Any]:
        """Extract metadata from JATS XML front matter.

        Parses the ``<front>`` element to extract title, DOI, PMID,
        PMCID, license info, and authors.

        Args:
            xml_string: JATS XML article string.

        Returns:
            Dict with keys: title, doi, pmid, pmcid, license, authors.
        """
        metadata: Dict[str, Any] = {
            "title": "",
            "doi": "",
            "pmid": "",
            "pmcid": "",
            "license": "",
            "authors": [],
        }

        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError:
            return metadata

        # Handle potential namespace prefix
        ns = ""
        if "}" in root.tag:
            ns = root.tag.split("}")[0] + "}"

        front = root.find(f".//{ns}front")
        if front is None:
            return metadata

        article_meta = front.find(f"{ns}article-meta")
        if article_meta is None:
            return metadata

        # Title
        title_group = article_meta.find(f"{ns}title-group")
        if title_group is not None:
            article_title = title_group.find(f"{ns}article-title")
            if article_title is not None:
                # Get all text including from child elements
                metadata["title"] = "".join(
                    article_title.itertext()
                ).strip()

        # Article IDs (DOI, PMID, PMCID)
        for article_id in article_meta.findall(f"{ns}article-id"):
            pub_id_type = article_id.get("pub-id-type", "")
            text = (article_id.text or "").strip()
            if pub_id_type == "doi":
                metadata["doi"] = text
            elif pub_id_type == "pmid":
                metadata["pmid"] = text
            elif pub_id_type == "pmc":
                metadata["pmcid"] = text

        # License
        permissions = article_meta.find(f"{ns}permissions")
        if permissions is not None:
            license_el = permissions.find(f"{ns}license")
            if license_el is not None:
                href = license_el.get(
                    "{http://www.w3.org/1999/xlink}href", ""
                )
                if href:
                    metadata["license"] = href
                else:
                    metadata["license"] = "".join(
                        license_el.itertext()
                    ).strip()[:200]

        # Authors
        authors: List[Dict[str, str]] = []
        for contrib_group in article_meta.findall(
            f"{ns}contrib-group"
        ):
            for contrib in contrib_group.findall(f"{ns}contrib"):
                if contrib.get("contrib-type") != "author":
                    continue
                name_el = contrib.find(f"{ns}name")
                if name_el is None:
                    continue
                surname_el = name_el.find(f"{ns}surname")
                given_el = name_el.find(f"{ns}given-names")
                surname = (
                    surname_el.text or ""
                ) if surname_el is not None else ""
                given = (
                    given_el.text or ""
                ) if given_el is not None else ""
                authors.append({
                    "surname": surname,
                    "given_names": given,
                })
        metadata["authors"] = authors

        return metadata

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

    def _needs_text_download(
        self, existing: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if plaintext needs to be downloaded for a document."""
        if not self.download_text:
            return False
        if existing is None:
            return True
        if self.force:
            return True
        attachments = existing.get("_attachments", {})
        if "article.txt" in attachments:
            return False
        if (not self.recheck_text
                and existing.get("text_available") is False):
            return False
        return True

    def _ensure_document(
        self,
        pmcid: str,
        doc_id: str,
        existing: Optional[Dict[str, Any]],
    ) -> None:
        """Create a minimal document if it doesn't exist yet.

        Ensures the CouchDB document exists so that subsequent steps
        (XML attachment, text attachment) have something to attach to.
        """
        if existing is not None:
            return

        doc: Dict[str, Any] = {
            "_id": doc_id,
            "pmcid": pmcid,
            "source": "pmc",
        }
        set_timestamps(doc, is_new=True)
        self.db.save(doc)

        if self.verbosity >= 2:
            print(f"Created document for PMC{pmcid}")

    def _download_and_attach_xml(
        self, pmcid: str, doc_id: str,
    ) -> None:
        """Download and attach JATS XML for a PMC article.

        Fetches JATS XML via the OAI-PMH service and stores it as
        an ``article.xml`` CouchDB attachment. Also extracts metadata
        (title, DOI, authors, etc.) from the JATS front matter and
        saves it on the document.
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

        # Extract metadata from JATS front matter
        metadata = self._extract_metadata_from_xml(xml_string)

        doc = self.db[doc_id]
        doc["xml_available"] = True
        doc["xml_format"] = "jats"
        doc["is_jats"] = True
        doc["is_taxpub"] = False
        # Update metadata fields (don't overwrite existing non-empty values)
        for key in ("title", "doi", "pmid", "pmcid", "license", "authors"):
            value = metadata.get(key)
            if value and not doc.get(key):
                doc[key] = value
        self.db.save(doc)

        doc = self.db[doc_id]
        self.db.put_attachment(
            doc,
            BytesIO(xml_string.encode("utf-8")),
            "article.xml",
            "application/xml",
        )
        if self.verbosity >= 2:
            title = metadata.get("title", "")
            if title:
                print(f"  Attached JATS XML for PMC{pmcid} — "
                      f"{title[:60]}")
            else:
                print(f"  Attached JATS XML for PMC{pmcid}")

    def _download_and_attach_text(
        self, pmcid: str, doc_id: str,
    ) -> None:
        """Download and attach plaintext for a PMC article.

        Fetches plaintext via NCBI E-utilities efetch with
        ``retmode=text`` and stores it as an ``article.txt`` CouchDB
        attachment.
        """
        if doc_id not in self.db:
            if self.verbosity >= 1:
                print(f"  Cannot attach text for PMC{pmcid}: "
                      "document not found")
            return

        if self.verbosity >= 3:
            print(f"Fetching plaintext for PMC{pmcid}")

        from ingestors.extract_plaintext import plaintext_from_efetch

        try:
            text = plaintext_from_efetch(
                f"PMC{pmcid}",
                api_key=self.ncbi_api_key,
            )
        except ValueError as exc:
            if self.verbosity >= 1:
                print(f"  Plaintext unavailable for PMC{pmcid}: "
                      f"{exc}")
            doc = self.db[doc_id]
            doc["text_available"] = False
            self.db.save(doc)
            return

        doc = self.db[doc_id]
        doc["text_available"] = True
        self.db.save(doc)
        doc = self.db[doc_id]
        self.db.put_attachment(
            doc,
            BytesIO(text.encode("utf-8")),
            "article.txt",
            "text/plain",
        )
        if self.verbosity >= 2:
            print(f"  Attached plaintext for PMC{pmcid}")

    def _ingest_article(self, pmcid: str) -> None:
        """Ingest a single article by PMCID.

        Determines what work is needed (XML, text, or both) and
        processes accordingly. Skips if the document already has
        the requested data (unless force is set).
        """
        doc_id = self._make_doc_id(pmcid)

        # Check for existing document
        existing = (
            self.db[doc_id] if doc_id in self.db else None
        )

        # Determine what work is needed
        needs_xml = self._needs_xml_download(existing)
        needs_text = self._needs_text_download(existing)

        if not needs_xml and not needs_text:
            if self.verbosity >= 2:
                print(f"Skipping PMC{pmcid} "
                      "(already complete)")
            return

        if self.verbosity >= 2:
            parts = []
            if needs_xml:
                parts.append("XML")
            if needs_text:
                parts.append("text")
            print(f"Processing PMC{pmcid} "
                  f"({' + '.join(parts)})")

        if self.dry_run:
            if self.verbosity >= 2:
                print(f"Dry run: PMC{pmcid}")
            return

        # Ensure document exists before attaching
        self._ensure_document(pmcid, doc_id, existing)

        if needs_xml:
            self._download_and_attach_xml(pmcid, doc_id)

        if needs_text:
            self._download_and_attach_text(pmcid, doc_id)


# Backwards-compatible alias used in publications.py configs
PmcBiocIngestor = PmcIngestor
