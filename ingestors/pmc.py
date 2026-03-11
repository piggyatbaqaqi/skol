"""
Ingestor for PubMed Central articles via NCBI E-utilities and BioC JSON API.

Discovers articles using E-utilities esearch, retrieves full text as BioC JSON
from the PMC Open Access subset, and enriches existing documents (e.g., from
Crossref) with PMC full text when a DOI match is found.
"""

import json
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
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BIOC_URL = (
        "https://www.ncbi.nlm.nih.gov"
        "/research/bionlp/RESTful/pmcoa.cgi/BioC_json"
    )
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

    def _ingest_article(self, pmcid: str) -> None:
        """Ingest a single article by PMCID.

        Skips if the document already has bioc_json (unless force
        is set). If a DOI-matched document exists from another
        source, enriches it instead of creating a duplicate.
        Tracks bioc_json_available to avoid retrying known-unavailable
        resources (overridden by recheck_bioc_json).
        """
        doc_id = self._make_doc_id(pmcid)
        pmc_url = self.PMC_ARTICLE_URL_TEMPLATE.format(pmcid)

        # Check for existing document by PMC doc ID
        existing_by_id = None
        if doc_id in self.db:
            existing_by_id = self.db[doc_id]
            if not self.force and "bioc_json" in existing_by_id:
                if self.verbosity >= 2:
                    print(f"Skipping PMC{pmcid} (already has bioc_json)")
                return
            # Skip if previously marked unavailable (unless rechecking)
            if (not self.recheck_bioc_json
                    and existing_by_id.get("bioc_json_available") is False):
                if self.verbosity >= 2:
                    print(f"Skipping PMC{pmcid} (bioc_json unavailable)")
                return

        if self.dry_run:
            if self.verbosity >= 2:
                print(f"Dry run: PMC{pmcid}")
            return

        if not self.download_bioc_json:
            return

        if self.verbosity >= 3:
            print(f"Fetching BioC JSON for PMC{pmcid}")

        # Fetch BioC JSON
        bioc = self._fetch_bioc_json(pmcid)
        if bioc is None:
            if self.verbosity >= 1:
                print(f"  BioC JSON unavailable for PMC{pmcid}")
            # Record that BioC JSON is not available
            if existing_by_id is not None:
                existing_by_id["bioc_json_available"] = False
                self.db.save(existing_by_id)
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
        elif existing_by_id is not None:
            # Update existing PMC document (force reprocess)
            existing_by_id["bioc_json"] = bioc
            existing_by_id["bioc_json_available"] = True
            set_timestamps(existing_by_id)
            self.db.save(existing_by_id)
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
