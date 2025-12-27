"""
Example usage of the Ingestor classes.

This demonstrates how to use the IngentaIngestor to ingest data from
RSS feeds and local BibTeX files.
"""

from pathlib import Path
from urllib.robotparser import RobotFileParser

import couchdb

from ingestors.ingenta import IngentaIngestor


def main() -> None:
    """Example usage of IngentaIngestor."""
    # Set up CouchDB connection
    couch = couchdb.Server('http://localhost:5984')
    db = couch['skol_dev']  # or your database name

    # Set up robot parser
    user_agent = "synoptickeyof.life"
    ingenta_rp = RobotFileParser()
    ingenta_rp.set_url("https://www.ingentaconnect.com/robots.txt")
    ingenta_rp.read()

    # Create IngentaIngestor instance
    ingestor = IngentaIngestor(
        db=db,
        user_agent=user_agent,
        robot_parser=ingenta_rp
    )

    # Example 1: Ingest from RSS feed
    # Mycotaxon
    ingestor.ingest_from_rss(
        rss_url='https://api.ingentaconnect.com/content/mtax/mt?format=rss'
    )

    # Example 2: Ingest from another RSS feed
    # Studies in Mycology
    ingestor.ingest_from_rss(
        rss_url='https://api.ingentaconnect.com/content/wfbi/sim?format=rss'
    )

    # Example 3: Ingest from local BibTeX files
    ingestor.ingest_from_local_bibtex(
        root=Path("/data/skol/www/www.ingentaconnect.com")
    )


if __name__ == '__main__':
    main()
