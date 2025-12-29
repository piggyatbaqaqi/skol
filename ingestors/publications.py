"""
Publication registry for managing ingestion source configurations.

This module provides a centralized registry of publication sources and their
configurations for the ingestion system.
"""

from typing import Dict, Optional, Any


class PublicationRegistry:
    """
    Registry for publication source configurations.

    This class manages all predefined publication sources, their configurations,
    and provides methods to access publication information.
    """

    # Default configuration values for all publications
    DEFAULTS: Dict[str, Any] = {
        'rate_limit_min_ms': 1000,  # Minimum delay between requests (milliseconds)
        'rate_limit_max_ms': 5000,  # Maximum delay between requests (milliseconds)
    }

    # Robots.txt URLs for each source
    ROBOTS_URLS: Dict[str, str] = {
        'ingenta': 'https://www.ingentaconnect.com/robots.txt',
        'mykoweb-journals': 'https://mykoweb.com/robots.txt',
        'mykoweb-literature': 'https://mykoweb.com/robots.txt',
        'mykoweb-caf': 'https://mykoweb.com/robots.txt',
        'mykoweb-crepidotus': 'https://mykoweb.com/robots.txt',
        'mykoweb-oldbooks': 'https://mykoweb.com/robots.txt',
        'mykoweb-gsmnp': 'https://mykoweb.com/robots.txt',
        'mykoweb-pholiota': 'https://mykoweb.com/robots.txt',
        'mykoweb-misc': 'https://mykoweb.com/robots.txt',
        'mycosphere': 'https://mycosphere.org/robots.txt',
        'taylor-francis-mycology': 'https://www.tandfonline.com/robots.txt',
    }

    # Publication source configurations
    SOURCES: Dict[str, Dict[str, Any]] = {
        'mycotaxon-rss': {
            'name': 'Mycotaxon',
            'source': 'ingenta',
            'ingestor_class': 'IngentaIngestor',
            'mode': 'rss',
            'rss_url': 'https://api.ingentaconnect.com/content/mtax/mt?format=rss',
        },
        'mycotaxon': {
            'name': 'Mycotaxon',
            'source': 'ingenta',
            'ingestor_class': 'IngentaIngestor',
            'mode': 'index',
            'index_url': 'https://api.ingentaconnect.com/content/mtax/mt?format=index',
        },
        'studies-in-mycology': {
            'name': 'Studies in Mycology',
            'source': 'ingenta',
            'ingestor_class': 'IngentaIngestor',
            'mode': 'index',
            'index_url': 'https://api.ingentaconnect.com/content/wfbi/sim?format=index',
        },
        'studies-in-mycology-rss': {
            'name': 'Studies in Mycology',
            'source': 'ingenta',
            'ingestor_class': 'IngentaIngestor',
            'mode': 'rss',
            'rss_url': 'https://api.ingentaconnect.com/content/wfbi/sim?format=rss',
        },
        'persoonia': {
            'name': 'Persoonia',
            'source': 'ingenta',
            'ingestor_class': 'IngentaIngestor',
            'mode': 'index',
            'index_url': 'https://api.ingentaconnect.com/content/wfbi/pimj?format=index',
        },
        'persoonia-rss': {
            'name': 'Persoonia',
            'source': 'ingenta',
            'ingestor_class': 'IngentaIngestor',
            'mode': 'rss',
            'rss_url': 'https://api.ingentaconnect.com/content/wfbi/pimj?format=rss',
        },
        'ingenta-local': {
            'name': 'Ingenta Local BibTeX Files',
            'source': 'ingenta',
            'ingestor_class': 'LocalIngentaIngestor',
            'mode': 'local',
            'local_path': '/data/skol/www/www.ingentaconnect.com',
        },
        'mykoweb-journals': {
            'name': 'Mykoweb Journals (Mycotaxon, Persoonia, Sydowia)',
            'source': 'mykoweb-journals',
            'ingestor_class': 'LocalMykowebJournalsIngestor',
            'mode': 'local',
            'local_path': '/data/skol/www/mykoweb.com/systematics/journals',
            'local_path_prefix': '/data/skol/www/mykoweb.com/systematics/journals',
            'url_prefix': 'https://mykoweb.com/systematics/journals',
        },
        'mykoweb-literature': {
            'name': 'Mykoweb Literature/Books',
            'source': 'mykoweb-literature',
            'ingestor_class': 'LocalMykowebLiteratureIngestor',
            'mode': 'local',
            'local_path': '/data/skol/www/mykoweb.com/systematics/literature',
            'local_path_prefix': '/data/skol/www/mykoweb.com/systematics/literature',
            'url_prefix': 'https://mykoweb.com/systematics/literature',
        },
        'mykoweb-caf': {
            'name': 'Mykoweb CAF PDFs',
            'source': 'mykoweb-caf',
            'ingestor_class': 'LocalMykowebLiteratureIngestor',
            'mode': 'local',
            'local_path': '/data/skol/www/mykoweb.com/CAF',
            'local_path_prefix': '/data/skol/www/mykoweb.com/CAF',
            'url_prefix': 'https://mykoweb.com/CAF',
        },
        'mykoweb-crepidotus': {
            'name': 'Mykoweb Crepidotus',
            'source': 'mykoweb-crepidotus',
            'ingestor_class': 'LocalMykowebLiteratureIngestor',
            'mode': 'local',
            'local_path': '/data/skol/www/mykoweb.com/Crepidotus',
            'local_path_prefix': '/data/skol/www/mykoweb.com/Crepidotus',
            'url_prefix': 'https://mykoweb.com/Crepidotus',
        },
        'mykoweb-oldbooks': {
            'name': 'Mykoweb Old Books',
            'source': 'mykoweb-oldbooks',
            'ingestor_class': 'LocalMykowebLiteratureIngestor',
            'mode': 'local',
            'local_path': '/data/skol/www/mykoweb.com/OldBooks',
            'local_path_prefix': '/data/skol/www/mykoweb.com/OldBooks',
            'url_prefix': 'https://mykoweb.com/OldBooks',
        },
        'mykoweb-gsmnp': {
            'name': 'Mykoweb GSMNP',
            'source': 'mykoweb-gsmnp',
            'ingestor_class': 'LocalMykowebLiteratureIngestor',
            'mode': 'local',
            'local_path': '/data/skol/www/mykoweb.com/GSMNP',
            'local_path_prefix': '/data/skol/www/mykoweb.com/GSMNP',
            'url_prefix': 'https://mykoweb.com/GSMNP',
        },
        'mykoweb-pholiota': {
            'name': 'Mykoweb Pholiota',
            'source': 'mykoweb-pholiota',
            'ingestor_class': 'LocalMykowebLiteratureIngestor',
            'mode': 'local',
            'local_path': '/data/skol/www/mykoweb.com/Pholiota',
            'local_path_prefix': '/data/skol/www/mykoweb.com/Pholiota',
            'url_prefix': 'https://mykoweb.com/Pholiota',
        },
        'mykoweb-misc': {
            'name': 'Mykoweb Misc',
            'source': 'mykoweb-misc',
            'ingestor_class': 'LocalMykowebLiteratureIngestor',
            'mode': 'local',
            'local_path': '/data/skol/www/mykoweb.com/misc',
            'local_path_prefix': '/data/skol/www/mykoweb.com/misc',
            'url_prefix': 'https://mykoweb.com/misc',
        },
        'mycosphere': {
            'name': 'Mycosphere',
            'source': 'mycosphere',
            'ingestor_class': 'MycosphereIngestor',
            'mode': 'web',
            'archives_url': 'https://mycosphere.org/archives.php',
            'rate_limit_min_ms': 1000,
            'rate_limit_max_ms': 5000,
        },
        'taylor-francis-mycology': {
            'name': 'Mycology (Taylor & Francis)',
            'source': 'taylor-francis-mycology',
            'ingestor_class': 'TaylorFrancisIngestor',
            'mode': 'web',
            'archives_url': 'https://www.tandfonline.com/loi/tmyc20',
            'journal_name': 'Mycology',
            'issn': '2150-1203',
            'eissn': '2150-1211',
            'rate_limit_min_ms': 1000,
            'rate_limit_max_ms': 5000,
        },
    }

    @classmethod
    def get_all(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all publication source configurations.

        Each configuration includes default values merged with source-specific settings.

        Returns:
            Dictionary mapping publication keys to their configurations
        """
        result = {}
        for key, config in cls.SOURCES.items():
            cfg = cls.DEFAULTS.copy()
            cfg.update(config)
            result[key] = cfg
        return result

    @classmethod
    def get(cls, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific publication configuration by key.

        Configuration includes default values merged with source-specific settings.

        Args:
            key: Publication key (e.g., 'mycotaxon', 'mycosphere')

        Returns:
            Publication configuration dict, or None if not found
        """
        config = cls.SOURCES.get(key)
        if config is None:
            return None
        cfg = cls.DEFAULTS.copy()
        cfg.update(config)
        return cfg

    @classmethod
    def get_robots_url(cls, source: str, custom_url: Optional[str] = None) -> str:
        """
        Get the robots.txt URL for a given source.

        Args:
            source: Name of the data source (e.g., 'ingenta', 'mycosphere')
            custom_url: Custom robots.txt URL (overrides default)

        Returns:
            URL to robots.txt file, or empty string if not found
        """
        if custom_url:
            return custom_url

        return cls.ROBOTS_URLS.get(source, '')

    @classmethod
    def list_publications(cls) -> list[str]:
        """
        Get a list of all available publication keys.

        Returns:
            Sorted list of publication keys
        """
        return sorted(cls.SOURCES.keys())

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """
        Get default configuration values.

        Returns:
            Dictionary of default configuration values
        """
        return cls.DEFAULTS.copy()
