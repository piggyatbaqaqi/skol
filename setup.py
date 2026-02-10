#!/usr/bin/env python3
"""
Setup script for skol (required by stdeb for Debian packaging).
"""

from setuptools import setup, find_packages

setup(
    name="skol",
    version="0.1.0",
    description="Taxonomic text classification and extraction pipeline for mycological literature",
    author="Christopher Murphy, La Monte Henry Piggy Yarroll, David Caspers",
    license="GPL-3.0-or-later",
    packages=find_packages(include=[
        "skol", "skol.*",
        "skol_classifier", "skol_classifier.*",
        "ingestors", "ingestors.*",
        "training", "training.*",
        "yedda_parser", "yedda_parser.*",
        "bin",
    ]),
    py_modules=[
        "constants",
        "couchdb_file",
        "file",
        "fileobj",
        "finder",
        "indexfungorum_authors",
        "iso4",
        "label",
        "line",
        "mycobank_authors",
        "mycobank_species",
        "nomenclature",
        "paragraph",
        "pdf_section_extractor",
        "skol_compat",
        "span",
        "taxa_json_translator",
        "taxon",
        "tokenizer",
    ],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "CouchDB>=1.2",
        "numpy",
        "pandas",
        "pyspark>=3.5.0,<4.0",
        "redis>=4.0.0",
        "regex>=2024.0.0",
        "requests>=2.32.0",
        "scikit-learn>=1.0.0",
        "tqdm",
        "python-dateutil>=2.9.0",
    ],
    entry_points={
        "console_scripts": [
            "skol-ingest=bin.ingest:main",
            "skol-train=bin.train_classifier:main",
            "skol-predict=bin.predict_classifier:main",
            "skol-extract-taxa=bin.extract_taxa_to_couchdb:main",
            "skol-embed-taxa=bin.embed_taxa:main",
        ],
    },
    data_files=[
        ("share/skol", ["debian/skol.service"]),
        ("share/skol/ontologies", [
            "data/ontologies/pato.obo",
            "data/ontologies/fao.obo",
        ]),
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
