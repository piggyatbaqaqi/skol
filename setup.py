#!/usr/bin/env python3
"""
Setup script for skol (required by stdeb for Debian packaging).
"""

from setuptools import setup

setup(
    name="skol",
    version="0.9.0",
    description="Taxonomic text classification and extraction pipeline for mycological literature",
    author="Christopher Murphy, La Monte Henry Piggy Yarroll, David Caspers",
    license="GPL-3.0-or-later",
    # packages and py-modules live in pyproject.toml [tool.setuptools],
    # which the build uses; a second list here drifted (it lacked
    # span_resolver, and the wheel shipped without it).
    include_package_data=True,
    python_requires=">=3.14",
    # install_requires lives in pyproject.toml [project].dependencies — the
    # canonical declaration under PEP 517 build backends.  Keeping a second
    # list here just drifts out of sync (it did, repeatedly).
    extras_require={
        # RAPIDS GPU acceleration
        # Install with: pip install --extra-index-url=https://pypi.nvidia.com skol[gpu]
        "gpu": [
            "cudf-cu12",
            "dask-cudf-cu12",
            "cuml-cu12",
            "cugraph-cu12",
        ],
    },
    entry_points={
        "console_scripts": [
            "skol-ingest=bin.ingest:main",
            "skol-train=bin.train_classifier:main",
            "skol-predict=bin.predict_classifier:main",
            "skol-extract-treatments=bin.extract_treatments_to_couchdb:main",
            "skol-embed-treatments=bin.embed_treatments:main",
        ],
    },
    data_files=[
        ("share/skol", ["debian/skol.service"]),
        ("etc/logrotate.d", ["etc/logrotate.d/skol"]),
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
