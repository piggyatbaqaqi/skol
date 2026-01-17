#!/usr/bin/env python3
"""
Setup script for skol-django (required by stdeb for Debian packaging).
"""

from setuptools import setup, find_packages

setup(
    name="skol-django",
    version="0.1.0",
    description="Django web application for SKOL taxonomic search and user management",
    author="Christopher Murphy, La Monte Henry Piggy Yarroll, David Caspers",
    license="GPL-3.0-or-later",
    packages=find_packages(include=["skolweb", "skolweb.*", "search", "search.*", "accounts", "accounts.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "Django>=4.2,<5.0",
        "djangorestframework>=3.14.0",
        "django-cors-headers>=4.0.0",
        "redis>=4.5.0",
    ],
    entry_points={
        "console_scripts": [
            "skol-django=skolweb.manage:main",
        ],
    },
    data_files=[
        ("share/skol-django", ["debian/skol-django.service"]),
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Django",
        "Framework :: Django :: 4.2",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
