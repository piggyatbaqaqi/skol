"""
Setup script for SKOL Text Classifier
"""

from setuptools import setup, find_packages

with open("skol_classifier/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="skol-classifier",
    version="0.0.1",
    author="Christopher Murphy, La Monte Henry Piggy Yarroll, David Caspers",
    description="PySpark-based text classification pipeline for taxonomic literature",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pyspark>=3.0.0",
        "spark-nlp",
        "regex>=2021.0.0",
        "redis>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    include_package_data=True,
    keywords="text classification, pyspark, nlp, taxonomy, machine learning",
)
