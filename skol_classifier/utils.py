"""
Utility functions for SKOL text classification
"""

import glob
import warnings
from typing import List
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def get_file_list(
    folder: str,
    pattern: str = "**/*.txt*",
    exclude_pattern: str = "Sydowia"
) -> List[str]:
    """
    List all files matching the pattern in a folder.

    Args:
        folder: Directory path to search
        pattern: Glob pattern for file matching (default: "**/*.txt*")
        exclude_pattern: Pattern to exclude from results (default: "Sydowia")

    Returns:
        List of file paths matching the criteria

    Raises:
        FileNotFoundError: If the folder doesn't exist
        PermissionError: If access to the folder is denied
    """
    try:
        files = [
            file for file in glob.glob(f'{folder}/{pattern}', recursive=True)
            if exclude_pattern not in file
        ]
        return files
    except FileNotFoundError:
        print(f"Folder '{folder}' not found.")
        raise
    except PermissionError:
        print(f"Permission denied to access folder '{folder}'.")
        raise
