"""
Data loading utilities.

This module keeps data loading separate from analysis, feature engineering,
and modeling.

Why this matters:
- A production pipeline should have one clear entry point for reading data.
- Centralized loading makes it easier to add future checks, logging, or support
  for databases/cloud storage.
"""

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load the LPBF dataset from a CSV file.

    Parameters
    ----------
    path:
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.

    ValueError
        If the CSV exists but contains no rows.

    Business reasoning
    ------------------
    We fail early if the input file is missing or empty. This prevents the model
    pipeline from continuing with invalid data and producing misleading results.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    return df