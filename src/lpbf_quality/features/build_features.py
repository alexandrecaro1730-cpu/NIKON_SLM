"""
Feature preparation for LPBF quality prediction.

Business objective
------------------
Prepare model-ready data while enforcing the central production rule:

    Use only what is known at the decision time.

This supports a scenario-based quality intelligence system:
- pre-build risk screening
- in-build risk update as the production MVP
- post-build diagnostic benchmark

Coding objective
----------------
Convert a raw validated dataset into X/y matrices for one selected scenario.
"""

import pandas as pd

from lpbf_quality.config.settings import TARGET_COLUMN
from lpbf_quality.features.feature_sets import (
    get_feature_set,
    validate_no_timing_leakage,
)


def build_model_frame(
    df: pd.DataFrame,
    scenario: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build features X and target y for a selected modeling scenario.

    Business objective
    ------------------
    Ensure the model only receives features that would be available at the time
    the business decision is made.

    Example:
    - A pre-build model must not use porosity or tensile strength because those
      values are only known after inspection/testing.
    - An in-build model can use monitoring features such as melt pool signals.
    - A post-build model may use inspection/test results, but only for diagnosis.

    Coding objective
    ----------------
    1. Validate the scenario feature set.
    2. Check that required columns exist.
    3. Return X and y for scikit-learn training/evaluation.
    """

    validate_no_timing_leakage(scenario)

    selected_features = get_feature_set(scenario)
    required_columns = selected_features + [TARGET_COLUMN]

    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Input data is missing required columns for scenario "
            f"'{scenario}': {missing_columns}"
        )

    X = df[selected_features].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y