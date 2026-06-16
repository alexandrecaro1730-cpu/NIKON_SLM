"""
Dataset validation checks.

This module creates a practical data quality report before any modeling step.

Why this matters:
- In manufacturing, bad input data can lead to wrong quality decisions.
- Validation helps detect missing values, schema changes, invalid values,
  duplicate records, and suspicious target labels.
- These checks are not only technical; they are part of production risk control.
"""

import pandas as pd

from lpbf_quality.config.settings import (
    EXPECTED_COLUMNS,
    QUALITY_ORDER,
    TARGET_COLUMN,
)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether the dataset contains the expected columns.

    Business reasoning:
    If a machine, database export, or operator-generated CSV changes its column
    names, the model could receive wrong or incomplete inputs. Detecting schema
    changes early avoids silent pipeline failures.
    """

    records = []

    actual_columns = set(df.columns)
    expected_columns = set(EXPECTED_COLUMNS)

    # Expected columns that are missing are serious because the model or reports
    # may depend on them.
    for col in sorted(expected_columns - actual_columns):
        records.append(
            {
                "check": "missing_expected_column",
                "column": col,
                "status": "FAIL",
                "details": "Expected column is missing from dataset.",
            }
        )

    # Unexpected columns are not always a failure. They may indicate new sensors,
    # new exports, or harmless metadata. We flag them for review.
    for col in sorted(actual_columns - expected_columns):
        records.append(
            {
                "check": "unexpected_column",
                "column": col,
                "status": "WARN",
                "details": "Column is not part of the expected schema.",
            }
        )

    if not records:
        records.append(
            {
                "check": "schema",
                "column": None,
                "status": "PASS",
                "details": "All expected columns are present.",
            }
        )

    return pd.DataFrame(records)


def validate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report missing values by column.

    Business reasoning:
    Missing values may come from sensor failures, incomplete machine logs,
    manual entry problems, or unavailable post-build tests. The model can impute
    some missing values, but the business still needs visibility into them.
    """

    rows = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_percent = float(df[col].isna().mean() * 100)

        rows.append(
            {
                "column": col,
                "missing_count": missing_count,
                "missing_percent": round(missing_percent, 2),
                "status": "WARN" if missing_count > 0 else "PASS",
            }
        )

    return pd.DataFrame(rows)


def validate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for duplicate rows.

    Business reasoning:
    Duplicate samples can distort model training and evaluation. For example,
    if the same build appears in both training and test data, performance may
    look better than it really is.
    """

    duplicate_count = int(df.duplicated().sum())

    return pd.DataFrame(
        [
            {
                "check": "duplicate_rows",
                "duplicate_count": duplicate_count,
                "status": "WARN" if duplicate_count > 0 else "PASS",
            }
        ]
    )


def validate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the target column used for prediction.

    Business reasoning:
    The model predicts Quality_Class. If labels are missing, misspelled, or
    inconsistent, the model learns the wrong business definition of quality.
    """

    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame(
            [
                {
                    "check": "target_column",
                    "status": "FAIL",
                    "details": f"{TARGET_COLUMN} not found.",
                }
            ]
        )

    observed = set(df[TARGET_COLUMN].dropna().unique())
    expected = set(QUALITY_ORDER)

    rows = []

    # Labels outside the approved list are likely data entry or export issues.
    for label in sorted(observed - expected):
        rows.append(
            {
                "check": "unexpected_target_label",
                "value": label,
                "status": "FAIL",
            }
        )

    # Missing classes are not necessarily wrong, but they affect training and
    # evaluation. For example, a dataset with no "Poor" builds cannot teach the
    # model how to detect poor quality.
    for label in sorted(expected - observed):
        rows.append(
            {
                "check": "missing_target_label",
                "value": label,
                "status": "WARN",
            }
        )

    class_counts = df[TARGET_COLUMN].value_counts(dropna=False)

    for label, count in class_counts.items():
        rows.append(
            {
                "check": "target_distribution",
                "value": label,
                "count": int(count),
                "percent": round(float(count / len(df) * 100), 2),
                "status": "INFO",
            }
        )

    return pd.DataFrame(rows)


def validate_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether numeric values are physically reasonable.

    Business reasoning:
    Industrial datasets often contain impossible values due to sensor errors,
    unit conversion issues, or manual input mistakes. These checks do not prove
    the data is perfect, but they catch obvious problems before training.
    """

    checks = {
        "Powder_Size_um": (0, None),
        "Oxygen_Content_percent": (0, 100),
        "Laser_Power_W": (0, None),
        "Scan_Speed_mm_s": (0, None),
        "Hatch_Spacing_mm": (0, None),
        "Layer_Thickness_um": (0, None),
        "Preheat_Temp_C": (0, None),
        "Shielding_Gas_Flow_L_min": (0, None),
        "Melt_Pool_Width_um": (0, None),
        "Melt_Pool_Depth_um": (0, None),
        "Melt_Pool_Temp_C": (0, None),
        "Cooling_Rate_K_s": (0, None),
        "Energy_Density_J_mm3": (0, None),
        "Thermal_Gradient": (0, None),
        "Relative_Density_percent": (0, 100),
        "Porosity_percent": (0, 100),
        "Microhardness_HV": (0, None),
        "Surface_Roughness_Ra": (0, None),
        "Tensile_Strength_MPa": (0, None),
        "Yield_Strength_MPa": (0, None),
        "Elongation_percent": (0, 100),
        "Defect_Count": (0, None),
    }

    rows = []

    for col, (min_value, max_value) in checks.items():
        if col not in df.columns:
            continue

        invalid = pd.Series(False, index=df.index)

        if min_value is not None:
            invalid |= df[col] < min_value

        if max_value is not None:
            invalid |= df[col] > max_value

        rows.append(
            {
                "column": col,
                "invalid_count": int(invalid.sum()),
                "status": "WARN" if invalid.any() else "PASS",
                "rule": f"{min_value} <= {col} <= {max_value}",
            }
        )

    return pd.DataFrame(rows)


def build_data_quality_report(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run all validation checks and return them as separate report tables.

    Business reasoning:
    Returning separate tables makes the output easy to save as CSV files and
    include in the final PowerPoint. Each table answers a specific question:
    schema, missing values, duplicates, target labels, and physical validity.
    """

    return {
        "schema": validate_schema(df),
        "missing_values": validate_missing_values(df),
        "duplicates": validate_duplicates(df),
        "target": validate_target(df),
        "physical_ranges": validate_physical_ranges(df),
    }