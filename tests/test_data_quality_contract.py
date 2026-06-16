"""
Business objective
------------------
Verify that the project detects basic data-quality risks before modeling.

Coding objective
----------------
Test the data-quality report functions on a small controlled dataset.
"""

import pandas as pd

from lpbf_quality.data.validate_data import build_data_quality_report


def test_data_quality_reports_are_created(sample_lpbf_dataframe: pd.DataFrame) -> None:
    reports = build_data_quality_report(sample_lpbf_dataframe)

    assert isinstance(reports, dict)
    assert "missing_values" in reports
    assert "duplicates" in reports
    assert "physical_ranges" in reports


def test_missing_values_report_has_expected_columns(sample_lpbf_dataframe: pd.DataFrame) -> None:
    reports = build_data_quality_report(sample_lpbf_dataframe)
    missing = reports["missing_values"]

    assert "column" in missing.columns
    assert "missing_count" in missing.columns
    assert "missing_percent" in missing.columns
    assert "status" in missing.columns


def test_duplicate_rows_are_detectable(sample_lpbf_dataframe: pd.DataFrame) -> None:
    duplicated = pd.concat(
        [sample_lpbf_dataframe, sample_lpbf_dataframe.iloc[[0]]],
        ignore_index=True,
    )

    reports = build_data_quality_report(duplicated)
    duplicates = reports["duplicates"]

    assert "duplicate_count" in duplicates.columns
    assert duplicates["duplicate_count"].sum() >= 1