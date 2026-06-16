"""
Business objective
------------------
Verify that the independent challenge layer runs and produces audit artifacts.

Coding objective
----------------
Test that challenge_analysis returns the expected report dictionary and that
target leakage helper columns are not used as model features.
"""

import pandas as pd

from lpbf_quality.data.challenge_analysis import (
    run_challenge_analysis,
    _safe_feature_columns,
)


def test_challenge_analysis_outputs_expected_reports(
    tmp_path,
    sample_lpbf_dataframe: pd.DataFrame,
) -> None:
    reports = run_challenge_analysis(
        df=sample_lpbf_dataframe,
        output_dir=tmp_path,
    )

    expected = {
        "distribution_challenge",
        "physics_correlation_challenge",
        "energy_density_formula_challenge",
        "quality_separation_challenge",
        "defect_trend_challenge",
        "quick_model_predictability_challenge",
        "leakage_challenge",
    }

    assert expected.issubset(reports.keys())

    for report_name in expected:
        assert (tmp_path / f"{report_name}.csv").exists()


def test_safe_feature_columns_exclude_target_derived_columns(
    sample_lpbf_dataframe: pd.DataFrame,
) -> None:
    df = sample_lpbf_dataframe.copy()
    df["Quality_Class_numeric"] = [0, 1, 2, 3, 0, 2]
    df["target_encoded"] = [0.1, 0.2, 0.3, 0.4, 0.1, 0.3]

    features = _safe_feature_columns(df)

    assert "Quality_Class" not in features
    assert "Quality_Class_numeric" not in features
    assert "target_encoded" not in features
    assert "Sample_ID" not in features