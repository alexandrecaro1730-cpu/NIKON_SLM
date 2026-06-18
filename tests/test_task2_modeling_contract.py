"""
Tests for the Task 2 LPBF modeling contract.

Business objective
------------------
Verify that the modeling system is leakage-safe and deployable enough for a
proof-of-concept industrial ML workflow.

The most important rule is:

    Use only what is known at the decision time.

Coding objective
----------------
Test:
- pre-build feature view excludes post-build leakage features
- in-build feature view excludes post-build leakage features
- post-build diagnostic view allows post-build features but excludes IDs/target
- feature building returns valid X/y
- model training pipeline runs on a small synthetic dataset
"""

import pandas as pd

from lpbf_quality.config.settings import POST_BUILD_FEATURES
from lpbf_quality.features.build_features import build_model_frame
from lpbf_quality.features.feature_sets import get_feature_set
from lpbf_quality.models.training import train_model


def make_synthetic_lpbf_data() -> pd.DataFrame:
    """
    Build a small synthetic dataset for pipeline tests.

    Business objective
    ------------------
    Test pipeline behavior without depending on the real CSV file.

    Coding objective
    ----------------
    Create all expected LPBF columns with enough rows per quality class for
    stratified splitting.
    """

    rows = []
    classes = ["Poor", "Moderate", "Good", "Excellent"]

    for i in range(40):
        quality = classes[i % len(classes)]

        rows.append(
            {
                "Sample_ID": f"S{i:03d}",
                "Alloy_Type": "Ti-6Al-4V",
                "Powder_Morphology": "Spherical",
                "Scan_Strategy": "Stripe",
                "Powder_Size_um": 30 + i % 5,
                "Oxygen_Content_percent": 0.1 + (i % 3) * 0.01,
                "Laser_Power_W": 250 + i % 20,
                "Scan_Speed_mm_s": 900 + i % 30,
                "Hatch_Spacing_mm": 0.1,
                "Layer_Thickness_um": 30,
                "Preheat_Temp_C": 150,
                "Shielding_Gas_Flow_L_min": 20,
                "Melt_Pool_Width_um": 100 + i % 10,
                "Melt_Pool_Depth_um": 50 + i % 5,
                "Melt_Pool_Temp_C": 1600 + i % 20,
                "Cooling_Rate_K_s": 100000 + i,
                "Energy_Density_J_mm3": 55 + i % 4,
                "Thermal_Gradient": 1000 + i,
                "Relative_Density_percent": 99.0,
                "Porosity_percent": 0.2,
                "Microhardness_HV": 350,
                "Surface_Roughness_Ra": 8,
                "Tensile_Strength_MPa": 1000,
                "Yield_Strength_MPa": 900,
                "Elongation_percent": 12,
                "Defect_Count": i % 3,
                "Quality_Class": quality,
            }
        )

    return pd.DataFrame(rows)


def test_prebuild_feature_set_excludes_postbuild_features():
    """
    Business objective
    ------------------
    Confirm that the pre-build model cannot use future inspection/test results.
    """

    features = set(get_feature_set("prebuild"))

    assert not features.intersection(POST_BUILD_FEATURES)
    assert "Sample_ID" not in features
    assert "Quality_Class" not in features


def test_inbuild_feature_set_excludes_postbuild_features():
    """
    Business objective
    ------------------
    Confirm that the in-build MVP model uses monitoring features but still does
    not use post-build inspection/test results.
    """

    features = set(get_feature_set("inbuild"))

    assert not features.intersection(POST_BUILD_FEATURES)
    assert "Sample_ID" not in features
    assert "Quality_Class" not in features


def test_postbuild_feature_set_allows_diagnostic_features_but_excludes_ids():
    """
    Business objective
    ------------------
    Confirm that the post-build view can support diagnosis while still excluding
    identifiers and the target itself.
    """

    features = set(get_feature_set("postbuild"))

    assert features.intersection(POST_BUILD_FEATURES)
    assert "Sample_ID" not in features
    assert "Quality_Class" not in features


def test_build_model_frame_returns_scenario_features_and_target():
    """
    Business objective
    ------------------
    Confirm that the modeling frame separates input features from the target.
    """

    df = make_synthetic_lpbf_data()

    X, y = build_model_frame(df, scenario="inbuild")

    assert len(X) == len(df)
    assert len(y) == len(df)
    assert "Quality_Class" not in X.columns
    assert "Sample_ID" not in X.columns


def test_training_pipeline_runs_on_synthetic_data():
    """
    Business objective
    ------------------
    Confirm that the Task 2 pipeline is deployable enough to fit, predict, and
    produce metrics.
    """

    df = make_synthetic_lpbf_data()

    result = train_model(
        df=df,
        scenario="inbuild",
        model_type="random_forest",
        test_size=0.25,
        random_state=42,
    )

    assert "pipeline" in result
    assert "metrics" in result
    assert "balanced_accuracy" in result["metrics"]
    assert "macro_f1" in result["metrics"]
    assert result["metrics"]["metadata"]["scenario"] == "inbuild"
    assert result["metrics"]["metadata"]["scenario_description"]["operational_use"] == (
        "Production MVP risk update"
    )