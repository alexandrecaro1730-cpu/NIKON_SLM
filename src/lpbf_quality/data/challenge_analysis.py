"""
Business objective
------------------
Challenge the initial EDA conclusions before modeling.

This module is intentionally skeptical. Its role is to check whether the dataset
really supports the conclusions we are making, especially for a physics/R&D
manufacturing project.

Coding objective
----------------
Run independent diagnostic checks for:
- suspiciously uniform feature distributions
- near-zero correlations between expected LPBF physics variables
- weak separation between quality classes
- possible synthetic or pre-cleaned data behavior
- possible leakage or identifier columns
- energy density physical consistency
- density / porosity consistency
- defect count relationship with quality
- quick baseline predictability sanity check
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, kruskal

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lpbf_quality.config.settings import (
    CATEGORICAL_FEATURES,
    POST_BUILD_FEATURES,
    TARGET_COLUMN,
)


IDENTIFIER_COLUMNS = {"Sample_ID", "ID", "index"}
LEAKAGE_NAME_PATTERNS = [
    "quality_class_numeric",
    "quality_numeric",
    "target",
    "label",
]

def _safe_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Business objective
    ------------------
    Ensure identifiers are not treated as physical process variables.

    Coding objective
    ----------------
    Return numeric columns excluding target and obvious ID columns.
    """

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    return [
        col
        for col in numeric_cols
        if col not in IDENTIFIER_COLUMNS and col != TARGET_COLUMN
    ]


def _uniformity_score(series: pd.Series, bins: int = 20) -> float:
    """
    Business objective
    ------------------
    Detect whether a feature looks artificially uniform, which can happen in
    synthetic or bounded generated datasets.

    Coding objective
    ----------------
    Compare histogram entropy to the maximum possible entropy.
    A score close to 1 means the distribution is very uniform.
    """

    clean = series.dropna()

    if clean.nunique() <= 1:
        return 0.0

    counts, _ = np.histogram(clean, bins=bins)
    counts = counts[counts > 0]

    if len(counts) <= 1:
        return 0.0

    return float(entropy(counts) / np.log(len(counts)))


def challenge_feature_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business objective
    ------------------
    Check whether numeric variables look like real noisy production data or more
    like bounded/synthetic data.

    Coding objective
    ----------------
    Compute distribution uniformity and basic range statistics.
    """

    rows = []

    for col in _safe_numeric_columns(df):
        series = df[col].dropna()

        if series.empty:
            continue

        uniformity = _uniformity_score(series)

        rows.append(
            {
                "feature": col,
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(),
                "unique_values": series.nunique(),
                "uniformity_score": uniformity,
                "interpretation": (
                    "Highly uniform / bounded-looking"
                    if uniformity >= 0.95
                    else "Not strongly uniform"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "uniformity_score",
        ascending=False,
    )


def challenge_physics_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business objective
    ------------------
    Test whether expected LPBF physics relationships are visible in the data.

    Coding objective
    ----------------
    Calculate correlations for selected physics-based feature pairs.
    """

    expected_pairs = [
        (
            "Laser_Power_W",
            "Energy_Density_J_mm3",
            "Expected positive if energy density is derived from power",
        ),
        (
            "Scan_Speed_mm_s",
            "Energy_Density_J_mm3",
            "Expected negative if energy density uses speed denominator",
        ),
        (
            "Energy_Density_J_mm3",
            "Relative_Density_percent",
            "Often expected positive within valid process window",
        ),
        (
            "Energy_Density_J_mm3",
            "Porosity_percent",
            "Often expected negative within valid process window",
        ),
        (
            "Melt_Pool_Depth_um",
            "Relative_Density_percent",
            "Deeper stable melt pool may improve fusion",
        ),
        (
            "Laser_Power_W",
            "Melt_Pool_Depth_um",
            "Higher power may increase melt pool depth",
        ),
        (
            "Laser_Power_W",
            "Melt_Pool_Temp_C",
            "Higher power may increase melt pool temperature",
        ),
        (
            "Thermal_Gradient",
            "Cooling_Rate_K_s",
            "Thermal gradient and cooling rate may be related",
        ),
        (
            "Relative_Density_percent",
            "Porosity_percent",
            "Expected negative relationship between density and porosity",
        ),
        (
            "Defect_Count",
            "Quality_Class",
            "Defects should relate to quality outcome",
        ),
    ]

    rows = []

    quality_mapping = {
        "Poor": 0,
        "Moderate": 1,
        "Good": 2,
        "Excellent": 3,
    }

    work_df = df.copy()

    if TARGET_COLUMN in work_df.columns:
        work_df["Quality_Class_numeric"] = work_df[TARGET_COLUMN].map(
            quality_mapping
        )

    for left, right, expectation in expected_pairs:
        right_col = "Quality_Class_numeric" if right == "Quality_Class" else right

        if left not in work_df.columns or right_col not in work_df.columns:
            continue

        corr = work_df[[left, right_col]].corr().iloc[0, 1]

        rows.append(
            {
                "relationship": f"{left} vs {right}",
                "correlation": corr,
                "absolute_correlation": abs(corr),
                "expected_physics": expectation,
                "interpretation": (
                    "Weak / not visible in this dataset"
                    if abs(corr) < 0.10
                    else "Visible relationship"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "absolute_correlation",
        ascending=False,
    )


def challenge_energy_density_formula(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business objective
    ------------------
    Verify whether Energy_Density_J_mm3 is physically consistent with the common
    LPBF volumetric energy density formula:

        E = P / (v * h * t)

    where:
    - P = laser power
    - v = scan speed
    - h = hatch spacing
    - t = layer thickness converted from micrometers to millimeters

    Coding objective
    ----------------
    Recalculate energy density and compare it with the dataset column.
    """

    required = {
        "Laser_Power_W",
        "Scan_Speed_mm_s",
        "Hatch_Spacing_mm",
        "Layer_Thickness_um",
        "Energy_Density_J_mm3",
    }

    if not required.issubset(df.columns):
        return pd.DataFrame(
            [
                {
                    "check": "energy_density_formula",
                    "status": "SKIPPED",
                    "reason": "Required columns not available",
                }
            ]
        )

    work = df[list(required)].dropna().copy()

    if work.empty:
        return pd.DataFrame(
            [
                {
                    "check": "energy_density_formula",
                    "status": "SKIPPED",
                    "reason": "No complete rows available",
                }
            ]
        )

    layer_thickness_mm = work["Layer_Thickness_um"] / 1000.0

    calculated = (
        work["Laser_Power_W"]
        / (
            work["Scan_Speed_mm_s"]
            * work["Hatch_Spacing_mm"]
            * layer_thickness_mm
        )
    )

    observed = work["Energy_Density_J_mm3"]

    correlation = observed.corr(calculated)
    mae = float(np.mean(np.abs(observed - calculated)))
    mean_absolute_percent_error = float(
        np.mean(np.abs((observed - calculated) / observed)) * 100
    )

    return pd.DataFrame(
        [
            {
                "check": "energy_density_formula",
                "rows_checked": len(work),
                "correlation_observed_vs_calculated": correlation,
                "mean_absolute_error": mae,
                "mean_absolute_percent_error": mean_absolute_percent_error,
                "status": "PASS" if correlation >= 0.95 else "WARN",
                "interpretation": (
                    "Energy density appears physically derived from process parameters"
                    if correlation >= 0.95
                    else "Energy density does not strongly match the common LPBF formula; column may be synthetic or independently generated"
                ),
            }
        ]
    )


def challenge_quality_class_separation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business objective
    ------------------
    Check whether features actually differ across quality classes.

    Coding objective
    ----------------
    Use Kruskal-Wallis tests as a non-parametric sanity check for class
    separation. This is not used as final proof, only as a challenge signal.
    """

    rows = []

    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame()

    for col in _safe_numeric_columns(df):
        groups = [
            values[col].dropna().values
            for _, values in df.groupby(TARGET_COLUMN)
            if values[col].dropna().shape[0] > 0
        ]

        if len(groups) < 2:
            continue

        try:
            statistic, p_value = kruskal(*groups)
        except ValueError:
            continue

        rows.append(
            {
                "feature": col,
                "kruskal_statistic": statistic,
                "p_value": p_value,
                "class_separation_signal": (
                    "Weak / no clear class separation"
                    if p_value >= 0.05
                    else "Potential class separation"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("p_value")


def challenge_defect_trend_by_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business objective
    ------------------
    Check whether defect count behaves as expected across quality classes.

    Engineering expectation:
    Poor builds should generally have more defects than Excellent builds.

    Coding objective
    ----------------
    Summarize average defect count by class and check whether the trend is
    directionally consistent.
    """

    if TARGET_COLUMN not in df.columns or "Defect_Count" not in df.columns:
        return pd.DataFrame()

    class_order = ["Poor", "Moderate", "Good", "Excellent"]

    summary = (
        df.groupby(TARGET_COLUMN)["Defect_Count"]
        .agg(["count", "mean", "median", "std"])
        .reindex(class_order)
        .reset_index()
    )

    means = summary["mean"].dropna().tolist()

    expected_decreasing = all(
        left >= right for left, right in zip(means, means[1:])
    )

    summary["trend_check"] = (
        "PASS: defect count decreases with quality"
        if expected_decreasing
        else "WARN: defect count does not decrease clearly with quality"
    )

    return summary


def challenge_quick_model_predictability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business objective
    ------------------
    Run a small baseline model as a sanity check.

    If a simple model performs close to random chance, the dataset may contain
    weak signal or randomly assigned labels.

    Coding objective
    ----------------
    Train a small Random Forest using a simple train/test split and report
    accuracy and balanced accuracy.

    This is not the final model. It is only a challenge check.
    """

    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=[TARGET_COLUMN]).copy()

    feature_cols = _safe_feature_columns(work)

    X = work[feature_cols]
    y = work[TARGET_COLUMN]

    if y.nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "check": "quick_random_forest_predictability",
                    "status": "SKIPPED",
                    "reason": "Target has fewer than 2 classes",
                    "features_used": len(feature_cols),
                    "classes": y.nunique(),
                    "random_chance_baseline": None,
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "interpretation": "Not enough target classes for model sanity check",
                }
            ]
        )

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        return pd.DataFrame(
            [
                {
                    "check": "quick_random_forest_predictability",
                    "status": "SKIPPED",
                    "reason": (
                        "At least one class has fewer than 2 samples, so "
                        "stratified train/test split is not possible"
                    ),
                    "features_used": len(feature_cols),
                    "classes": y.nunique(),
                    "minimum_class_count": min_class_count,
                    "random_chance_baseline": 1.0 / y.nunique(),
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "interpretation": "Dataset too small for reliable model sanity check",
                }
            ]
        )
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        col for col in CATEGORICAL_FEATURES if col in X.columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    random_chance = 1.0 / y.nunique()

    return pd.DataFrame(
        [
            {
                "check": "quick_random_forest_predictability",
                "features_used": len(feature_cols),
                "classes": y.nunique(),
                "random_chance_baseline": random_chance,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "interpretation": (
                    "Signal appears weak / close to random"
                    if balanced_accuracy < random_chance + 0.10
                    else "Model detects some predictive signal"
                ),
            }
        ]
    )


def challenge_leakage_and_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business objective
    ------------------
    Identify columns that should not be blindly used for operational prediction.

    Coding objective
    ----------------
    Flag identifiers and post-build features.
    """

    rows = []

    for col in df.columns:
        if col in IDENTIFIER_COLUMNS:
            rows.append(
                {
                    "feature": col,
                    "risk_type": "Identifier leakage / no physical meaning",
                    "recommendation": "Exclude from EDA feature ranking and model training",
                }
            )

        if col in POST_BUILD_FEATURES:
            rows.append(
                {
                    "feature": col,
                    "risk_type": "Timing leakage",
                    "recommendation": "Use only for post-build diagnosis, not operational prediction before inspection",
                }
            )

    return pd.DataFrame(rows)

 

def _looks_like_target_leakage_column(col: str) -> bool:
    col_lower = col.lower()
    return any(pattern in col_lower for pattern in LEAKAGE_NAME_PATTERNS)


def _safe_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Business objective
    ------------------
    Prevent accidental target leakage from helper columns created during EDA.

    Coding objective
    ----------------
    Return feature columns excluding IDs, target, and target-derived columns.
    """

    return [
        col
        for col in df.columns
        if col not in IDENTIFIER_COLUMNS
        and col != TARGET_COLUMN
        and not _looks_like_target_leakage_column(col)
    ]

def run_challenge_analysis(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """
    Business objective
    ------------------
    Produce independent challenge reports that verify or challenge our initial
    EDA conclusions.

    Coding objective
    ----------------
    Save challenge CSV files and return them for plotting/reporting.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = {
        "distribution_challenge": challenge_feature_distributions(df),
        "physics_correlation_challenge": challenge_physics_correlations(df),
        "energy_density_formula_challenge": challenge_energy_density_formula(df),
        "quality_separation_challenge": challenge_quality_class_separation(df),
        "defect_trend_challenge": challenge_defect_trend_by_quality(df),
        "quick_model_predictability_challenge": challenge_quick_model_predictability(df),
        "leakage_challenge": challenge_leakage_and_identifiers(df),
    }

    for name, report in reports.items():
        report.to_csv(output_dir / f"{name}.csv", index=False)

    return reports