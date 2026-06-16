"""
Statistical and ML-readiness validation.

These checks go beyond basic schema validation.

Business purpose:
- Basic validation tells us whether the file is readable.
- Statistical validation tells us whether the data is suitable for modeling.
- Production risk checks help explain what could go wrong if the model is used
  in a real LPBF production environment.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from lpbf_quality.config.settings import (
    CATEGORICAL_FEATURES,
    POST_BUILD_FEATURES,
    PROCESS_FEATURES,
    QUALITY_ORDER,
    TARGET_COLUMN,
)


def build_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect statistical outliers using the IQR rule.

    Business note:
    Outliers are not automatically removed because extreme values may represent
    valid experimental builds or edge process windows.
    """

    rows = []
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        series = df[col].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        rows.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(len(outliers)),
                "outlier_percent": round(len(outliers) / len(df) * 100, 2),
                "status": "WARN" if len(outliers) > 0 else "PASS",
                "interpretation": (
                    "Outliers detected. Review before removing; values may be "
                    "valid production experiments."
                    if len(outliers) > 0
                    else "No IQR outliers detected."
                ),
            }
        )

    return pd.DataFrame(rows)


def build_class_balance_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze target class balance.

    Business note:
    If poor-quality builds are rare, accuracy can become misleading. In that
    case, recall and F1-score for risky classes become more important.
    """

    counts = df[TARGET_COLUMN].value_counts(dropna=False)
    total = len(df)

    min_count = counts.min()
    max_count = counts.max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    rows = []

    for label, count in counts.items():
        rows.append(
            {
                "quality_class": label,
                "count": int(count),
                "percent": round(count / total * 100, 2),
                "imbalance_ratio_majority_to_minority": round(imbalance_ratio, 2),
                "status": "WARN" if imbalance_ratio > 3 else "PASS",
            }
        )

    return pd.DataFrame(rows)


def build_correlation_report(
    df: pd.DataFrame,
    threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Detect highly correlated numeric feature pairs.

    Business note:
    Highly correlated features are not always bad, but they may make model
    interpretation harder and can indicate duplicated information.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    corr = numeric_df.corr()

    rows = []
    cols = corr.columns.tolist()

    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = corr.loc[col_a, col_b]

            if abs(value) >= threshold:
                rows.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(value, 4),
                        "absolute_correlation": round(abs(value), 4),
                        "status": "WARN",
                        "interpretation": (
                            "High correlation detected. Review redundancy and "
                            "impact on model explainability."
                        ),
                    }
                )

    if not rows:
        rows.append(
            {
                "feature_a": None,
                "feature_b": None,
                "correlation": None,
                "absolute_correlation": None,
                "status": "PASS",
                "interpretation": "No highly correlated feature pairs detected.",
            }
        )

    return pd.DataFrame(rows)


def build_constant_feature_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect constant or near-constant features.

    Business note:
    Features with no variation cannot help prediction and may indicate sensor
    export issues or unused columns.
    """

    rows = []

    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)
        unique_ratio = unique_count / len(df)

        if unique_count <= 1:
            status = "WARN"
            interpretation = "Constant feature. Remove from modeling."
        elif unique_ratio < 0.01:
            status = "WARN"
            interpretation = "Near-constant feature. Review usefulness."
        else:
            status = "PASS"
            interpretation = "Sufficient variation."

        rows.append(
            {
                "feature": col,
                "unique_count": int(unique_count),
                "unique_ratio": round(unique_ratio, 4),
                "status": status,
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def build_skewness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure numeric feature skewness.

    Business note:
    Strong skew can affect some models and may indicate process values with
    long tails or abnormal operating conditions.
    """

    rows = []

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        skewness = df[col].skew()

        rows.append(
            {
                "feature": col,
                "skewness": round(float(skewness), 4),
                "absolute_skewness": round(abs(float(skewness)), 4),
                "status": "WARN" if abs(skewness) > 1 else "PASS",
                "interpretation": (
                    "Strong skew detected. Consider robust scaling or "
                    "transformation if using sensitive models."
                    if abs(skewness) > 1
                    else "No strong skew detected."
                ),
            }
        )

    return pd.DataFrame(rows)


def build_leakage_candidate_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag known post-build inspection features as leakage risks.

    Business note:
    These columns may be valid for diagnosis but should not be used in an
    operational model that predicts quality before final inspection.
    """

    rows = []

    for feature in POST_BUILD_FEATURES:
        if feature not in df.columns:
            continue

        rows.append(
            {
                "feature": feature,
                "risk_level": "HIGH",
                "production_use": "EXCLUDE_FROM_OPERATIONAL_MODEL",
                "status": "WARN",
                "reason": (
                    "Likely measured after build completion. Using it for "
                    "prediction could leak the answer into the model."
                ),
            }
        )

    for feature in PROCESS_FEATURES + CATEGORICAL_FEATURES:
        if feature not in df.columns:
            continue

        rows.append(
            {
                "feature": feature,
                "risk_level": "LOW",
                "production_use": "ALLOWED",
                "status": "PASS",
                "reason": "Feature is available before or during production.",
            }
        )

    return pd.DataFrame(rows)


def build_mutual_information_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate feature relevance to Quality_Class using mutual information.

    Business note:
    This is an early signal of potentially useful predictors. It is not final
    model explainability, but it helps focus engineering discussion.
    """

    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame()

    feature_df = df.drop(columns=[TARGET_COLUMN]).copy()
    target = df[TARGET_COLUMN]

    # Drop ID-like columns because they should not explain production quality.
    for col in ["Sample_ID"]:
        if col in feature_df.columns:
            feature_df = feature_df.drop(columns=[col])

    categorical_cols = feature_df.select_dtypes(exclude="number").columns
    numeric_cols = feature_df.select_dtypes(include="number").columns

    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(
        feature_df[numeric_cols].median()
    )

    if len(categorical_cols) > 0:
        feature_df[categorical_cols] = feature_df[categorical_cols].fillna("Missing")
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        feature_df[categorical_cols] = encoder.fit_transform(feature_df[categorical_cols])

    scores = mutual_info_classif(
        feature_df,
        target,
        discrete_features="auto",
        random_state=42,
    )

    report = pd.DataFrame(
        {
            "feature": feature_df.columns,
            "mutual_information_score": scores,
        }
    )

    report["mutual_information_score"] = report[
        "mutual_information_score"
    ].round(6)

    report = report.sort_values(
        "mutual_information_score",
        ascending=False,
    )

    report["status"] = "INFO"
    report["interpretation"] = (
        "Higher values suggest stronger relationship with Quality_Class."
    )

    return report


def build_production_risk_assessment(
    statistical_reports: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize major production risks from statistical reports.

    Business note:
    This creates a concise table for reviewers and PowerPoint communication.
    """

    rows = []

    leakage_report = statistical_reports.get("leakage_candidates", pd.DataFrame())
    leakage_warnings = (
        int((leakage_report["status"] == "WARN").sum())
        if "status" in leakage_report.columns
        else 0
    )

    rows.append(
        {
            "risk": "Data leakage",
            "severity": "HIGH" if leakage_warnings > 0 else "LOW",
            "evidence": f"{leakage_warnings} leakage-risk features detected.",
            "recommended_action": (
                "Exclude post-build inspection features from operational model."
            ),
        }
    )

    outlier_report = statistical_reports.get("outliers", pd.DataFrame())
    outlier_features = (
        int((outlier_report["outlier_count"] > 0).sum())
        if "outlier_count" in outlier_report.columns
        else 0
    )

    rows.append(
        {
            "risk": "Outliers / unusual process windows",
            "severity": "MEDIUM" if outlier_features > 0 else "LOW",
            "evidence": f"{outlier_features} numeric features contain IQR outliers.",
            "recommended_action": (
                "Review outliers with process engineers before removing them."
            ),
        }
    )

    class_report = statistical_reports.get("class_balance", pd.DataFrame())
    imbalance_warn = (
        bool((class_report["status"] == "WARN").any())
        if "status" in class_report.columns
        else False
    )

    rows.append(
        {
            "risk": "Class imbalance",
            "severity": "MEDIUM" if imbalance_warn else "LOW",
            "evidence": "Class imbalance warning detected."
            if imbalance_warn
            else "No severe class imbalance detected.",
            "recommended_action": (
                "Use class-specific recall/F1 metrics, not accuracy only."
            ),
        }
    )

    corr_report = statistical_reports.get("correlations", pd.DataFrame())
    corr_warn = (
        int((corr_report["status"] == "WARN").sum())
        if "status" in corr_report.columns
        else 0
    )

    rows.append(
        {
            "risk": "Feature redundancy",
            "severity": "MEDIUM" if corr_warn > 0 else "LOW",
            "evidence": f"{corr_warn} highly correlated feature pairs detected.",
            "recommended_action": (
                "Review correlated features for interpretability and robustness."
            ),
        }
    )

    rows.append(
        {
            "risk": "Machine/process drift",
            "severity": "MEDIUM",
            "evidence": "Dataset does not confirm future machines or powders match training distribution.",
            "recommended_action": (
                "Monitor input distributions and retrain when drift is detected."
            ),
        }
    )

    return pd.DataFrame(rows)


def build_statistical_validation_reports(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Run all statistical validation reports.
    """

    reports = {
        "outliers": build_outlier_report(df),
        "class_balance": build_class_balance_report(df),
        "correlations": build_correlation_report(df),
        "constant_features": build_constant_feature_report(df),
        "skewness": build_skewness_report(df),
        "leakage_candidates": build_leakage_candidate_report(df),
        "mutual_information": build_mutual_information_report(df),
    }

    reports["production_risk_assessment"] = build_production_risk_assessment(reports)

    return reports