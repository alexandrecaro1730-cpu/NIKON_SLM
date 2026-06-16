"""
Exploratory data analysis plots.

This module creates optional deep-dive plots for understanding the dataset.

Why this matters:
- Mandatory checks tell us whether the data is usable.
- Deep EDA helps us understand process behavior, quality drivers, outliers,
  leakage risks, and model objectives.
- The generated figures can be reused directly in the final PowerPoint.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lpbf_quality.config.settings import (
    CATEGORICAL_FEATURES,
    POST_BUILD_FEATURES,
    PROCESS_FEATURES,
    QUALITY_ORDER,
    TARGET_COLUMN,
)


def _save_plot(output_path: Path) -> None:
    """
    Save current matplotlib figure and close it.

    Centralizing this avoids repeated code and ensures all plots are saved
    consistently.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_target_distribution(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Plot number of samples by quality class.

    Business purpose:
    Shows whether the dataset is balanced. If poor builds are rare, recall and
    class-specific metrics become more important than accuracy alone.
    """

    output_dir = Path(output_dir)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, order=QUALITY_ORDER)
    plt.title("Quality Class Distribution")
    plt.xlabel("Quality Class")
    plt.ylabel("Number of Samples")

    _save_plot(output_dir / "target_distribution.png")


def plot_missing_values(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Plot missing-value percentage by column.

    Business purpose:
    Highlights unreliable sensors, incomplete logs, or missing inspection data.
    """

    output_dir = Path(output_dir)

    missing = df.isna().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]

    if missing.empty:
        return

    plt.figure(figsize=(10, max(4, len(missing) * 0.35)))
    missing.plot(kind="barh")
    plt.title("Missing Values by Feature")
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Feature")

    _save_plot(output_dir / "missing_values.png")


def plot_numeric_distributions(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Plot distributions for numeric features.

    Business purpose:
    Helps detect skewed inputs, unusual operating windows, and outliers.
    """

    output_dir = Path(output_dir)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        _save_plot(output_dir / f"distribution_{col}.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Plot correlation heatmap for numeric features.

    Business purpose:
    Shows relationships between process parameters, monitoring signals, and
    post-build quality measurements.
    """

    output_dir = Path(output_dir)

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return

    plt.figure(figsize=(16, 12))
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Numeric Feature Correlation Heatmap")

    _save_plot(output_dir / "correlation_heatmap.png")


def plot_boxplots_by_quality(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Plot numeric features grouped by quality class.

    Business purpose:
    Helps identify which parameters shift between Poor, Moderate, Good,
    and Excellent builds.
    """

    output_dir = Path(output_dir)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        if col == TARGET_COLUMN:
            continue

        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df, x=TARGET_COLUMN, y=col, order=QUALITY_ORDER)
        plt.title(f"{col} by Quality Class")
        plt.xlabel("Quality Class")
        plt.ylabel(col)

        _save_plot(output_dir / f"boxplot_{col}_by_quality.png")


def plot_categorical_relationships(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Plot categorical feature relationships with quality.

    Business purpose:
    Shows whether alloy type, powder morphology, or scan strategy are associated
    with different quality outcomes.
    """

    output_dir = Path(output_dir)

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue

        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=col, hue=TARGET_COLUMN, hue_order=QUALITY_ORDER)
        plt.title(f"{col} vs Quality Class")
        plt.xlabel(col)
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=30)

        _save_plot(output_dir / f"categorical_{col}_vs_quality.png")


def plot_energy_density_process_window(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Plot laser power vs scan speed colored by quality class.

    Business purpose:
    This is a practical LPBF process-window view. It helps communicate whether
    quality classes cluster in certain power/speed operating regions.
    """

    output_dir = Path(output_dir)

    required = {"Laser_Power_W", "Scan_Speed_mm_s", TARGET_COLUMN}

    if not required.issubset(df.columns):
        return

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Scan_Speed_mm_s",
        y="Laser_Power_W",
        hue=TARGET_COLUMN,
        hue_order=QUALITY_ORDER,
        alpha=0.8,
    )
    plt.title("LPBF Process Window: Laser Power vs Scan Speed")
    plt.xlabel("Scan Speed (mm/s)")
    plt.ylabel("Laser Power (W)")

    _save_plot(output_dir / "process_window_power_vs_speed.png")


def save_summary_tables(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Save useful EDA summary tables.

    Business purpose:
    These CSV files provide traceable evidence for observations made in the
    report or presentation.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    df[numeric_cols].describe().T.to_csv(output_dir / "numeric_summary.csv")

    categorical_summary = []

    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        for value, count in counts.items():
            categorical_summary.append(
                {
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "percent": round(float(count / len(df) * 100), 2),
                }
            )

    pd.DataFrame(categorical_summary).to_csv(
        output_dir / "categorical_summary.csv",
        index=False,
    )

    df[TARGET_COLUMN].value_counts(dropna=False).to_csv(
        output_dir / "target_distribution.csv"
    )


def save_leakage_review(output_dir: str | Path) -> None:
    """
    Save a leakage-risk review table.

    Business purpose:
    This documents which features are safe for operational prediction and which
    should be treated as post-build diagnostic information.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for col in PROCESS_FEATURES:
        rows.append(
            {
                "feature": col,
                "group": "process_or_monitoring",
                "production_use": "Allowed",
                "reason": "Available before or during build.",
            }
        )

    for col in POST_BUILD_FEATURES:
        rows.append(
            {
                "feature": col,
                "group": "post_build_inspection",
                "production_use": "Exclude from operational model",
                "reason": "Likely measured after build; may cause data leakage.",
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "leakage_review.csv", index=False)


def run_optional_deep_eda(
    df: pd.DataFrame,
    eda_output_dir: str | Path,
    figures_output_dir: str | Path,
) -> None:
    """
    Run optional deep EDA.

    This is not required for every production prediction, but it is very useful
    during development, model review, and interview presentation preparation.
    """

    save_summary_tables(df, eda_output_dir)
    save_leakage_review(eda_output_dir)

    plot_target_distribution(df, figures_output_dir)
    plot_missing_values(df, figures_output_dir)
    plot_numeric_distributions(df, figures_output_dir)
    plot_correlation_heatmap(df, figures_output_dir)
    plot_boxplots_by_quality(df, figures_output_dir)
    plot_categorical_relationships(df, figures_output_dir)
    plot_energy_density_process_window(df, figures_output_dir)