"""
Presentation-first plots for Task 2 modeling results.

Business objective
------------------
Create simple, slide-ready visuals for the Task 2 Quality Prediction Model.

The presentation story is:

1. The project is not just one classifier.
   It is a four-stage LPBF quality intelligence workflow.

2. Different production stages need different success metrics.
   Pre-build and in-build prioritize risk detection.
   Post-build prioritizes balanced four-class diagnosis.
   Monitoring prioritizes drift and trend signals.

3. Logistic Regression is the best operational risk-detection model.
   It has the highest Risk F2 for pre-build and in-build.

4. Random Forest can look stronger on generic multi-class metrics,
   but it is weaker for the operational risk-detection decision.

5. The model is not production-ready without monitoring and validation on
   physically coherent production data.

Coding objective
----------------
Read:
- reports/metrics/comparisons/model_comparison.csv
- reports/metrics/comparisons/model_recommendations.csv

Write:
- reports/figures/task2/slide1_quality_intelligence_framework.png
- reports/figures/task2/slide2_metric_policy_table.png
- reports/figures/task2/slide3_operational_risk_f2.png
- reports/figures/task2/slide4_inbuild_tradeoff.png
- reports/figures/task2/slide5_monitoring_plan.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lpbf_quality.config.settings import FIGURES_DIR, METRICS_DIR


SCENARIO_ORDER = ["prebuild", "inbuild", "postbuild"]

MODEL_ORDER = [
    "dummy",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
]

MODEL_DISPLAY_NAMES = {
    "dummy": "Dummy",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist. Gradient Boosting",
}

SCENARIO_DISPLAY_NAMES = {
    "prebuild": "Pre-build",
    "inbuild": "In-build",
    "postbuild": "Post-build",
}

STATUS_DISPLAY_NAMES = {
    "CANDIDATE_FOR_ENGINEERING_VALIDATION": "Validation candidate",
    "NOT_RECOMMENDED_FOR_DEPLOYMENT": "Not deployment-ready",
    "REVIEW_REQUIRED": "Review required",
}


def ensure_output_dir(output_dir: Path | None = None) -> Path:
    """
    Create and return the Task 2 figure directory.
    """

    output_dir = output_dir or (FIGURES_DIR / "task2")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model_comparison(comparison_path: Path | None = None) -> pd.DataFrame:
    """
    Load model comparison results.
    """

    comparison_path = comparison_path or (
        METRICS_DIR / "comparisons" / "model_comparison.csv"
    )

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing model comparison CSV: {comparison_path}. "
            "Run scripts/compare_models.py --scenario all first."
        )

    return pd.read_csv(comparison_path)


def load_model_recommendations(
    recommendations_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load model recommendation results.
    """

    recommendations_path = recommendations_path or (
        METRICS_DIR / "comparisons" / "model_recommendations.csv"
    )

    if not recommendations_path.exists():
        raise FileNotFoundError(
            f"Missing model recommendations CSV: {recommendations_path}. "
            "Run scripts/compare_models.py --scenario all first."
        )

    return pd.read_csv(recommendations_path)


def display_model_name(model_type: str) -> str:
    """
    Convert internal model type into a readable label.
    """

    return MODEL_DISPLAY_NAMES.get(model_type, model_type)


def display_scenario_name(scenario: str) -> str:
    """
    Convert internal scenario name into a readable label.
    """

    return SCENARIO_DISPLAY_NAMES.get(scenario, scenario)


def display_status(status: str) -> str:
    """
    Convert long deployment status into a slide-friendly label.
    """

    return STATUS_DISPLAY_NAMES.get(status, status)


def save_figure(output_path: Path) -> None:
    """
    Save the current matplotlib figure.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def get_operational_subset(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows for pre-build and in-build operational risk models.
    """

    required_columns = {
        "scenario",
        "model_type",
        "risk_f2",
        "risk_recall",
        "macro_f1",
        "balanced_accuracy",
    }

    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        raise ValueError(
            f"Comparison CSV is missing required columns: {sorted(missing_columns)}"
        )

    return comparison[comparison["scenario"].isin(["prebuild", "inbuild"])].copy()


def plot_slide1_quality_intelligence_framework(output_path: Path) -> None:
    """
    Slide 1: explain the four-stage business framework.

    Business objective
    ------------------
    Make it clear that this project is a production decision system, not just a
    one-off classifier.
    """

    stages = [
        {
            "title": "1. Pre-build\nrisk screening",
            "question": "Should this build be flagged\nbefore machine time is spent?",
            "metric": "Risk F2",
        },
        {
            "title": "2. In-build\nrisk update",
            "question": "Is risk increasing while\nthere is still time to act?",
            "metric": "Risk F2",
        },
        {
            "title": "3. Post-build\ndiagnosis",
            "question": "What explains the final\nquality outcome?",
            "metric": "Macro F1",
        },
        {
            "title": "4. Production\nmonitoring",
            "question": "Is the process drifting\nor degrading over time?",
            "metric": "Drift signals",
        },
    ]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")

    for index, stage in enumerate(stages):
        x_position = index + 0.5

        ax.text(
            x_position,
            0.78,
            stage["title"],
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.45", "alpha": 0.15},
        )

        ax.text(
            x_position,
            0.47,
            stage["question"],
            ha="center",
            va="center",
            fontsize=12,
        )

        ax.text(
            x_position,
            0.22,
            f"Primary metric:\n{stage['metric']}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x_position + 0.42, 0.78),
                xytext=(x_position + 0.08, 0.78),
                arrowprops={"arrowstyle": "->", "lw": 1.5},
            )

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)

    ax.set_title(
        "LPBF Quality Intelligence: Four Production Objectives",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    save_figure(output_path)


def plot_slide2_metric_policy_table(output_path: Path) -> None:
    """
    Slide 2: show the metric policy as a table.

    Business objective
    ------------------
    This is clearer than a bar chart because the values are not quantities.
    They are policy choices.
    """

    rows = [
        [
            "Pre-build",
            "Risk screening",
            "Risk F2",
            "Risk recall",
            "Missing a risky build wastes machine time and powder",
        ],
        [
            "In-build",
            "Risk update",
            "Risk F2",
            "Risk recall",
            "Missing risk removes the chance to intervene",
        ],
        [
            "Post-build",
            "Diagnosis",
            "Macro F1",
            "Balanced accuracy",
            "All four quality outcomes should be explained fairly",
        ],
        [
            "Monitoring",
            "Production control",
            "Drift signals",
            "Alert thresholds",
            "Detect process, machine, powder, or model degradation",
        ],
    ]

    columns = [
        "Objective",
        "Business use",
        "Primary metric",
        "Tie-breaker",
        "Reason",
    ]

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for column_index in range(len(columns)):
        table[(0, column_index)].set_text_props(fontweight="bold")

    ax.set_title(
        "Model Selection Metric Depends on the Production Decision",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    save_figure(output_path)


def plot_slide3_operational_risk_f2(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Slide 3: compare the two important operational candidates.

    Business objective
    ------------------
    Make the operational model-selection result easy to present.

    Logistic Regression is selected because it has higher Risk F2 for both
    pre-build and in-build risk detection.
    """

    operational = get_operational_subset(comparison)

    selected_models = ["logistic_regression", "random_forest"]

    operational = operational[
        operational["model_type"].isin(selected_models)
    ].copy()

    pivot = operational.pivot(
        index="scenario",
        columns="model_type",
        values="risk_f2",
    )

    pivot = pivot.loc[["prebuild", "inbuild"], selected_models]

    pivot.index = [display_scenario_name(scenario) for scenario in pivot.index]
    pivot.columns = [display_model_name(model) for model in pivot.columns]

    ax = pivot.plot(kind="bar", figsize=(11, 6))

    ax.set_title(
        "Operational Risk Detection: Logistic Regression Is the Better Candidate",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_ylabel("Risk F2")
    ax.set_xlabel("Production timing")
    ax.set_ylim(0, max(0.55, float(pivot.max().max()) + 0.08))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Model")

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f",
            padding=3,
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0.01,
        0.97,
        "Risk F2 weights recall more than precision, which fits the cost of missing Poor/Moderate builds.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "alpha": 0.12},
    )

    save_figure(output_path)

def plot_slide4_inbuild_tradeoff(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Slide 4: show the in-build tradeoff using only the two relevant models.

    Business objective
    ------------------
    Make the main reasoning clear:

    - Random Forest has better general four-class classification.
    - Logistic Regression is better for operational Poor/Moderate risk detection.
    """

    inbuild = comparison[comparison["scenario"] == "inbuild"].copy()

    selected_models = ["logistic_regression", "random_forest"]
    inbuild = inbuild[inbuild["model_type"].isin(selected_models)]

    if inbuild.empty:
        raise ValueError("No in-build Logistic Regression or Random Forest rows found.")

    plot_df = inbuild.set_index("model_type").loc[
        selected_models,
        ["risk_f2", "macro_f1"],
    ]

    plot_df.index = [display_model_name(model) for model in plot_df.index]
    plot_df.columns = [
        "Risk F2\nbusiness risk metric",
        "Macro F1\ngeneral ML metric",
    ]

    ax = plot_df.plot(kind="bar", figsize=(12, 6))

    ax.set_title(
        "In-build Model Choice Depends on the Objective",
        fontsize=17,
        fontweight="bold",
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, max(0.55, float(plot_df.max().max()) + 0.08))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Metric", loc="upper right")

    ax.text(
        0.01,
        0.98,
        "Decision: choose Logistic Regression for the in-build MVP because Risk F2 is the business metric.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "alpha": 0.12},
    )

    save_figure(output_path)


def plot_slide5_monitoring_plan(output_path: Path) -> None:
    """
    Slide 5: show the production monitoring layer.

    Business objective
    ------------------
    Explain that deployment is not only a saved model. Production needs ongoing
    monitoring for quality, process, and model behavior.
    """

    monitoring_rows = [
        [
            "Quality outcome drift",
            "Poor + Moderate rate",
            "Detect rising quality risk",
        ],
        [
            "Material / inspection drift",
            "Porosity, density, defect count",
            "Detect degraded physical outcomes",
        ],
        [
            "Process drift",
            "Power, speed, melt-pool, thermal gradient",
            "Detect machine or process-window shift",
        ],
        [
            "Model behavior drift",
            "Prediction confidence and class mix",
            "Detect model reliability degradation",
        ],
        [
            "Action layer",
            "Alerts, review thresholds, retraining trigger",
            "Prevent silent model failure",
        ],
    ]

    columns = ["Monitoring area", "Signals to track", "Why it matters"]

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis("off")

    table = ax.table(
        cellText=monitoring_rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 2.2)

    for column_index in range(len(columns)):
        table[(0, column_index)].set_text_props(fontweight="bold")

    ax.set_title(
        "Production Monitoring: Required Before Real Deployment",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    save_figure(output_path)


def run_modeling_plots(
    comparison_path: Path | None = None,
    recommendations_path: Path | None = None,
    confusion_matrix_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Generate slide-ready Task 2 figures.

    Business objective
    ------------------
    Produce a small set of figures that directly map to the presentation slides.

    Coding objective
    ----------------
    Load saved model comparison artifacts and write PNG files.
    """

    # Kept for backward compatibility with scripts/run_modeling_plots.py.
    _ = recommendations_path
    _ = confusion_matrix_path

    output_dir = ensure_output_dir(output_dir)
    comparison = load_model_comparison(comparison_path)

    outputs = {
        "slide1_quality_intelligence_framework": output_dir
        / "slide1_quality_intelligence_framework.png",
        "slide2_metric_policy_table": output_dir / "slide2_metric_policy_table.png",
        "slide3_operational_risk_f2": output_dir
        / "slide3_operational_risk_f2.png",
        "slide4_inbuild_tradeoff": output_dir / "slide4_inbuild_tradeoff.png",
        "slide5_monitoring_plan": output_dir / "slide5_monitoring_plan.png",
    }

    plot_slide1_quality_intelligence_framework(
        output_path=outputs["slide1_quality_intelligence_framework"],
    )

    plot_slide2_metric_policy_table(
        output_path=outputs["slide2_metric_policy_table"],
    )

    plot_slide3_operational_risk_f2(
        comparison=comparison,
        output_path=outputs["slide3_operational_risk_f2"],
    )

    plot_slide4_inbuild_tradeoff(
        comparison=comparison,
        output_path=outputs["slide4_inbuild_tradeoff"],
    )

    plot_slide5_monitoring_plan(
        output_path=outputs["slide5_monitoring_plan"],
    )

    return outputs