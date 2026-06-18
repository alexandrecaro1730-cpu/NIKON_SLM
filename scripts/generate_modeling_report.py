"""
Generate a modeling report for the LPBF quality prediction workflow.

Business objective
------------------
Create a clear, interview-ready Task 2 report from the saved model comparison
artifacts.

The report explains:
- which production scenarios were evaluated
- which model families were compared
- which model performed best per scenario
- whether the model is recommended for deployment
- why accuracy alone is misleading
- why weak post-build performance is an important data-quality signal
- why weak performance is an important engineering finding, not a project failure
- how the workflow gives a direction for future improvement with better data

This supports the case-study message:

    The goal is not to maximize offline accuracy with leaked or unavailable
    features. The goal is to build a leakage-safe quality prediction workflow
    aligned with real production decisions.

Coding objective
----------------
Read:
- reports/metrics/comparisons/model_comparison.csv
- reports/metrics/comparisons/model_recommendations.csv

Write:
- reports/metrics/modeling_report.md
"""

import argparse
from pathlib import Path

import pandas as pd

from lpbf_quality.config.settings import METRICS_DIR


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Business objective
    ------------------
    Allow the report generator to be reused with default project paths or with
    custom comparison/recommendation files.

    Coding objective
    ----------------
    Return parsed CLI arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--comparison",
        default=METRICS_DIR / "comparisons" / "model_comparison.csv",
        type=Path,
        help="Path to model comparison CSV.",
    )

    parser.add_argument(
        "--recommendations",
        default=METRICS_DIR / "comparisons" / "model_recommendations.csv",
        type=Path,
        help="Path to model recommendations CSV.",
    )

    parser.add_argument(
        "--output",
        default=METRICS_DIR / "modeling_report.md",
        type=Path,
        help="Path to output Markdown report.",
    )

    return parser.parse_args()


def load_required_csv(path: Path, description: str) -> pd.DataFrame:
    """
    Load a required CSV file.

    Business objective
    ------------------
    Fail early if expected modeling artifacts are missing. This prevents a
    misleading or empty report from being generated.

    Coding objective
    ----------------
    Validate that the file exists and return it as a pandas DataFrame.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. "
            "Run scripts/compare_models.py before generating the report."
        )

    return pd.read_csv(path)


def format_float(value: float | int | str | None, digits: int = 3) -> str:
    """
    Format numeric values for the Markdown report.

    Business objective
    ------------------
    Keep report metrics readable for reviewers and presentation slides.

    Coding objective
    ----------------
    Convert numeric values to fixed decimal strings and gracefully handle
    missing values.
    """

    if pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return str(value)


def get_best_model_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best model per scenario.

    Business objective
    ------------------
    Summarize which model won each production timing scenario.

    Coding objective
    ----------------
    Sort by macro F1 first and balanced accuracy second, then keep the top row
    per scenario.
    """

    required_columns = {
        "scenario",
        "model_type",
        "macro_f1",
        "balanced_accuracy",
    }

    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        raise ValueError(
            f"Comparison CSV is missing required columns: {sorted(missing_columns)}"
        )

    sorted_comparison = comparison.sort_values(
        ["scenario", "macro_f1", "balanced_accuracy"],
        ascending=[True, False, False],
    )

    return sorted_comparison.groupby("scenario", as_index=False).head(1)


def build_model_ladder_section(comparison: pd.DataFrame) -> str:
    """
    Build a report section explaining the model complexity ladder.

    Business objective
    ------------------
    Show that this is a real data-science modeling comparison, not only a system
    architecture exercise.

    Coding objective
    ----------------
    Generate Markdown text describing the model categories used.
    """

    required_columns = {"model_type", "model_family", "complexity_level"}

    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        raise ValueError(
            f"Comparison CSV is missing required columns: {sorted(missing_columns)}"
        )

    available_models = comparison[
        ["model_type", "model_family", "complexity_level"]
    ].drop_duplicates()

    available_models = available_models.sort_values("complexity_level")

    lines = [
        "## Model complexity ladder",
        "",
        "The modeling workflow compares increasing levels of model complexity:",
        "",
    ]

    for _, row in available_models.iterrows():
        lines.append(
            f"- **{row['model_type']}** "
            f"({row['model_family']}, complexity level {row['complexity_level']})"
        )

    lines.extend(
        [
            "",
            "This is a controlled comparison, not AutoML. The purpose is to test "
            "whether additional model complexity produces meaningful improvement "
            "over simple baselines.",
            "",
        ]
    )

    return "\n".join(lines)


def build_scenario_summary_section(
    comparison: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> str:
    """
    Build a scenario-level modeling summary.

    Business objective
    ------------------
    Communicate which model performed best for each production timing scenario
    and whether it is suitable for deployment.

    Coding objective
    ----------------
    Generate a Markdown table from comparison and recommendation artifacts.
    """

    best_models = get_best_model_rows(comparison)

    merged = best_models.merge(
        recommendations,
        on="scenario",
        how="left",
        suffixes=("", "_recommendation"),
    )

    lines = [
        "## Scenario-level results",
        "",
        "| Scenario | Best model | Balanced accuracy | Macro F1 | Deployment status |",
        "|---|---|---:|---:|---|",
    ]

    for _, row in merged.iterrows():
        deployment_status = row.get("deployment_status", "N/A")

        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{row['model_type']} | "
            f"{format_float(row['balanced_accuracy'])} | "
            f"{format_float(row['macro_f1'])} | "
            f"{deployment_status} |"
        )

    lines.append("")

    return "\n".join(lines)


def build_inbuild_mvp_section(comparison: pd.DataFrame) -> str:
    """
    Build a focused section for the in-build MVP.

    Business objective
    ------------------
    Highlight the production MVP scenario because it balances actionability and
    available signal.

    Coding objective
    ----------------
    Extract in-build model results and generate a readable Markdown section.
    """

    inbuild = comparison[comparison["scenario"] == "inbuild"].copy()

    if inbuild.empty:
        return (
            "## In-build MVP interpretation\n\n"
            "No in-build comparison results were found.\n"
        )

    inbuild = inbuild.sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=False,
    )

    best = inbuild.iloc[0]

    lines = [
        "## In-build MVP interpretation",
        "",
        "The in-build scenario is the recommended production MVP because it uses "
        "pre-build parameters plus monitoring features that could plausibly be "
        "available during manufacturing.",
        "",
        f"The best in-build model was **{best['model_type']}**, with:",
        "",
        f"- balanced accuracy: **{format_float(best['balanced_accuracy'])}**",
        f"- macro F1: **{format_float(best['macro_f1'])}**",
        "",
        "This result should be interpreted cautiously. If the best model only "
        "slightly improves over the DummyClassifier, the limiting factor is likely "
        "data signal and physical realism rather than only model choice.",
        "",
    ]

    return "\n".join(lines)


def build_postbuild_interpretation_section(comparison: pd.DataFrame) -> str:
    """
    Build a post-build diagnostic interpretation section.

    Business objective
    ------------------
    Explain why weak post-build performance is an important data-quality signal.

    Coding objective
    ----------------
    Generate Markdown text that connects post-build model results to the Task 1
    finding of weak physical realism.
    """

    postbuild = comparison[comparison["scenario"] == "postbuild"].copy()

    if postbuild.empty:
        return (
            "## Post-build diagnostic interpretation\n\n"
            "No post-build comparison results were found.\n"
        )

    postbuild = postbuild.sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=False,
    )

    best = postbuild.iloc[0]

    return f"""
## Post-build diagnostic interpretation

The post-build diagnostic scenario allows inspection and mechanical-test
features. In a physically coherent dataset, this scenario would normally be
expected to perform better than the operational pre-build and in-build models.

The best post-build model was **{best['model_type']}**, with:

- balanced accuracy: **{format_float(best['balanced_accuracy'])}**
- macro F1: **{format_float(best['macro_f1'])}**

Because post-build performance remains close to the naive baseline, this result
further supports the Task 1 finding that the dataset does not strongly preserve
expected LPBF physics or quality relationships.
""".strip()


def build_full_comparison_table(comparison: pd.DataFrame) -> str:
    """
    Build a full model comparison table.

    Business objective
    ------------------
    Provide an auditable result table for documentation and reviewer inspection.

    Coding objective
    ----------------
    Generate a Markdown table with all scenario/model metrics.
    """

    display_columns = [
        "scenario",
        "model_type",
        "model_family",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "accuracy",
    ]

    available_columns = [
        column for column in display_columns if column in comparison.columns
    ]

    table_df = comparison[available_columns].copy()

    sort_columns = [
        column
        for column in ["scenario", "macro_f1", "balanced_accuracy"]
        if column in table_df.columns
    ]

    ascending = [True, False, False][: len(sort_columns)]

    table_df = table_df.sort_values(sort_columns, ascending=ascending)

    lines = [
        "## Full model comparison",
        "",
        "| Scenario | Model | Family | Balanced accuracy | Macro F1 | Weighted F1 | Accuracy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for _, row in table_df.iterrows():
        lines.append(
            "| "
            f"{row.get('scenario', 'N/A')} | "
            f"{row.get('model_type', 'N/A')} | "
            f"{row.get('model_family', 'N/A')} | "
            f"{format_float(row.get('balanced_accuracy'))} | "
            f"{format_float(row.get('macro_f1'))} | "
            f"{format_float(row.get('weighted_f1'))} | "
            f"{format_float(row.get('accuracy'))} |"
        )

    lines.append("")

    return "\n".join(lines)


def build_accuracy_warning_section() -> str:
    """
    Build a section explaining why accuracy is not the primary selection metric.

    Business objective
    ------------------
    Prevent reviewers from misinterpreting high DummyClassifier accuracy as a
    useful production result.

    Coding objective
    ----------------
    Generate static Markdown text explaining metric choice.
    """

    return """
## Why accuracy alone is misleading

The DummyClassifier achieves relatively high accuracy because it predicts the
majority class. However, its balanced accuracy and macro F1 remain weak. This
means it does not provide useful class-sensitive quality prediction.

For this reason, model selection is based primarily on macro F1 and balanced
accuracy rather than accuracy alone. These metrics better reflect whether the
model can detect all quality classes, including operationally important risk
classes such as Poor and Moderate.
""".strip()


def build_recommendation_section(recommendations: pd.DataFrame) -> str:
    """
    Build a deployment recommendation section.

    Business objective
    ------------------
    Explain that the best model is not automatically production-ready.

    Coding objective
    ----------------
    Generate Markdown text from recommendation artifacts.
    """

    lines = [
        "## Deployment recommendation",
        "",
    ]

    for _, row in recommendations.iterrows():
        lines.extend(
            [
                f"### {row['scenario']}",
                "",
                f"- Best model: **{row.get('best_model_type', 'N/A')}**",
                f"- Deployment status: **{row.get('deployment_status', 'N/A')}**",
                f"- Balanced accuracy lift over Dummy: "
                f"**{format_float(row.get('balanced_accuracy_lift_over_dummy'))}**",
                f"- Macro F1 lift over Dummy: "
                f"**{format_float(row.get('macro_f1_lift_over_dummy'))}**",
                "",
                str(row.get("recommendation", "")),
                "",
            ]
        )

    return "\n".join(lines)


def build_final_interpretation_section() -> str:
    """
    Build the final business interpretation section.

    Business objective
    ------------------
    Give a clear conclusion that can be reused in README, PowerPoint, and the
    final case-study explanation.

    Coding objective
    ----------------
    Return static Markdown text aligned with the project strategy.
    """

    return """
## Final interpretation

The Task 2 modeling workflow is successful as an engineering and data-science
proof of concept.

It demonstrates:

- leakage-safe feature views by production decision timing
- a naive baseline
- an explainable statistical baseline
- nonlinear machine-learning baselines
- a stronger tabular boosting candidate
- class-sensitive evaluation metrics
- deployment-readiness checks
- saved metrics and model artifacts

However, the trained models should not be presented as trusted LPBF production
predictors on the current dataset.

The key finding is that increasing model complexity does not produce enough
improvement over the naive baseline. Even the post-build diagnostic scenario,
which includes inspection and mechanical-test features, remains close to
baseline performance. This supports the Task 1 conclusion that the current
dataset is technically suitable for demonstrating the workflow, but does not
contain strong enough physically coherent predictive signal to justify
deployment.

The recommended next step is not to over-tune the model. The recommended next
step is to collect or validate higher-quality production data with stronger
links between process parameters, melt-pool monitoring, inspection results, and
quality outcomes.

With better data, the same framework can show whether moving from statistical
baselines to nonlinear ensembles or boosting becomes justified. The current
result is therefore useful because it gives a disciplined path for future
improvement without overclaiming production readiness.
""".strip()


def build_report(
    comparison: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> str:
    """
    Build the full Markdown report.

    Business objective
    ------------------
    Produce a presentation-ready Task 2 modeling report.

    Coding objective
    ----------------
    Combine all report sections into one Markdown string.
    """

    sections = [
        "# LPBF Quality Prediction Modeling Report",
        "",
        "## Purpose",
        "",
        "This report summarizes Task 2 of the LPBF quality prediction project.",
        "",
        "The goal is not to maximize offline accuracy using every available "
        "column. The goal is to evaluate leakage-safe prediction workflows that "
        "match real production decision timing.",
        "",
        build_model_ladder_section(comparison),
        build_scenario_summary_section(comparison, recommendations),
        build_inbuild_mvp_section(comparison),
        build_postbuild_interpretation_section(comparison),
        build_full_comparison_table(comparison),
        build_accuracy_warning_section(),
        build_recommendation_section(recommendations),
        build_final_interpretation_section(),
        "",
    ]

    return "\n".join(sections)


def main() -> None:
    """
    Generate the modeling report.

    Business objective
    ------------------
    Convert model comparison artifacts into a readable report for project
    documentation and presentation.

    Coding objective
    ----------------
    Load CSVs, build Markdown, and save the output file.
    """

    args = parse_args()

    comparison = load_required_csv(
        path=args.comparison,
        description="model comparison CSV",
    )

    recommendations = load_required_csv(
        path=args.recommendations,
        description="model recommendations CSV",
    )

    report = build_report(
        comparison=comparison,
        recommendations=recommendations,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(f"Saved modeling report: {args.output}")


if __name__ == "__main__":
    main()