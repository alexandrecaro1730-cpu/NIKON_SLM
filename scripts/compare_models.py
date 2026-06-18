"""
Compare LPBF quality models across production scenarios.

Business objective
------------------
Provide a clear data-science comparison that supports the case-study story:

- We test increasing model complexity.
- We do not assume the most complex model is best.
- We judge whether improvement over the naive baseline is meaningful.
- We use different model-selection metrics for different production timelines.

Scenario-specific selection policy
----------------------------------
prebuild:
    Select by risk_f2, then risk_recall.
    Business reason: catch risky jobs before starting the build.

inbuild:
    Select by risk_f2, then risk_recall.
    Business reason: catch deteriorating builds while action is still possible.

postbuild:
    Select by macro_f1, then balanced_accuracy.
    Business reason: explain all final quality outcomes, not only binary risk.

Coding objective
----------------
Train candidate models for one or more scenarios and save:
- model comparison CSV
- recommendation CSV
"""

import argparse

import pandas as pd

from lpbf_quality.config.settings import METRICS_DIR
from lpbf_quality.data.load_data import load_csv
from lpbf_quality.models.model_selection import (
    build_model_recommendation,
    compare_models_for_scenario,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/raw/lpbf_titanium_dataset.csv",
        help="Path to input CSV file.",
    )

    parser.add_argument(
        "--scenario",
        choices=["prebuild", "inbuild", "postbuild", "all"],
        default="all",
        help="Production timing scenario to compare.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout test fraction.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def expand_scenarios(scenario: str) -> list[str]:
    """
    Expand the scenario CLI argument.
    """

    if scenario == "all":
        return ["prebuild", "inbuild", "postbuild"]

    return [scenario]


def build_display_table(
    comparison: pd.DataFrame,
    recommendation: dict,
) -> pd.DataFrame:
    """
    Build a terminal-friendly model comparison table.

    Business objective
    ------------------
    Show the reviewer which metric was actually used to select the best model.

    Coding objective
    ----------------
    Add generic display columns:
    - primary_metric_name
    - primary_metric_value
    - secondary_metric_name
    - secondary_metric_value

    This prevents confusion when different scenarios use different metrics.
    """

    primary_metric = recommendation["selection_primary_metric"]
    secondary_metric = recommendation["selection_secondary_metric"]

    display = comparison.copy()

    display["primary_metric"] = primary_metric
    display["primary_value"] = display[primary_metric]

    display["secondary_metric"] = secondary_metric
    display["secondary_value"] = display[secondary_metric]

    display_columns = [
        "model_type",
        "primary_metric",
        "primary_value",
        "secondary_metric",
        "secondary_value",
        "balanced_accuracy",
        "macro_f1",
    ]

    return display[display_columns]


def main() -> None:
    """
    Compare models and save recommendation artifacts.
    """

    args = parse_args()

    comparison_dir = METRICS_DIR / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.input)
    scenarios = expand_scenarios(args.scenario)

    comparison_tables = []
    recommendation_rows = []

    for scenario in scenarios:
        comparison = compare_models_for_scenario(
            df=df,
            scenario=scenario,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        recommendation = build_model_recommendation(comparison)
        recommendation["scenario"] = scenario

        comparison_tables.append(comparison)
        recommendation_rows.append(recommendation)

        display = build_display_table(
            comparison=comparison,
            recommendation=recommendation,
        )

        print()
        print(f"Scenario: {scenario}")
        print(f"Primary selection metric: {recommendation['selection_primary_metric']}")
        print(
            f"Secondary selection metric: "
            f"{recommendation['selection_secondary_metric']}"
        )
        print()
        print(display)
        print()
        print(f"Recommendation: {recommendation['deployment_status']}")
        print(recommendation["recommendation"])
        print()
        print("Business reason:")
        print(recommendation["business_selection_reason"])

    all_comparisons = pd.concat(comparison_tables, ignore_index=True)
    recommendations = pd.DataFrame(recommendation_rows)

    comparison_path = comparison_dir / "model_comparison.csv"
    recommendation_path = comparison_dir / "model_recommendations.csv"

    all_comparisons.to_csv(comparison_path, index=False)
    recommendations.to_csv(recommendation_path, index=False)

    print()
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved recommendations: {recommendation_path}")


if __name__ == "__main__":
    main()