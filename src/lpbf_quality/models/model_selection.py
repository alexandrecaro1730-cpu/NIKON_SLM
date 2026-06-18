"""
Model selection for LPBF quality prediction.

Business objective
------------------
Select models based on evidence and production priorities.

The current dataset has known limitations, so the system should not simply pick
the most complex model. It should compare:

- naive baseline
- statistical baseline
- nonlinear ML baseline
- stronger tabular boosting baseline

Then it should select models using scenario-specific business metrics:

- prebuild: risk_f2, then risk_recall
- inbuild: risk_f2, then risk_recall
- postbuild: macro_f1, then balanced_accuracy

Coding objective
----------------
Train multiple candidates for one scenario, compare metrics, and produce a
selection report.
"""

import pandas as pd

from lpbf_quality.models.model_registry import get_available_model_types
from lpbf_quality.models.selection_policy import get_selection_policy
from lpbf_quality.models.training import train_model


def compare_models_for_scenario(
    df: pd.DataFrame,
    scenario: str,
    test_size: float = 0.2,
    random_state: int = 42,
    candidate_models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare candidate models for one production scenario.

    Business objective
    ------------------
    Decide whether more complex ML adds value over naive and statistical
    baselines using the metric that matches the scenario's business purpose.

    Coding objective
    ----------------
    Train each candidate and return a comparison table sorted by the scenario's
    primary and secondary selection metrics.
    """

    candidate_models = candidate_models or get_available_model_types()
    selection_policy = get_selection_policy(scenario)

    primary_metric = selection_policy["primary_metric"]
    secondary_metric = selection_policy["secondary_metric"]

    rows = []

    for model_type in candidate_models:
        result = train_model(
            df=df,
            scenario=scenario,
            model_type=model_type,
            test_size=test_size,
            random_state=random_state,
        )

        metrics = result["metrics"]
        model_description = result["metadata"]["model_description"]

        rows.append(
            {
                "scenario": scenario,
                "model_type": model_type,
                "model_family": model_description["model_family"],
                "complexity_level": model_description["complexity_level"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "risk_precision": metrics["risk_precision"],
                "risk_recall": metrics["risk_recall"],
                "risk_f1": metrics["risk_f1"],
                "risk_f2": metrics["risk_f2"],
                "selection_primary_metric": primary_metric,
                "selection_secondary_metric": secondary_metric,
                "business_selection_reason": selection_policy["business_reason"],
                "business_role": model_description["business_role"],
                "improvement_question": model_description["improvement_question"],
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        [primary_metric, secondary_metric],
        ascending=False,
    ).reset_index(drop=True)

    return comparison


def build_model_recommendation(
    comparison: pd.DataFrame,
) -> dict:
    """
    Build a deployment recommendation from model comparison results.

    Business objective
    ------------------
    Avoid deploying a model just because it is the best among weak candidates.

    A model should only move forward if it meaningfully improves over the naive
    baseline on the scenario's business metric.

    Coding objective
    ----------------
    Compare the best model against DummyClassifier using scenario-specific
    primary and secondary metrics.
    """

    if comparison.empty:
        raise ValueError("Comparison table is empty.")

    scenario = comparison.iloc[0]["scenario"]
    selection_policy = get_selection_policy(scenario)

    primary_metric = selection_policy["primary_metric"]
    secondary_metric = selection_policy["secondary_metric"]

    comparison = comparison.sort_values(
        [primary_metric, secondary_metric],
        ascending=False,
    ).reset_index(drop=True)

    best_row = comparison.iloc[0].to_dict()
    dummy_rows = comparison[comparison["model_type"] == "dummy"]

    if dummy_rows.empty:
        return {
            "scenario": scenario,
            "best_model_type": best_row["model_type"],
            "deployment_status": "REVIEW_REQUIRED",
            "recommendation": (
                "Dummy baseline was not included, so deployment readiness cannot "
                "be judged against a naive baseline."
            ),
        }

    dummy_row = dummy_rows.iloc[0]

    primary_lift = float(best_row[primary_metric]) - float(dummy_row[primary_metric])
    secondary_lift = float(best_row[secondary_metric]) - float(
        dummy_row[secondary_metric]
    )

    primary_threshold = selection_policy["minimum_primary_lift_over_dummy"]
    secondary_threshold = selection_policy["minimum_secondary_lift_over_dummy"]

    if primary_lift < primary_threshold or secondary_lift < secondary_threshold:
        deployment_status = "NOT_RECOMMENDED_FOR_DEPLOYMENT"
        recommendation = (
            "The best model does not improve enough over the naive baseline on "
            "the scenario-specific business metric. Treat this as a valid workflow "
            "prototype, not a trusted production quality predictor."
        )
    else:
        deployment_status = "CANDIDATE_FOR_ENGINEERING_VALIDATION"
        recommendation = (
            "The best model improves meaningfully over the naive baseline on the "
            "scenario-specific business metric. Continue with engineering "
            "validation on physically coherent production data before deployment."
        )

    return {
        "scenario": scenario,
        "best_model_type": best_row["model_type"],
        "best_model_family": best_row["model_family"],
        "selection_primary_metric": primary_metric,
        "selection_secondary_metric": secondary_metric,
        "best_primary_metric_value": float(best_row[primary_metric]),
        "best_secondary_metric_value": float(best_row[secondary_metric]),
        "dummy_primary_metric_value": float(dummy_row[primary_metric]),
        "dummy_secondary_metric_value": float(dummy_row[secondary_metric]),
        "primary_metric_lift_over_dummy": primary_lift,
        "secondary_metric_lift_over_dummy": secondary_lift,
        "deployment_status": deployment_status,
        "business_selection_reason": selection_policy["business_reason"],
        "recommendation": recommendation,
    }