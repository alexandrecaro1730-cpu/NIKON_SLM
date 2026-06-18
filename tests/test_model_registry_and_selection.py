"""
Tests for model registry and model-selection logic.

Business objective
------------------
Verify that the project compares model complexity in a controlled and
explainable way.

The model-selection policy should reflect production priorities:
- pre-build and in-build models prioritize risk detection
- post-build diagnostic models prioritize balanced multi-class performance

Coding objective
----------------
Check that:
- expected model candidates exist
- every model can be constructed
- model recommendation logic returns the correct deployment status
"""

import pandas as pd

from lpbf_quality.models.model_registry import (
    build_classifier,
    get_available_model_types,
    get_model_description,
)
from lpbf_quality.models.model_selection import build_model_recommendation


def test_model_registry_contains_expected_complexity_ladder():
    """
    Business objective
    ------------------
    Confirm that the modeling workflow includes naive, statistical, nonlinear,
    and boosting candidates.
    """

    model_types = get_available_model_types()

    assert "dummy" in model_types
    assert "logistic_regression" in model_types
    assert "random_forest" in model_types
    assert "hist_gradient_boosting" in model_types


def test_model_registry_builds_all_models():
    """
    Business objective
    ------------------
    Confirm that every registered model candidate is usable by the training
    pipeline.
    """

    for model_type in get_available_model_types():
        model = build_classifier(model_type=model_type, random_state=42)
        description = get_model_description(model_type)

        assert model is not None
        assert "model_family" in description
        assert "business_role" in description
        assert "complexity_level" in description


def test_model_recommendation_marks_weak_lift_as_not_recommended():
    """
    Business objective
    ------------------
    Confirm that the system does not recommend deployment when the best model
    barely improves over the naive baseline on the scenario-specific business
    metric.

    For the in-build scenario, model selection uses:
    - primary metric: risk_f2
    - secondary metric: risk_recall
    """

    comparison = pd.DataFrame(
        [
            {
                "scenario": "inbuild",
                "model_type": "dummy",
                "model_family": "naive_baseline",
                "balanced_accuracy": 0.25,
                "macro_f1": 0.20,
                "risk_f2": 0.40,
                "risk_recall": 0.40,
            },
            {
                "scenario": "inbuild",
                "model_type": "random_forest",
                "model_family": "nonlinear_bagging_ml",
                "balanced_accuracy": 0.27,
                "macro_f1": 0.22,
                "risk_f2": 0.42,
                "risk_recall": 0.43,
            },
        ]
    ).sort_values(
        ["risk_f2", "risk_recall"],
        ascending=False,
    )

    recommendation = build_model_recommendation(comparison)

    assert recommendation["deployment_status"] == "NOT_RECOMMENDED_FOR_DEPLOYMENT"
    assert recommendation["selection_primary_metric"] == "risk_f2"
    assert recommendation["selection_secondary_metric"] == "risk_recall"


def test_model_recommendation_allows_engineering_validation_when_lift_is_meaningful():
    """
    Business objective
    ------------------
    Confirm that the system can identify a model as a candidate for further
    engineering validation when it clearly improves over the naive baseline on
    the scenario-specific business metric.

    For the in-build scenario, this means clear improvement in risk_f2 and
    risk_recall.
    """

    comparison = pd.DataFrame(
        [
            {
                "scenario": "inbuild",
                "model_type": "dummy",
                "model_family": "naive_baseline",
                "balanced_accuracy": 0.25,
                "macro_f1": 0.20,
                "risk_f2": 0.40,
                "risk_recall": 0.40,
            },
            {
                "scenario": "inbuild",
                "model_type": "hist_gradient_boosting",
                "model_family": "boosting_tabular_ml",
                "balanced_accuracy": 0.40,
                "macro_f1": 0.38,
                "risk_f2": 0.55,
                "risk_recall": 0.58,
            },
        ]
    ).sort_values(
        ["risk_f2", "risk_recall"],
        ascending=False,
    )

    recommendation = build_model_recommendation(comparison)

    assert (
        recommendation["deployment_status"]
        == "CANDIDATE_FOR_ENGINEERING_VALIDATION"
    )
    assert recommendation["selection_primary_metric"] == "risk_f2"
    assert recommendation["selection_secondary_metric"] == "risk_recall"