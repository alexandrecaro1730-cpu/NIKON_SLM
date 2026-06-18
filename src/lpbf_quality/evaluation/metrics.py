"""
Evaluation metrics for LPBF quality classification.

Business objective
------------------
Evaluate model performance in a way that matches industrial quality decisions.

Accuracy alone is not enough because:
- a model can get acceptable headline accuracy by predicting the majority class
- missing Poor or Moderate builds can be operationally expensive
- production users need to know whether risky builds are detected

This module therefore reports both:

1. General multi-class metrics
   - accuracy
   - balanced accuracy
   - macro F1
   - weighted F1
   - per-class precision / recall / F1
   - confusion matrix

2. Business risk metrics
   - risk precision
   - risk recall
   - risk F1
   - risk F2

Risk classes are:
- Poor
- Moderate

Acceptable classes are:
- Good
- Excellent

Coding objective
----------------
Compute JSON-serializable metrics that can be saved as model artifacts and used
for scenario-specific model selection.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from lpbf_quality.config.settings import QUALITY_ORDER


RISK_CLASSES = ["Poor", "Moderate"]


def _to_risk_binary(labels) -> list[int]:
    """
    Convert quality labels into binary risk labels.

    Business objective
    ------------------
    For operational decisions, the main question is often not the exact class.
    The main question is whether the build is risky enough to trigger attention.

    Risk = Poor or Moderate
    Non-risk = Good or Excellent

    Coding objective
    ----------------
    Return 1 for risky classes and 0 for acceptable classes.
    """

    return [1 if label in RISK_CLASSES else 0 for label in labels]


def evaluate_risk_detection(y_true, y_pred) -> dict:
    """
    Evaluate binary risk detection.

    Business objective
    ------------------
    Measure whether the model catches Poor and Moderate builds.

    This is especially important for:
    - pre-build risk screening
    - in-build risk update

    In those stages, missing risky builds is usually worse than raising extra
    operator warnings.

    Coding objective
    ----------------
    Convert multi-class labels into binary risk labels and compute precision,
    recall, F1, and F2.
    """

    y_true_risk = _to_risk_binary(y_true)
    y_pred_risk = _to_risk_binary(y_pred)

    return {
        "risk_precision": float(
            precision_score(y_true_risk, y_pred_risk, zero_division=0)
        ),
        "risk_recall": float(
            recall_score(y_true_risk, y_pred_risk, zero_division=0)
        ),
        "risk_f1": float(
            f1_score(y_true_risk, y_pred_risk, zero_division=0)
        ),
        "risk_f2": float(
            fbeta_score(y_true_risk, y_pred_risk, beta=2, zero_division=0)
        ),
        "risk_metric_interpretation": {
            "risk_precision": (
                "Of builds flagged as risky, how many were actually Poor or "
                "Moderate?"
            ),
            "risk_recall": (
                "Of all real Poor or Moderate builds, how many did the model "
                "catch?"
            ),
            "risk_f1": (
                "Balanced score between risk precision and risk recall."
            ),
            "risk_f2": (
                "Risk score that weights recall more heavily than precision. "
                "Useful when missing risky builds is more costly than false alarms."
            ),
        },
    }


def evaluate_classification(
    y_true,
    y_pred,
    labels: list[str] | None = None,
) -> dict:
    """
    Evaluate a multi-class LPBF quality classifier.

    Business objective
    ------------------
    Measure both:
    - general classification quality across all four classes
    - operational risk detection for Poor and Moderate builds

    Coding objective
    ----------------
    Return a JSON-serializable dictionary that can be saved as a reproducible
    metrics artifact.
    """

    labels = labels or QUALITY_ORDER

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "metric_interpretation": {
            "accuracy": (
                "Overall correctness. Useful but can hide poor performance on "
                "minority or high-risk classes."
            ),
            "balanced_accuracy": (
                "Average recall across classes. Useful when quality classes are "
                "imbalanced or when each class matters operationally."
            ),
            "macro_f1": (
                "Average F1 across classes without weighting by class frequency. "
                "Useful to detect whether the model ignores smaller classes."
            ),
            "weighted_f1": (
                "F1 weighted by class frequency. Useful as a stable overall score, "
                "but it may still hide weak minority-class performance."
            ),
            "confusion_matrix": (
                "Shows which quality classes are confused. This is important for "
                "understanding whether risky builds are missed."
            ),
        },
    }

    metrics.update(evaluate_risk_detection(y_true, y_pred))

    return metrics


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    """
    Save metrics as a JSON artifact.

    Business objective
    ------------------
    Make model performance auditable and easy to include in documentation,
    PowerPoint, and production review.

    Coding objective
    ----------------
    Create the output directory if needed and write formatted JSON.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_confusion_matrix_csv(metrics: dict, output_path: Path) -> None:
    """
    Save the confusion matrix as a CSV artifact.

    Business objective
    ------------------
    Provide a simple table that quality engineers can read without loading
    Python or JSON.

    Coding objective
    ----------------
    Convert the confusion matrix from the metrics dictionary into a labeled CSV.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]

    cm_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    cm_df.to_csv(output_path)