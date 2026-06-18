"""
Model registry for the LPBF quality intelligence system.

Business objective
------------------
Define a controlled set of model candidates that represent increasing modeling
complexity.

The goal is not to try every algorithm. The goal is to show a disciplined
data-science comparison:

1. DummyClassifier
   Question:
   "Can we beat naive majority-class guessing?"

2. LogisticRegression
   Question:
   "Can a simple explainable statistical model detect useful signal?"

3. RandomForestClassifier
   Question:
   "Do nonlinear feature interactions improve prediction?"

4. HistGradientBoostingClassifier
   Question:
   "If future production data has stronger signal, does a stronger tabular
   boosting model provide a direction for improvement?"

Coding objective
----------------
Provide reusable model constructors and business metadata for training,
comparison, model selection, and saved artifacts.
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


MODEL_REGISTRY = {
    "dummy": {
        "model_family": "naive_baseline",
        "complexity_level": 0,
        "business_role": (
            "Minimum benchmark. If trained models cannot beat this, the dataset "
            "does not support useful prediction."
        ),
        "improvement_question": "Can any model beat naive majority-class guessing?",
    },
    "logistic_regression": {
        "model_family": "statistical_baseline",
        "complexity_level": 1,
        "business_role": (
            "Explainable statistical baseline for multi-class quality prediction."
        ),
        "improvement_question": (
            "Can a simple linear decision boundary detect meaningful signal?"
        ),
    },
    "random_forest": {
        "model_family": "nonlinear_bagging_ml",
        "complexity_level": 2,
        "business_role": (
            "Nonlinear tabular ML baseline that can capture feature interactions."
        ),
        "improvement_question": (
            "Do nonlinear process interactions improve prediction over the "
            "statistical baseline?"
        ),
    },
    "hist_gradient_boosting": {
        "model_family": "boosting_tabular_ml",
        "complexity_level": 3,
        "business_role": (
            "Stronger tabular model for future higher-quality production data. "
            "Used to test whether boosting provides a direction for improvement."
        ),
        "improvement_question": (
            "If better data is collected, does sequential boosting improve over "
            "Random Forest and Logistic Regression?"
        ),
    },
}


def get_available_model_types() -> list[str]:
    """
    Return supported model identifiers.

    Business objective
    ------------------
    Keep the modeling comparison transparent and controlled.

    Coding objective
    ----------------
    Return model names supported by build_classifier().
    """

    return list(MODEL_REGISTRY.keys())


def get_model_description(model_type: str) -> dict:
    """
    Return business metadata for a model candidate.

    Business objective
    ------------------
    Explain why each model exists in the comparison and how it supports the
    production-quality case study.

    Coding objective
    ----------------
    Return metadata for reports and saved model artifacts.
    """

    if model_type not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_type '{model_type}'. Valid options: {valid}")

    return MODEL_REGISTRY[model_type]


def build_classifier(model_type: str, random_state: int = 42):
    """
    Build a model candidate.

    Business objective
    ------------------
    Compare increasing model complexity without overengineering the proof of
    concept.

    Coding objective
    ----------------
    Return an initialized scikit-learn classifier.
    """

    if model_type == "dummy":
        return DummyClassifier(strategy="most_frequent")

    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.1,
            random_state=random_state,
        )

    valid = ", ".join(sorted(MODEL_REGISTRY))
    raise ValueError(f"Unknown model_type '{model_type}'. Valid options: {valid}")