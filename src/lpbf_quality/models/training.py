"""
Training utilities for the LPBF quality intelligence system.

Business objective
------------------
Build a practical, leakage-safe modeling workflow that demonstrates sound
engineering judgment.

The purpose is not to overclaim that the current dataset proves a physically
validated LPBF quality predictor. Task 1 showed limitations in the data signal
and physical realism.

Therefore, this module focuses on:

1. Decision-time feature separation
2. Baseline comparison
3. Statistical baseline comparison
4. Nonlinear machine-learning comparison
5. Stronger tabular boosting comparison
6. Class-sensitive evaluation
7. Saved artifacts for reproducibility
8. Honest metadata explaining intended use and limitations

Coding objective
----------------
Provide reusable functions to:
- build preprocessing
- train one model for one production scenario
- evaluate the model
- save model artifacts with business metadata
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lpbf_quality.config.settings import CATEGORICAL_FEATURES, QUALITY_ORDER
from lpbf_quality.evaluation.metrics import evaluate_classification
from lpbf_quality.features.build_features import build_model_frame
from lpbf_quality.features.feature_sets import (
    get_feature_set,
    get_scenario_description,
)
from lpbf_quality.models.model_registry import (
    build_classifier,
    get_model_description,
)


def _build_one_hot_encoder() -> OneHotEncoder:
    """
    Build a OneHotEncoder compatible with multiple scikit-learn versions.

    Business objective
    ------------------
    Keep the project easy to run in different reviewer environments.

    Coding objective
    ----------------
    Newer scikit-learn versions use sparse_output. Older versions use sparse.
    This helper supports both.
    """

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def model_needs_scaling(model_type: str) -> bool:
    """
    Decide whether numeric scaling should be included.

    Business objective
    ------------------
    Use preprocessing that matches the model family.

    Logistic Regression benefits from scaled numeric features. Tree-based
    models such as Random Forest and histogram gradient boosting do not require
    scaling.

    Coding objective
    ----------------
    Return True only for model families that benefit from StandardScaler.
    """

    return model_type == "logistic_regression"


def build_preprocessor(
    X: pd.DataFrame,
    model_type: str,
) -> ColumnTransformer:
    """
    Build preprocessing for mixed LPBF tabular data.

    Business objective
    ------------------
    Make the model robust to missing production values and categorical process
    settings while keeping preprocessing simple and explainable.

    Coding objective
    ----------------
    Numeric columns:
    - median imputation
    - optional scaling for Logistic Regression

    Categorical columns:
    - most-frequent imputation
    - one-hot encoding
    """

    categorical_columns = [
        column for column in CATEGORICAL_FEATURES if column in X.columns
    ]

    numeric_columns = [
        column for column in X.columns if column not in categorical_columns
    ]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if model_needs_scaling(model_type):
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _build_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_training_pipeline(
    X: pd.DataFrame,
    model_type: str,
    random_state: int = 42,
) -> Pipeline:
    """
    Build the full preprocessing + classifier pipeline.

    Business objective
    ------------------
    Save preprocessing and the trained model together so the same transformations
    are used in training, evaluation, and future production inference.

    Coding objective
    ----------------
    Return a scikit-learn Pipeline that can be fitted, evaluated, and saved.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X=X, model_type=model_type)),
            ("classifier", build_classifier(model_type, random_state=random_state)),
        ]
    )


def split_model_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data into train and test sets.

    Business objective
    ------------------
    Estimate model performance on unseen data while preserving class balance
    when possible.

    Coding objective
    ----------------
    Use stratification if every class has enough samples. If the dataset is too
    small for stratification, fall back to a regular split instead of failing
    unexpectedly.
    """

    class_counts = y.value_counts()
    can_stratify = len(class_counts) > 1 and class_counts.min() >= 2

    stratify = y if can_stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def build_model_metadata(
    scenario: str,
    model_type: str,
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> dict:
    """
    Build metadata saved with every model artifact.

    Business objective
    ------------------
    Make every trained model self-documenting:
    - What question does it answer?
    - When can it be used?
    - Which features were allowed?
    - What model family was tested?
    - Is it an operational model or only diagnostic?
    - What are the known dataset limitations?

    Coding objective
    ----------------
    Return a JSON/joblib-safe dictionary.
    """

    scenario_description = get_scenario_description(scenario)
    model_description = get_model_description(model_type)

    return {
        "scenario": scenario,
        "model_type": model_type,
        "model_description": model_description,
        "target": "Quality_Class",
        "features": get_feature_set(scenario),
        "scenario_description": scenario_description,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(X_train)),
        "n_rows_test": int(len(X_test)),
        "test_size": test_size,
        "random_state": random_state,
        "known_dataset_limitation": (
            "Task 1 showed that the dataset is technically clean and suitable "
            "for demonstrating a production ML workflow, but it does not strongly "
            "preserve expected LPBF physics or predictive signal. Therefore, this "
            "model should not be deployed as a trusted quality predictor until "
            "validated on physically coherent production data."
        ),
        "core_modeling_principle": (
            "Use only what is known at the decision time. Pre-build models use "
            "pre-build features only; in-build models use pre-build plus monitoring "
            "features; post-build models may use inspection/test features only for "
            "diagnosis."
        ),
    }


def train_model(
    df: pd.DataFrame,
    scenario: str,
    model_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train and evaluate one scenario/model combination.

    Business objective
    ------------------
    Train a model for one clearly defined production use case.

    This avoids the common mistake of building one high-scoring offline model
    that uses columns unavailable at the time of the real decision.

    Coding objective
    ----------------
    1. Build leakage-safe X/y for the scenario.
    2. Split data into train/test sets.
    3. Fit preprocessing + model pipeline.
    4. Predict on holdout data.
    5. Compute class-sensitive metrics.
    6. Attach business metadata.
    """

    X, y = build_model_frame(df=df, scenario=scenario)

    X_train, X_test, y_train, y_test = split_model_data(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_training_pipeline(
        X=X_train,
        model_type=model_type,
        random_state=random_state,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = evaluate_classification(
        y_true=y_test,
        y_pred=y_pred,
        labels=QUALITY_ORDER,
    )

    metadata = build_model_metadata(
        scenario=scenario,
        model_type=model_type,
        df=df,
        X_train=X_train,
        X_test=X_test,
        test_size=test_size,
        random_state=random_state,
    )

    metrics["metadata"] = metadata

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "metadata": metadata,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def save_model_artifact(
    pipeline: Pipeline,
    metadata: dict,
    output_path: Path,
) -> None:
    """
    Save a trained model artifact.

    Business objective
    ------------------
    Store the complete inference pipeline and business metadata so the model can
    be reviewed, reused, and connected to a future API/service.

    Coding objective
    ----------------
    Save a joblib file containing:
    - preprocessing + classifier pipeline
    - metadata explaining scenario, features, intended use, and limitations
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
    }

    joblib.dump(artifact, output_path)