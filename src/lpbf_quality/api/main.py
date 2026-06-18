"""
FastAPI application for the LPBF quality prediction service.

Business objective
------------------
Demonstrate how the Task 2 in-build quality-risk model could be used inside
an industrial production environment.

The model is intentionally exposed as a decision-support service, not as an
autonomous production controller. The API returns quality-risk information that
could be consumed by an MES, operator dashboard, quality system, or monitoring
pipeline.

Coding objective
----------------
Provide a small, runnable FastAPI service with:

- root navigation endpoint
- health check endpoint
- model metadata endpoint
- single-sample prediction endpoint
- strict in-build feature validation
- risk-group decision logic
- transparent production limitations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lpbf_quality.config.settings import PROJECT_ROOT
from lpbf_quality.features.feature_sets import get_feature_set


# ---------------------------------------------------------------------------
# Production configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "inbuild" / "logistic_regression.joblib"
)

RISK_CLASSES = {"Poor", "Moderate"}

MODEL_STATUS = "candidate_for_engineering_validation"

PRODUCTION_LIMITATION = (
    "This service is a proof-of-concept decision-support API. Task 1 showed "
    "that the dataset is technically clean but does not strongly preserve "
    "expected LPBF physics relationships. The model must be validated on "
    "physically coherent production data before trusted deployment."
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class InBuildPredictionRequest(BaseModel):
    """
    Input schema for the in-build operational prediction scenario.

    Business objective
    ------------------
    Accept only features that are available before or during the build.

    This intentionally excludes post-build inspection and mechanical-test
    features such as porosity, density, microhardness, strength, elongation,
    surface roughness, and defect count.

    Coding objective
    ----------------
    Pydantic validates types and makes the API contract explicit.

    Implementation note
    -------------------
    The examples use json_schema_extra instead of the deprecated Pydantic v1
    Field(..., example=...) style. This keeps the OpenAPI documentation useful
    while avoiding Pydantic v2 deprecation warnings.
    """

    Alloy_Type: str = Field(
        ...,
        json_schema_extra={"example": "Ti-6Al-4V"},
    )
    Powder_Morphology: str = Field(
        ...,
        json_schema_extra={"example": "Spherical"},
    )
    Scan_Strategy: str = Field(
        ...,
        json_schema_extra={"example": "Stripe"},
    )

    Powder_Size_um: float = Field(
        ...,
        json_schema_extra={"example": 35.0},
    )
    Oxygen_Content_percent: float = Field(
        ...,
        json_schema_extra={"example": 0.12},
    )

    Laser_Power_W: float = Field(
        ...,
        json_schema_extra={"example": 280.0},
    )
    Scan_Speed_mm_s: float = Field(
        ...,
        json_schema_extra={"example": 950.0},
    )
    Hatch_Spacing_mm: float = Field(
        ...,
        json_schema_extra={"example": 0.11},
    )
    Layer_Thickness_um: float = Field(
        ...,
        json_schema_extra={"example": 30.0},
    )
    Preheat_Temp_C: float = Field(
        ...,
        json_schema_extra={"example": 180.0},
    )
    Shielding_Gas_Flow_L_min: float = Field(
        ...,
        json_schema_extra={"example": 28.0},
    )

    Melt_Pool_Width_um: float = Field(
        ...,
        json_schema_extra={"example": 115.0},
    )
    Melt_Pool_Depth_um: float = Field(
        ...,
        json_schema_extra={"example": 55.0},
    )
    Melt_Pool_Temp_C: float = Field(
        ...,
        json_schema_extra={"example": 1850.0},
    )
    Cooling_Rate_K_s: float = Field(
        ...,
        json_schema_extra={"example": 850000.0},
    )
    Energy_Density_J_mm3: float = Field(
        ...,
        json_schema_extra={"example": 75.0},
    )
    Thermal_Gradient: float = Field(
        ...,
        json_schema_extra={"example": 12000.0},
    )


class PredictionResponse(BaseModel):
    """
    Output schema for a quality-risk prediction.

    Business objective
    ------------------
    Return a production-friendly decision object rather than only a raw ML
    class label.
    """

    scenario: str
    model_type: str
    model_status: str
    predicted_quality_class: str
    risk_group: str
    risk_flag: bool
    risk_probability: float | None
    class_probabilities: dict[str, float] | None
    recommended_action: str
    limitation: str


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model_artifact: dict[str, Any] | None = None


def load_model_artifact() -> dict[str, Any]:
    """
    Load the trained model artifact once and cache it.

    Business objective
    ------------------
    Production services should load a fixed, versioned model artifact and use
    it consistently for inference.

    Coding objective
    ----------------
    Load the joblib artifact generated by scripts/train.py.
    """

    global _model_artifact

    if _model_artifact is not None:
        return _model_artifact

    if not DEFAULT_MODEL_PATH.exists():
        raise RuntimeError(
            "Model artifact not found. Train the selected Task 2 model first: "
            "python scripts/train.py --input data/raw/lpbf_titanium_dataset.csv "
            "--scenario inbuild --model logistic_regression"
        )

    artifact = joblib.load(DEFAULT_MODEL_PATH)

    if "pipeline" not in artifact:
        raise RuntimeError(
            f"Invalid model artifact at {DEFAULT_MODEL_PATH}. "
            "Expected a dictionary containing a 'pipeline' key."
        )

    _model_artifact = artifact
    return artifact


def pydantic_to_dict(model: BaseModel) -> dict[str, Any]:
    """
    Convert a Pydantic model to a dictionary.

    This helper supports both Pydantic v1 and v2.
    """

    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def build_prediction_frame(request: InBuildPredictionRequest) -> pd.DataFrame:
    """
    Convert a validated API request into a one-row pandas DataFrame.

    Business objective
    ------------------
    Ensure the inference-time feature order matches the in-build model contract.

    Coding objective
    ----------------
    Use get_feature_set("inbuild") as the single source of truth.
    """

    payload = pydantic_to_dict(request)
    feature_columns = get_feature_set("inbuild")

    missing_features = [
        feature for feature in feature_columns if feature not in payload
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required in-build features.",
                "missing_features": missing_features,
            },
        )

    return pd.DataFrame(
        [{feature: payload[feature] for feature in feature_columns}]
    )


def get_recommended_action(
    predicted_quality_class: str,
    risk_probability: float | None,
) -> str:
    """
    Convert model output into a production-oriented recommendation.

    Business objective
    ------------------
    The model should support human decision-making, not silently control the
    LPBF machine.

    Coding objective
    ----------------
    Return a clear action string based on risk class and probability.
    """

    if predicted_quality_class in RISK_CLASSES:
        return (
            "Escalate for operator review, parameter check, or additional "
            "inspection planning."
        )

    if risk_probability is not None and risk_probability >= 0.40:
        return (
            "Continue build, but flag for closer monitoring because the risk "
            "probability is elevated."
        )

    return "Continue normal production monitoring."


def predict_quality(request: InBuildPredictionRequest) -> PredictionResponse:
    """
    Run one in-build quality prediction.
    """

    artifact = load_model_artifact()
    pipeline = artifact["pipeline"]
    metadata = artifact.get("metadata", {})

    X = build_prediction_frame(request)

    predicted_quality_class = str(pipeline.predict(X)[0])

    class_probabilities = None
    risk_probability = None

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[0]
        classes = [str(label) for label in pipeline.classes_]

        class_probabilities = {
            class_name: float(probability)
            for class_name, probability in zip(classes, probabilities)
        }

        risk_probability = float(
            sum(
                probability
                for class_name, probability in class_probabilities.items()
                if class_name in RISK_CLASSES
            )
        )

    risk_flag = predicted_quality_class in RISK_CLASSES
    risk_group = "Risk" if risk_flag else "Acceptable"

    return PredictionResponse(
        scenario="inbuild",
        model_type=str(metadata.get("model_type", "logistic_regression")),
        model_status=MODEL_STATUS,
        predicted_quality_class=predicted_quality_class,
        risk_group=risk_group,
        risk_flag=risk_flag,
        risk_probability=risk_probability,
        class_probabilities=class_probabilities,
        recommended_action=get_recommended_action(
            predicted_quality_class=predicted_quality_class,
            risk_probability=risk_probability,
        ),
        limitation=PRODUCTION_LIMITATION,
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LPBF Quality Prediction Service",
    version="0.1.0",
    description=(
        "Proof-of-concept in-build quality-risk API for LPBF titanium "
        "production monitoring."
    ),
)


@app.get("/")
def root() -> dict[str, str]:
    """
    Root endpoint with quick navigation hints.

    Business value
    --------------
    Helps reviewers find the API documentation and operational endpoints.
    """

    return {
        "service": "lpbf-quality-prediction",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """
    Health check endpoint.

    Business value
    --------------
    Allows deployment tools, CI, or monitoring systems to check whether the
    service process is alive.
    """

    return {
        "status": "ok",
        "service": "lpbf-quality-prediction",
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """
    Return model metadata and intended-use information.

    Business value
    --------------
    Makes model assumptions visible to operators, reviewers, and downstream
    systems.
    """

    try:
        artifact = load_model_artifact()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    metadata = artifact.get("metadata", {})

    return {
        "model_path": str(DEFAULT_MODEL_PATH),
        "scenario": metadata.get("scenario", "inbuild"),
        "model_type": metadata.get("model_type", "logistic_regression"),
        "model_status": MODEL_STATUS,
        "features": metadata.get("features", get_feature_set("inbuild")),
        "intended_use": (
            "In-build decision support for identifying Poor/Moderate quality "
            "risk while the build is still active or before final inspection."
        ),
        "not_intended_for": (
            "Autonomous machine control or final quality release without "
            "engineering validation."
        ),
        "limitation": PRODUCTION_LIMITATION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: InBuildPredictionRequest) -> PredictionResponse:
    """
    Predict the quality class and risk group for one in-build sample.
    """

    try:
        return predict_quality(request)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error