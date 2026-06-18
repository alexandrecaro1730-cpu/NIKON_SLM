"""
Tests for the Task 3 production-oriented API contract.

Business objective
------------------
Verify that the LPBF quality prediction service exposes the minimum production
service contract required for Task 3:

- service health
- model metadata
- validated in-build prediction input
- production-friendly prediction response

Coding objective
----------------
Use FastAPI TestClient to test the API without starting a real server process.

CI objective
------------
GitHub Actions starts from a clean repository checkout. Generated model
artifacts are intentionally not committed to Git, so the tests that call
/model-info and /predict must create the selected in-build model artifact when
it is missing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from lpbf_quality.api.main import app
from scripts.production_smoke_test import sample_inbuild_payload


client = TestClient(app)


def ensure_inbuild_model_artifact() -> None:
    """
    Ensure the selected Task 2 in-build model artifact exists.

    Business objective
    ------------------
    The production API is expected to load a trained model artifact. Because
    generated artifacts are reproducible and normally not committed, this
    helper makes the API tests reproducible in fresh environments such as
    GitHub Actions.

    Coding objective
    ----------------
    Train the selected in-build Logistic Regression model only when the
    expected artifact is missing.
    """

    model_path = Path("models/inbuild/logistic_regression.joblib")

    if model_path.exists():
        return

    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--input",
            "data/raw/lpbf_titanium_dataset.csv",
            "--scenario",
            "inbuild",
            "--model",
            "logistic_regression",
        ],
        check=True,
    )


def test_health_endpoint_returns_service_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["service"] == "lpbf-quality-prediction"


def test_root_endpoint_returns_navigation_links() -> None:
    response = client.get("/")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["docs"] == "/docs"
    assert payload["health"] == "/health"
    assert payload["model_info"] == "/model-info"
    assert payload["predict"] == "/predict"


def test_model_info_endpoint_exposes_inbuild_contract() -> None:
    ensure_inbuild_model_artifact()

    response = client.get("/model-info")

    assert response.status_code == 200

    payload = response.json()

    assert payload["scenario"] == "inbuild"
    assert payload["model_type"] == "logistic_regression"
    assert payload["model_status"] == "candidate_for_engineering_validation"

    features = payload["features"]

    assert "Melt_Pool_Width_um" in features
    assert "Melt_Pool_Depth_um" in features
    assert "Thermal_Gradient" in features

    assert "Porosity_percent" not in features
    assert "Defect_Count" not in features
    assert "Quality_Class" not in features
    assert "Sample_ID" not in features


def test_predict_endpoint_returns_operational_decision_response() -> None:
    ensure_inbuild_model_artifact()

    response = client.post(
        "/predict",
        json=sample_inbuild_payload(),
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["scenario"] == "inbuild"
    assert payload["model_type"] == "logistic_regression"
    assert payload["model_status"] == "candidate_for_engineering_validation"

    assert payload["predicted_quality_class"] in {
        "Poor",
        "Moderate",
        "Good",
        "Excellent",
    }

    assert payload["risk_group"] in {"Risk", "Acceptable"}
    assert isinstance(payload["risk_flag"], bool)

    assert "recommended_action" in payload
    assert payload["recommended_action"]

    assert "limitation" in payload
    assert "proof-of-concept" in payload["limitation"]


def test_predict_endpoint_rejects_missing_required_feature() -> None:
    payload = sample_inbuild_payload()
    payload.pop("Melt_Pool_Width_um")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422