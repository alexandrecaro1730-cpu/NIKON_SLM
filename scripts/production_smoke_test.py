"""
Production smoke test for the LPBF quality prediction service.

Business objective
------------------
Verify that the production-oriented API can be loaded and can return a valid
prediction response.

This is not a full model-performance test. It checks the operational path:
model artifact -> API schema -> prediction -> decision response.

Coding objective
----------------
1. Ensure the selected in-build model artifact exists.
2. Train it if missing.
3. Use FastAPI TestClient to call /, /health, /model-info, and /predict.
4. Fail with a non-zero exit code if the service contract breaks.
"""

from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient

from lpbf_quality.api.main import DEFAULT_MODEL_PATH, app


DATA_PATH = Path("data/raw/lpbf_titanium_dataset.csv")


def ensure_model_artifact_exists() -> None:
    """
    Train the selected Task 2 MVP model if the artifact is missing.

    Business objective
    ------------------
    Make the smoke test runnable from a fresh clone after dependencies and data
    are available.

    Coding objective
    ----------------
    Use scripts/train.py instead of duplicating training logic here.
    """

    if DEFAULT_MODEL_PATH.exists():
        return

    command = [
        sys.executable,
        "scripts/train.py",
        "--input",
        str(DATA_PATH),
        "--scenario",
        "inbuild",
        "--model",
        "logistic_regression",
    ]

    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            "Could not create required model artifact for production smoke test."
        )


def sample_inbuild_payload() -> dict:
    """
    Build a realistic in-build request payload.

    The values are representative example values for API validation and service
    smoke testing. They are not a production recommendation.
    """

    return {
        "Alloy_Type": "Ti-6Al-4V",
        "Powder_Morphology": "Spherical",
        "Scan_Strategy": "Stripe",
        "Powder_Size_um": 35.0,
        "Oxygen_Content_percent": 0.12,
        "Laser_Power_W": 280.0,
        "Scan_Speed_mm_s": 950.0,
        "Hatch_Spacing_mm": 0.11,
        "Layer_Thickness_um": 30.0,
        "Preheat_Temp_C": 180.0,
        "Shielding_Gas_Flow_L_min": 28.0,
        "Melt_Pool_Width_um": 115.0,
        "Melt_Pool_Depth_um": 55.0,
        "Melt_Pool_Temp_C": 1850.0,
        "Cooling_Rate_K_s": 850000.0,
        "Energy_Density_J_mm3": 75.0,
        "Thermal_Gradient": 12000.0,
    }


def main() -> None:
    ensure_model_artifact_exists()

    client = TestClient(app)

    root_response = client.get("/")
    assert root_response.status_code == 200
    assert root_response.json()["status"] == "ok"

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    model_info_response = client.get("/model-info")
    assert model_info_response.status_code == 200
    assert model_info_response.json()["scenario"] == "inbuild"

    prediction_response = client.post(
        "/predict",
        json=sample_inbuild_payload(),
    )

    assert prediction_response.status_code == 200

    prediction = prediction_response.json()

    required_response_keys = {
        "scenario",
        "model_type",
        "model_status",
        "predicted_quality_class",
        "risk_group",
        "risk_flag",
        "recommended_action",
        "limitation",
    }

    missing_keys = required_response_keys - set(prediction)

    assert not missing_keys, (
        f"Prediction response is missing required keys: {missing_keys}"
    )

    print("Production smoke test passed.")
    print(prediction)


if __name__ == "__main__":
    main()