"""
Central project configuration.

This file defines the dataset contract used across the project.

Why this matters:
- In industrial ML projects, the model should not silently accept unexpected data.
- Explicit column groups make the pipeline easier to review, test, and maintain.
- Separating process features from post-build inspection features helps prevent
  data leakage in production.
"""

from pathlib import Path

# Resolve the project root so paths work regardless of where scripts are run from.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Standard project folders.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

MODELS_DIR = PROJECT_ROOT / "models"

# Prediction target.
TARGET_COLUMN = "Quality_Class"

# ID columns identify samples but should not be used for model training.
ID_COLUMNS = ["Sample_ID"]

# Categorical inputs available before or during production.
CATEGORICAL_FEATURES = [
    "Alloy_Type",
    "Powder_Morphology",
    "Scan_Strategy",
]

# Process and monitoring features that could realistically be available
# before or during a build.
PROCESS_FEATURES = [
    "Powder_Size_um",
    "Oxygen_Content_percent",
    "Laser_Power_W",
    "Scan_Speed_mm_s",
    "Hatch_Spacing_mm",
    "Layer_Thickness_um",
    "Preheat_Temp_C",
    "Shielding_Gas_Flow_L_min",
    "Melt_Pool_Width_um",
    "Melt_Pool_Depth_um",
    "Melt_Pool_Temp_C",
    "Cooling_Rate_K_s",
    "Energy_Density_J_mm3",
    "Thermal_Gradient",
]

# Post-build measurements are useful for analysis, but dangerous for production
# prediction if the goal is to predict quality before inspection is completed.
# Including these in training may create data leakage.
POST_BUILD_FEATURES = [
    "Relative_Density_percent",
    "Porosity_percent",
    "Microhardness_HV",
    "Surface_Roughness_Ra",
    "Tensile_Strength_MPa",
    "Yield_Strength_MPa",
    "Elongation_percent",
    "Defect_Count",
]

# Full expected schema.
EXPECTED_COLUMNS = (
    ID_COLUMNS
    + CATEGORICAL_FEATURES
    + PROCESS_FEATURES
    + POST_BUILD_FEATURES
    + [TARGET_COLUMN]
)

# Ordered quality classes.
# This is useful for reports, plots, and business communication.
QUALITY_ORDER = ["Poor", "Moderate", "Good", "Excellent"]