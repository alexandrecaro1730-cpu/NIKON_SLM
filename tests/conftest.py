"""
Business objective
------------------
Provide reusable test fixtures so another developer can run the project tests
without needing the full production dataset.

Coding objective
----------------
Create a small synthetic LPBF-like dataset for contract and smoke tests.
"""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_lpbf_dataframe() -> pd.DataFrame:
    """
    Business objective
    ------------------
    Use a minimal but realistic table shape for production-readiness tests.

    Coding objective
    ----------------
    Return a small DataFrame with the columns expected by the project.
    """

    return pd.DataFrame(
        {
            "Sample_ID": [1, 2, 3, 4, 5, 6],
            "Alloy_Type": ["Ti-6Al-4V", "Ti-5553", "Near-Beta Ti", "Ti-6Al-4V", "Ti-5553", "Near-Beta Ti"],
            "Powder_Morphology": ["Spherical", "Irregular", "Spherical", "Irregular", "Spherical", "Irregular"],
            "Scan_Strategy": ["Stripe", "Island", "Chessboard", "Stripe", "Island", "Chessboard"],
            "Laser_Power_W": [200, 250, 300, 350, 220, 280],
            "Scan_Speed_mm_s": [800, 900, 1000, 1100, 850, 950],
            "Hatch_Spacing_mm": [0.09, 0.10, 0.11, 0.12, 0.09, 0.10],
            "Layer_Thickness_um": [30, 40, 50, 60, 30, 40],
            "Energy_Density_J_mm3": [92.6, 69.4, 54.5, 44.2, 95.8, 73.7],
            "Preheat_Temp_C": [100, 120, 150, 180, 110, 130],
            "Powder_Size_um": [25, 35, 45, 55, 30, 40],
            "Oxygen_Content_percent": [0.08, 0.10, 0.12, 0.15, 0.09, 0.11],
            "Melt_Pool_Temp_C": [1600, 1700, 1800, 1900, 1650, 1750],
            "Melt_Pool_Depth_um": [60, 80, 100, 120, 70, 90],
            "Cooling_Rate_K_s": [100000, 120000, 140000, 160000, 110000, 130000],
            "Thermal_Gradient": [20, 30, 40, 50, 25, 35],
            "Relative_Density_percent": [95.5, 96.5, 97.5, 98.5, 96.0, 97.0],
            "Porosity_percent": [4.5, 3.5, 2.5, 1.5, 4.0, 3.0],
            "Microhardness_HV": [320, 340, 360, 380, 330, 350],
            "Surface_Roughness_Ra": [8.0, 7.0, 6.0, 5.0, 7.5, 6.5],
            "Tensile_Strength_MPa": [900, 950, 1000, 1050, 925, 975],
            "Yield_Strength_MPa": [800, 850, 900, 950, 825, 875],
            "Elongation_percent": [8, 10, 12, 14, 9, 11],
            "Defect_Count": [10, 8, 5, 2, 9, 6],
            "Quality_Class": ["Poor", "Moderate", "Good", "Excellent", "Poor", "Good"],
        }
    )


@pytest.fixture()
def sample_csv_path(tmp_path: Path, sample_lpbf_dataframe: pd.DataFrame) -> Path:
    """
    Business objective
    ------------------
    Allow CLI tests to run from a temporary CSV file.

    Coding objective
    ----------------
    Save the sample DataFrame to disk and return its path.
    """

    path = tmp_path / "sample_lpbf.csv"
    sample_lpbf_dataframe.to_csv(path, index=False)
    return path