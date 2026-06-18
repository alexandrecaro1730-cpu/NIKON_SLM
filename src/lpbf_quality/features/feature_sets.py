"""
Scenario-based feature definitions for the LPBF quality intelligence system.

Business objective
------------------
The goal is not to build one offline model that uses every available column.

In a real LPBF production environment, different decisions happen at different
times:

1. Pre-build risk screening
   Question:
   "Before starting the build, does this job look risky?"

   Business value:
   Prevent avoidable waste of machine time, powder, and post-processing cost.

2. In-build risk update
   Question:
   "During the build, is quality risk increasing?"

   Business value:
   Trigger operator alerts, inspection escalation, pause/abort decisions, or
   parameter review before the full batch is lost.

   This is the recommended production MVP because it balances actionability
   and available signal.

3. Post-build diagnostic benchmark
   Question:
   "After inspection and testing, what patterns explain the quality outcome?"

   Business value:
   Support root-cause analysis, engineering review, process learning, powder
   supplier investigation, and machine investigation.

   This is not the operational prediction model because post-build inspection
   and mechanical-test features are not available at the decision time.

4. Production monitoring / drift detection
   Question:
   "Is the machine, powder, sensor, or process window degrading over time?"

   Business value:
   Detect process degradation even when individual sample-level predictions
   are uncertain.

Coding objective
----------------
Define leakage-safe feature views that enforce the principle:

    Use only what is known at the decision time.

This prevents misleading offline performance caused by training an operational
model on post-build inspection results.
"""

from lpbf_quality.config.settings import (
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    POST_BUILD_FEATURES,
    PROCESS_FEATURES,
    TARGET_COLUMN,
)

# ---------------------------------------------------------------------------
# Features known before the build starts.
#
# Business objective:
# These features support early risk screening before committing machine time,
# powder, and downstream inspection/post-processing resources.
# ---------------------------------------------------------------------------
PRE_BUILD_PROCESS_FEATURES = [
    "Powder_Size_um",
    "Oxygen_Content_percent",
    "Laser_Power_W",
    "Scan_Speed_mm_s",
    "Hatch_Spacing_mm",
    "Layer_Thickness_um",
    "Preheat_Temp_C",
    "Shielding_Gas_Flow_L_min",
]

# ---------------------------------------------------------------------------
# Features available or computable during the build.
#
# Business objective:
# These features support in-build quality-risk updates. This is the main MVP
# because it can still influence operational action before the build is fully
# lost or before final inspection is completed.
# ---------------------------------------------------------------------------
IN_BUILD_MONITORING_FEATURES = [
    "Melt_Pool_Width_um",
    "Melt_Pool_Depth_um",
    "Melt_Pool_Temp_C",
    "Cooling_Rate_K_s",
    "Energy_Density_J_mm3",
    "Thermal_Gradient",
]

# ---------------------------------------------------------------------------
# Scenario feature sets.
#
# Business objective:
# Each scenario answers a different production question. This is safer and more
# useful than one large model that mixes features from different decision times.
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    # Pre-build risk screening:
    # only features known before manufacturing starts.
    "prebuild": CATEGORICAL_FEATURES + PRE_BUILD_PROCESS_FEATURES,

    # In-build risk update:
    # pre-build features plus monitoring features available during production.
    "inbuild": (
        CATEGORICAL_FEATURES
        + PRE_BUILD_PROCESS_FEATURES
        + IN_BUILD_MONITORING_FEATURES
    ),

    # Post-build diagnostic benchmark:
    # all process and post-build features are allowed because this model is used
    # after inspection/testing, not for real-time operational prediction.
    "postbuild": CATEGORICAL_FEATURES + PROCESS_FEATURES + POST_BUILD_FEATURES,
}


SCENARIO_DESCRIPTIONS = {
    "prebuild": {
        "business_question": "Before starting the build, does this job look risky?",
        "business_value": (
            "Avoid wasting machine time, powder, and post-processing cost on "
            "jobs that already look high risk before production starts."
        ),
        "decision_time": "Before build start",
        "operational_use": "Early risk screening",
        "is_operational_prediction": True,
    },
    "inbuild": {
        "business_question": "During the build, is quality risk increasing?",
        "business_value": (
            "Trigger operator alerts, inspection escalation, pause/abort rules, "
            "or parameter review before the full batch is lost."
        ),
        "decision_time": "During build",
        "operational_use": "Production MVP risk update",
        "is_operational_prediction": True,
    },
    "postbuild": {
        "business_question": (
            "After inspection and testing, what patterns explain the quality outcome?"
        ),
        "business_value": (
            "Support root-cause analysis, engineering review, process learning, "
            "powder investigation, and machine investigation."
        ),
        "decision_time": "After build and inspection",
        "operational_use": "Diagnostic benchmark only",
        "is_operational_prediction": False,
    },
}


def get_feature_set(scenario: str) -> list[str]:
    """
    Return the allowed features for a production decision scenario.

    Business objective
    ------------------
    Make sure each model only receives information that would be available at
    the relevant decision time.

    Coding objective
    ----------------
    Return a reusable list of feature columns for training, evaluation, tests,
    and future API validation.
    """

    if scenario not in FEATURE_SETS:
        valid = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"Unknown scenario '{scenario}'. Valid options: {valid}")

    return FEATURE_SETS[scenario]


def get_scenario_description(scenario: str) -> dict:
    """
    Return business metadata for a scenario.

    Business objective
    ------------------
    Keep the model artifact self-explanatory for engineering reviewers,
    interviewers, and future production users.

    Coding objective
    ----------------
    Provide metadata that can be saved together with model metrics and artifacts.
    """

    if scenario not in SCENARIO_DESCRIPTIONS:
        valid = ", ".join(sorted(SCENARIO_DESCRIPTIONS))
        raise ValueError(f"Unknown scenario '{scenario}'. Valid options: {valid}")

    return SCENARIO_DESCRIPTIONS[scenario]


def validate_no_timing_leakage(scenario: str) -> None:
    """
    Validate that operational models do not use unavailable future information.

    Business objective
    ------------------
    Prevent an offline model from looking better than it would perform in real
    production by using inspection or mechanical-test data that is only known
    after the build.

    Coding objective
    ----------------
    Raise an error if:
    - ID columns are used as features
    - the target is used as a feature
    - pre-build or in-build models contain post-build inspection/test columns
    """

    features = set(get_feature_set(scenario))

    forbidden_columns = set(ID_COLUMNS + [TARGET_COLUMN])

    if scenario in {"prebuild", "inbuild"}:
        forbidden_columns |= set(POST_BUILD_FEATURES)

    leaked_columns = sorted(features & forbidden_columns)

    if leaked_columns:
        raise ValueError(
            f"Feature set '{scenario}' contains timing-leakage columns: "
            f"{leaked_columns}"
        )