"""
Scenario-specific model selection policy.

Business objective
------------------
Use a model-selection metric that matches the production decision.

Different production stages have different priorities:

1. Pre-build risk screening
   Main risk:
   Starting a build that should have been flagged as risky.

   Primary metric:
   risk_f2, because recall of risky builds matters more than false alarms.

2. In-build risk update
   Main risk:
   Missing a deteriorating build while there is still time to act.

   Primary metric:
   risk_f2, because catching Poor/Moderate builds is more important than
   perfectly classifying all four quality labels.

3. Post-build diagnostic benchmark
   Main risk:
   Misunderstanding the quality outcome and root-cause patterns.

   Primary metric:
   macro_f1, because post-build diagnosis should explain all quality classes
   rather than only binary risk.

Coding objective
----------------
Define scenario-specific primary and secondary metrics for model comparison and
deployment-readiness checks.
"""


SCENARIO_SELECTION_POLICY = {
    "prebuild": {
        "primary_metric": "risk_f2",
        "secondary_metric": "risk_recall",
        "minimum_primary_lift_over_dummy": 0.05,
        "minimum_secondary_lift_over_dummy": 0.05,
        "business_reason": (
            "Pre-build screening should prioritize catching risky jobs before "
            "machine time, powder, and post-processing resources are committed."
        ),
    },
    "inbuild": {
        "primary_metric": "risk_f2",
        "secondary_metric": "risk_recall",
        "minimum_primary_lift_over_dummy": 0.05,
        "minimum_secondary_lift_over_dummy": 0.05,
        "business_reason": (
            "In-build monitoring should prioritize detecting Poor/Moderate risk "
            "while there is still time for inspection escalation, operator alert, "
            "or parameter review."
        ),
    },
    "postbuild": {
        "primary_metric": "macro_f1",
        "secondary_metric": "balanced_accuracy",
        "minimum_primary_lift_over_dummy": 0.05,
        "minimum_secondary_lift_over_dummy": 0.05,
        "business_reason": (
            "Post-build diagnosis should explain all quality classes, so balanced "
            "multi-class behavior is more important than only binary risk detection."
        ),
    },
}


def get_selection_policy(scenario: str) -> dict:
    """
    Return model selection policy for a production scenario.

    Business objective
    ------------------
    Make model selection reflect production priorities instead of using one
    generic metric for all use cases.

    Coding objective
    ----------------
    Return a dictionary with primary/secondary metric names and thresholds.
    """

    if scenario not in SCENARIO_SELECTION_POLICY:
        valid = ", ".join(sorted(SCENARIO_SELECTION_POLICY))
        raise ValueError(f"Unknown scenario '{scenario}'. Valid options: {valid}")

    return SCENARIO_SELECTION_POLICY[scenario]