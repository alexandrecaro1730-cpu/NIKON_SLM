"""
Train LPBF quality prediction models.

Business objective
------------------
Train a scenario-based quality intelligence system instead of one misleading
"all features" model.

The artifact structure is organized by production decision timing:

models/
    prebuild/
    inbuild/
    postbuild/

reports/metrics/
    prebuild/
    inbuild/
    postbuild/

This makes it clear which model belongs to which operational use case.

Coding objective
----------------
Load data, train selected models, save model artifacts, and save auditable
metrics files in scenario-specific folders.
"""

import argparse

from lpbf_quality.config.settings import METRICS_DIR, MODELS_DIR
from lpbf_quality.data.load_data import load_csv
from lpbf_quality.evaluation.metrics import (
    save_confusion_matrix_csv,
    save_metrics_json,
)
from lpbf_quality.models.model_registry import get_available_model_types
from lpbf_quality.models.training import save_model_artifact, train_model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Business objective
    ------------------
    Allow reviewers or production users to train one scenario/model or the full
    comparison suite from the command line.

    Coding objective
    ----------------
    Return parsed CLI arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/raw/lpbf_titanium_dataset.csv",
        help="Path to input CSV file.",
    )

    parser.add_argument(
        "--scenario",
        choices=["prebuild", "inbuild", "postbuild", "all"],
        default="all",
        help="Prediction scenario to train.",
    )

    parser.add_argument(
        "--model",
        choices=get_available_model_types() + ["all"],
        default="all",
        help="Model type to train.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout test fraction.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def expand_arg(value: str, available_values: list[str]) -> list[str]:
    """
    Expand 'all' into concrete values.

    Business objective
    ------------------
    Make it easy to train either one focused model or the full comparison suite.

    Coding objective
    ----------------
    Convert a CLI option into a list that can be looped over.
    """

    if value == "all":
        return available_values

    return [value]


def main() -> None:
    """
    Train selected LPBF quality models.

    Business objective
    ------------------
    Produce auditable model artifacts that support the Task 2 story:
    leakage-safe modeling, statistical baseline comparison, nonlinear ML
    comparison, boosting direction, and honest limitations.

    Coding objective
    ----------------
    For each selected scenario/model:
    - train model
    - save model artifact under models/<scenario>/
    - save metrics JSON under reports/metrics/<scenario>/
    - save confusion matrix CSV under reports/metrics/<scenario>/
    - print concise terminal summary
    """

    args = parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = expand_arg(
        args.scenario,
        ["prebuild", "inbuild", "postbuild"],
    )

    models = expand_arg(
        args.model,
        get_available_model_types(),
    )

    df = load_csv(args.input)

    for scenario in scenarios:
        for model_type in models:
            result = train_model(
                df=df,
                scenario=scenario,
                model_type=model_type,
                test_size=args.test_size,
                random_state=args.random_state,
            )

            # Business objective:
            # Store artifacts by production stage so reviewers can immediately
            # see which model belongs to which decision timing.
            scenario_model_dir = MODELS_DIR / scenario
            scenario_metrics_dir = METRICS_DIR / scenario

            artifact_path = scenario_model_dir / f"{model_type}.joblib"
            metrics_path = scenario_metrics_dir / f"{model_type}_metrics.json"
            confusion_path = scenario_metrics_dir / (
                f"{model_type}_confusion_matrix.csv"
            )

            save_model_artifact(
                pipeline=result["pipeline"],
                metadata=result["metadata"],
                output_path=artifact_path,
            )

            save_metrics_json(
                metrics=result["metrics"],
                output_path=metrics_path,
            )

            save_confusion_matrix_csv(
                metrics=result["metrics"],
                output_path=confusion_path,
            )

            print(
                f"Trained {scenario}/{model_type} | "
                f"balanced_accuracy={result['metrics']['balanced_accuracy']:.3f} | "
                f"macro_f1={result['metrics']['macro_f1']:.3f} | "
                f"artifact={artifact_path}"
            )


if __name__ == "__main__":
    main()