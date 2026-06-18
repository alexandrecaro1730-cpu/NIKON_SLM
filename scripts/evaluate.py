"""
Evaluate a saved LPBF quality model.

Business objective
------------------
Evaluate a saved model using the same decision-time feature logic used during
training.

This protects against accidental evaluation with the wrong feature set.

Metrics are saved under:

reports/metrics/<scenario>/

Coding objective
----------------
Load a saved joblib artifact, recover its scenario metadata, rebuild X/y using
the correct scenario feature set, generate predictions, and save metrics.
"""

import argparse
from pathlib import Path

import joblib

from lpbf_quality.config.settings import METRICS_DIR, QUALITY_ORDER
from lpbf_quality.data.load_data import load_csv
from lpbf_quality.evaluation.metrics import (
    evaluate_classification,
    save_confusion_matrix_csv,
    save_metrics_json,
)
from lpbf_quality.features.build_features import build_model_frame


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Business objective
    ------------------
    Allow a saved model to be re-evaluated on a CSV export.

    Coding objective
    ----------------
    Return parsed CLI arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/raw/lpbf_titanium_dataset.csv",
        help="Path to evaluation CSV file.",
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to saved .joblib model artifact.",
    )

    parser.add_argument(
        "--output-name",
        default="external_evaluation",
        help="Prefix for saved evaluation files.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Evaluate a saved LPBF quality model.

    Business objective
    ------------------
    Check model behavior on an evaluation dataset while preserving the original
    model's intended production scenario.

    Coding objective
    ----------------
    Load artifact, infer scenario, rebuild scenario-specific features, predict,
    evaluate, and save metrics under reports/metrics/<scenario>/.
    """

    args = parse_args()

    artifact = joblib.load(args.model_path)
    pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]

    scenario = metadata["scenario"]

    scenario_metrics_dir = METRICS_DIR / scenario
    scenario_metrics_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.input)
    X, y = build_model_frame(df=df, scenario=scenario)

    y_pred = pipeline.predict(X)

    metrics = evaluate_classification(
        y_true=y,
        y_pred=y_pred,
        labels=QUALITY_ORDER,
    )

    metrics["metadata"] = {
        **metadata,
        "evaluation_input": args.input,
        "evaluation_mode": "external_file",
        "n_rows_evaluated": int(len(df)),
    }

    save_metrics_json(
        metrics=metrics,
        output_path=scenario_metrics_dir / f"{args.output_name}_metrics.json",
    )

    save_confusion_matrix_csv(
        metrics=metrics,
        output_path=scenario_metrics_dir
        / f"{args.output_name}_confusion_matrix.csv",
    )

    print(
        f"Evaluated model from {args.model_path} | "
        f"scenario={scenario} | "
        f"balanced_accuracy={metrics['balanced_accuracy']:.3f} | "
        f"macro_f1={metrics['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()