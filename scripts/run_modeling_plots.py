"""
Run Task 2 modeling plot generation.
"""

import argparse
from pathlib import Path

from lpbf_quality.config.settings import FIGURES_DIR, METRICS_DIR
from lpbf_quality.visualization.modeling_plots import run_modeling_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--comparison",
        default=METRICS_DIR / "comparisons" / "model_comparison.csv",
        type=Path,
    )

    parser.add_argument(
        "--recommendations",
        default=METRICS_DIR / "comparisons" / "model_recommendations.csv",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        default=FIGURES_DIR / "task2",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs = run_modeling_plots(
        comparison_path=args.comparison,
        recommendations_path=args.recommendations,
        output_dir=args.output_dir,
    )

    print("Generated Task 2 modeling plots:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()