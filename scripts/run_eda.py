"""
Business objective
------------------
Run the complete LPBF data exploration and quality assessment workflow.

This script supports the interview task by separating:
1. mandatory production-style data checks
2. statistical and ML-readiness checks
3. independent challenge / realism checks
4. optional deep EDA plots
5. business-facing presentation plots

Coding objective
----------------
Provide one command-line entry point that loads the CSV, runs validation,
saves reports, logs decisions, optionally generates plots, and produces
challenge reports that test whether our initial EDA conclusions are reliable.
"""

import argparse
from pathlib import Path

from lpbf_quality.config.settings import EDA_DIR, FIGURES_DIR, REPORTS_DIR
from lpbf_quality.data.challenge_analysis import run_challenge_analysis
from lpbf_quality.data.load_data import load_csv
from lpbf_quality.data.quality_gates import evaluate_quality_gates
from lpbf_quality.data.statistical_validation import (
    build_statistical_validation_reports,
)
from lpbf_quality.data.validate_data import build_data_quality_report
from lpbf_quality.utils.logging import configure_logger
from lpbf_quality.visualization.business_plots import run_business_plots
from lpbf_quality.visualization.eda_plots import run_optional_deep_eda


def parse_args() -> argparse.Namespace:
    """
    Business objective
    ------------------
    Allow the user to control whether the pipeline runs only mandatory checks or
    also generates deep EDA and presentation artifacts.

    Coding objective
    ----------------
    Parse command-line arguments for input file, deep EDA mode, and strict
    warning behavior.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file.",
    )

    parser.add_argument(
        "--run-deep-eda",
        action="store_true",
        help="Generate detailed EDA tables, challenge reports, and plots.",
    )

    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Treat warnings as failures.",
    )

    return parser.parse_args()


def save_reports(reports: dict, output_dir: Path) -> None:
    """
    Business objective
    ------------------
    Save every validation output as an auditable CSV artifact that can be
    reviewed by engineers, quality teams, or interviewers.

    Coding objective
    ----------------
    Iterate over a dictionary of pandas DataFrames and save each one to the
    selected output directory.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, report in reports.items():
        report.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    """
    Business objective
    ------------------
    Execute the LPBF data assessment workflow end-to-end.

    Coding objective
    ----------------
    Load data, run mandatory reports, evaluate gates, optionally run deep EDA,
    challenge analysis, and business-facing plot generation.
    """

    args = parse_args()

    log_path = REPORTS_DIR / "eda" / "production_data_quality.log"
    logger = configure_logger(log_path)

    logger.info("Starting LPBF data analysis pipeline.")
    logger.info("Input file: %s", args.input)

    df = load_csv(args.input)

    logger.info("Dataset loaded successfully.")
    logger.info("Rows: %s | Columns: %s", df.shape[0], df.shape[1])

    logger.info("Running mandatory Level 1 data quality checks.")
    data_quality_reports = build_data_quality_report(df)
    save_reports(data_quality_reports, EDA_DIR)

    logger.info("Running mandatory Level 2 statistical and ML-readiness checks.")
    statistical_reports = build_statistical_validation_reports(df)
    save_reports(statistical_reports, EDA_DIR)

    logger.info("Running Level 3 challenge and physics-realism checks.")
    challenge_reports = run_challenge_analysis(
        df=df,
        output_dir=EDA_DIR / "challenge",
    )

    all_mandatory_reports = {
        **data_quality_reports,
        **statistical_reports,
    }

    gate_result = evaluate_quality_gates(all_mandatory_reports)

    logger.info("Overall mandatory pipeline status: %s", gate_result.status)
    logger.info("Fail count: %s", gate_result.fail_count)
    logger.info("Warning count: %s", gate_result.warn_count)
    logger.info("Decision message: %s", gate_result.message)

    logger.info(
        "Challenge reports generated: %s",
        ", ".join(challenge_reports.keys()),
    )

    if gate_result.status == "FAIL":
        logger.error("Pipeline stopped because mandatory checks failed.")
        raise SystemExit(1)

    if gate_result.status == "WARN" and args.fail_on_warning:
        logger.error("Pipeline stopped because warnings are treated as failures.")
        raise SystemExit(1)

    if gate_result.status == "WARN":
        logger.warning(
            "Pipeline completed with warnings. Review reports/eda before modeling."
        )

    if args.run_deep_eda:
        logger.info("Running optional deep EDA.")
        run_optional_deep_eda(
            df=df,
            eda_output_dir=EDA_DIR,
            figures_output_dir=FIGURES_DIR,
        )
        logger.info("Optional deep EDA completed.")

        logger.info("Generating business-facing presentation plots.")
        run_business_plots(
            df=df,
            eda_output_dir=EDA_DIR,
        )
        logger.info("Business-facing presentation plots completed.")

    logger.info("LPBF data analysis pipeline completed successfully.")


if __name__ == "__main__":
    main()