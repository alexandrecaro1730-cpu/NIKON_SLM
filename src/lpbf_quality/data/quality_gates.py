"""
Production data quality gates.

This module converts validation reports into production-style decisions:
PASS, WARN, or FAIL.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class QualityGateResult:
    status: str
    fail_count: int
    warn_count: int
    message: str


def evaluate_quality_gates(reports: dict[str, pd.DataFrame]) -> QualityGateResult:
    """
    Evaluate all report tables and return one production decision.
    """

    fail_count = 0
    warn_count = 0

    for _, report in reports.items():
        if "status" not in report.columns:
            continue

        fail_count += int((report["status"] == "FAIL").sum())
        warn_count += int((report["status"] == "WARN").sum())

    if fail_count > 0:
        return QualityGateResult(
            status="FAIL",
            fail_count=fail_count,
            warn_count=warn_count,
            message="Blocking data quality issues found. Pipeline should stop.",
        )

    if warn_count > 0:
        return QualityGateResult(
            status="WARN",
            fail_count=fail_count,
            warn_count=warn_count,
            message="Non-blocking warnings found. Review recommended before modeling.",
        )

    return QualityGateResult(
        status="PASS",
        fail_count=fail_count,
        warn_count=warn_count,
        message="All mandatory data and statistical checks passed.",
    )