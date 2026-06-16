"""
Business objective
------------------
Verify that a new developer can run the EDA entry point from the command line.

This test is a smoke test, not a production-quality-data test. The small fixture
may fail strict production quality gates, but the CLI should fail in a controlled
and understandable way instead of crashing with an unhandled exception.

Coding objective
----------------
Run scripts/run_eda.py against a small CSV and assert that the command executes
through the pipeline and either succeeds or exits through the expected quality
gate path.
"""

import subprocess
import sys
from pathlib import Path


def test_run_eda_cli_smoke(sample_csv_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_eda.py",
            "--input",
            str(sample_csv_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = result.stdout + result.stderr

    assert "Starting LPBF data analysis pipeline" in combined_output
    assert "Dataset loaded successfully" in combined_output
    assert "Running mandatory Level 1 data quality checks" in combined_output
    assert "Running mandatory Level 2 statistical and ML-readiness checks" in combined_output
    assert "Running Level 3 challenge and physics-realism checks" in combined_output

    assert result.returncode in {0, 1}

    if result.returncode == 1:
        assert "Pipeline stopped because mandatory checks failed" in combined_output