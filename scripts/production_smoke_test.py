"""
Business objective
------------------
Provide a one-command production-readiness smoke test.

This helps an interviewer or teammate verify that the project can be installed,
tested, and executed without opening notebooks.

Coding objective
----------------
Run the pytest suite from Python and return a non-zero exit code if tests fail.
"""

import subprocess
import sys


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        check=False,
    )

    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()