"""
Run the LPBF quality prediction API locally.

Business objective
------------------
Provide a simple command for reviewers or teammates to start the production-
oriented proof-of-concept service.

Coding objective
----------------
Run the FastAPI application with uvicorn.
"""

import uvicorn


def main() -> None:
    uvicorn.run(
        "lpbf_quality.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()