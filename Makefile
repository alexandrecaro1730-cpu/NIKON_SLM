.PHONY: install test smoke eda deep-eda

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests -q

smoke:
	python scripts/production_smoke_test.py

eda:
	python scripts/run_eda.py --input data/raw/lpbf_titanium_dataset.csv

deep-eda:
	python scripts/run_eda.py --input data/raw/lpbf_titanium_dataset.csv --run-deep-eda