.PHONY: install test smoke eda deep-eda \
	train train-inbuild train-all-models train-boosting-inbuild \
	evaluate-inbuild \
	compare-models compare-inbuild \
	modeling-report

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

train:
	python scripts/train.py --input data/raw/lpbf_titanium_dataset.csv

train-inbuild:
	python scripts/train.py --input data/raw/lpbf_titanium_dataset.csv --scenario inbuild --model random_forest

train-all-models:
	python scripts/train.py --input data/raw/lpbf_titanium_dataset.csv --scenario all --model all

train-boosting-inbuild:
	python scripts/train.py --input data/raw/lpbf_titanium_dataset.csv --scenario inbuild --model hist_gradient_boosting

evaluate-inbuild:
	python scripts/evaluate.py --input data/raw/lpbf_titanium_dataset.csv --model-path models/inbuild/random_forest.joblib --output-name random_forest_full_data_eval

compare-models:
	python scripts/compare_models.py --input data/raw/lpbf_titanium_dataset.csv --scenario all

compare-inbuild:
	python scripts/compare_models.py --input data/raw/lpbf_titanium_dataset.csv --scenario inbuild

modeling-report:
	python scripts/generate_modeling_report.py

