# LPBF Titanium Quality Prediction

Scenario-based quality prediction and production-readiness workflow for a Laser Powder Bed Fusion (LPBF) titanium dataset.

This project is designed as an engineering-oriented machine-learning proof of concept. The goal is not to chase the highest offline score, but to build a reliable workflow that respects production decision timing, avoids feature leakage, communicates model limitations clearly, and demonstrates how a model could be integrated into an industrial production environment.

---

## 1. Project objective

The dataset contains LPBF titanium process, monitoring, inspection, and quality information. The target variable is:

```text
Quality_Class = Poor, Moderate, Good, Excellent
```

The project is structured around four practical production objectives:

| Objective                | Decision timing          | Purpose                                                      |
| ------------------------ | ------------------------ | ------------------------------------------------------------ |
| Pre-build risk screening | Before the build starts  | Flag risky jobs before machine time and powder are committed |
| In-build risk update     | During the build         | Update quality risk while there is still time to intervene   |
| Post-build diagnosis     | After inspection/testing | Understand quality outcomes and root-cause patterns          |
| Production monitoring    | Over time                | Detect process drift, quality drift, and model degradation   |

The recommended production MVP is the **in-build risk update** workflow because it uses material, process, and monitoring features while excluding post-build inspection/test features.

---

## 2. Key engineering principles

This project follows four rules:

1. **Use only features available at the decision time.**
   Pre-build and in-build models must not use post-build inspection or mechanical-test variables.

2. **Compare against simple baselines.**
   A model is only useful if it improves meaningfully over a naive baseline.

3. **Select metrics based on the business objective.**
   Accuracy alone is not enough because it can hide weak detection of risky Poor/Moderate builds.

4. **Do not overclaim production readiness.**
   The dataset is technically clean, but Task 1 showed weak LPBF physics realism. The model should be treated as a workflow-validation candidate, not a trusted production predictor.

---

## 3. Repository structure

```text
.
├── data/
│   └── raw/
│       └── lpbf_titanium_dataset.csv
├── models/
│   ├── prebuild/
│   ├── inbuild/
│   └── postbuild/
├── reports/
│   ├── eda/
│   ├── figures/
│   └── metrics/
├── scripts/
│   ├── run_eda.py
│   ├── compare_models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inspect_feature_sets.py
│   ├── generate_modeling_report.py
│   ├── run_modeling_plots.py
│   ├── production_smoke_test.py
│   └── serve_api.py
├── src/
│   └── lpbf_quality/
│       ├── api/
│       ├── config/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       ├── schemas/
│       ├── utils/
│       └── visualization/
├── tests/
├── Makefile
├── pyproject.toml
└── README.md
```

Generated model artifacts are written to:

```text
models/<scenario>/<model_type>.joblib
```

Those files are reproducible and normally should not be committed to Git.

---

## 4. Environment setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

Or use:

```bash
make install
```

---

## 5. Run tests

Run the full test suite:

```bash
python -m pytest tests -q
```

Or:

```bash
make test
```

Expected output:

```text
20 passed
```

Business value:

```text
Confirms that data validation, feature-set contracts, model registry, model selection, Task 2 modeling, and Task 3 API service contracts are working.
```

---

## 6. Task 1 — Data Exploration and Quality Assessment

Task 1 investigates whether the dataset is usable for modeling and whether it behaves like plausible LPBF production data.

### 6.1 Run standard EDA

```bash
python scripts/run_eda.py --input data/raw/lpbf_titanium_dataset.csv
```

Or:

```bash
make eda
```

Generates:

```text
reports/eda/*.csv
reports/eda/challenge/*.csv
reports/eda/production_data_quality.log
```

Business value:

```text
Checks whether the dataset is usable for modeling:
- schema
- missing values
- duplicates
- physical ranges
- target distribution
- leakage candidates
- LPBF realism challenges
```

### 6.2 Run deep EDA and plots

```bash
python scripts/run_eda.py \
  --input data/raw/lpbf_titanium_dataset.csv \
  --run-deep-eda
```

Or:

```bash
make deep-eda
```

Generates:

```text
reports/figures/*.png
reports/figures/business/*.png
```

Business value:

```text
Creates presentation-ready plots for:
- class distribution
- process window risk
- physics correlation checks
- energy density quality trends
- feature distributions
- leakage and prediction timing
```

### 6.3 Task 1 conclusion

Task 1 found that the dataset is technically clean:

```text
- no missing values
- no duplicate rows
- no invalid physical ranges
- no IQR outliers detected
```

But it also found important limitations:

```text
- feature distributions are unusually bounded and smooth
- expected LPBF physics relationships are weak
- energy density does not strongly match the expected formula
- quality-class separation is weak
- post-build features are timing-sensitive
```

Conclusion:

```text
The dataset is suitable for demonstrating a production ML workflow, but not enough to claim physically validated LPBF prediction performance.
```

---

## 7. Task 2 — Quality Prediction Model

Task 2 builds and evaluates models for `Quality_Class`.

The modeling workflow uses scenario-specific feature views:

| Scenario   | Features used                                                | Excluded                                                      |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------- |
| Pre-build  | Categorical + powder/material + planned process features     | Sample_ID, Quality_Class, post-build inspection/test features |
| In-build   | Pre-build features + melt-pool / thermal monitoring features | Sample_ID, Quality_Class, post-build inspection/test features |
| Post-build | Process + monitoring + post-build inspection/test features   | Sample_ID, Quality_Class                                      |
| Monitoring | Aggregate quality/process/model signals over time            | Not a single-row classifier                                   |

---

## 8. Inspect feature sets

Run:

```bash
python scripts/inspect_feature_sets.py
```

Generates:

```text
Terminal report showing the exact features used by each scenario.
```

Expected result:

```text
Feature-set inspection passed.
```

Business value:

```text
Confirms that operational models avoid timing leakage. Pre-build and in-build models exclude post-build inspection/test variables.
```

---

## 9. Compare models

Run the full comparison:

```bash
python scripts/compare_models.py \
  --input data/raw/lpbf_titanium_dataset.csv \
  --scenario all
```

Or:

```bash
make compare-models
```

Generates:

```text
reports/metrics/comparisons/model_comparison.csv
reports/metrics/comparisons/model_recommendations.csv
```

Business value:

```text
Compares model complexity against business objectives:
- DummyClassifier
- Logistic Regression
- Random Forest
- Histogram Gradient Boosting
```

The comparison uses scenario-specific selection metrics:

| Scenario   | Primary metric | Secondary metric  | Business reason                                           |
| ---------- | -------------- | ----------------- | --------------------------------------------------------- |
| Pre-build  | risk_f2        | risk_recall       | Catch risky jobs before production starts                 |
| In-build   | risk_f2        | risk_recall       | Catch Poor/Moderate builds while action is still possible |
| Post-build | macro_f1       | balanced_accuracy | Explain all four quality classes fairly                   |

---

## 10. Current Task 2 model-selection results

Latest comparison results:

| Scenario   | Selected model      | Primary metric | Result | Recommendation                       |
| ---------- | ------------------- | -------------: | -----: | ------------------------------------ |
| Pre-build  | Logistic Regression |        Risk F2 |  0.428 | Candidate for engineering validation |
| In-build   | Logistic Regression |        Risk F2 |  0.441 | Candidate for engineering validation |
| Post-build | Logistic Regression |       Macro F1 |  0.262 | Not recommended for deployment       |

Important interpretation:

```text
Random Forest has slightly better general in-build multi-class performance,
but Logistic Regression is much better on operational Poor/Moderate risk detection.

Therefore, Logistic Regression is the better in-build MVP candidate because
the production objective is to catch risky builds, not maximize a generic score.
```

---

## 11. Train model artifacts

Train all scenarios and all registered models:

```bash
python scripts/train.py \
  --input data/raw/lpbf_titanium_dataset.csv \
  --scenario all \
  --model all
```

Or:

```bash
make train-all-models
```

Generates:

```text
models/prebuild/*.joblib
models/inbuild/*.joblib
models/postbuild/*.joblib

reports/metrics/prebuild/*_metrics.json
reports/metrics/prebuild/*_confusion_matrix.csv

reports/metrics/inbuild/*_metrics.json
reports/metrics/inbuild/*_confusion_matrix.csv

reports/metrics/postbuild/*_metrics.json
reports/metrics/postbuild/*_confusion_matrix.csv
```

Business value:

```text
Creates reproducible model artifacts and auditable metric files for each production scenario.
```

Train only the recommended in-build MVP candidate:

```bash
python scripts/train.py \
  --input data/raw/lpbf_titanium_dataset.csv \
  --scenario inbuild \
  --model logistic_regression
```

Or:

```bash
make train-inbuild
```

---

## 12. Evaluate a saved model

Example:

```bash
python scripts/evaluate.py \
  --input data/raw/lpbf_titanium_dataset.csv \
  --model-path models/inbuild/logistic_regression.joblib \
  --output-name logistic_regression_full_data_eval
```

Or:

```bash
make evaluate-inbuild
```

Generates:

```text
reports/metrics/inbuild/logistic_regression_full_data_eval_metrics.json
reports/metrics/inbuild/logistic_regression_full_data_eval_confusion_matrix.csv
```

Business value:

```text
Evaluates a saved model artifact using the same scenario-specific feature view used during training.
```

---

## 13. Generate Task 2 modeling report

Run:

```bash
python scripts/generate_modeling_report.py
```

Or:

```bash
make modeling-report
```

Generates:

```text
reports/metrics/modeling_report.md
```

Business value:

```text
Creates a Markdown summary of:
- model complexity ladder
- scenario-level results
- deployment recommendation
- metric interpretation
- limitations and next steps
```

---

## 14. Generate Task 2 presentation plots

Run:

```bash
python scripts/run_modeling_plots.py
```

Or:

```bash
make modeling-plots
```

Generates:

```text
reports/figures/task2/slide1_quality_intelligence_framework.png
reports/figures/task2/slide2_metric_policy_table.png
reports/figures/task2/slide3_operational_risk_f2.png
reports/figures/task2/slide4_inbuild_tradeoff.png
reports/figures/task2/slide5_monitoring_plan.png
```

Business value:

```text
Creates visuals for explaining:
- the four production objectives
- why metrics differ by production stage
- why Logistic Regression is selected for operational risk detection
- why production monitoring is needed before deployment
```

---

## 15. Task 3 — Production-Oriented Service

Task 3 wraps the selected in-build Logistic Regression model in a small FastAPI service.

The service is designed as a proof-of-concept decision-support API, not as an autonomous production controller.

### 15.1 Service endpoints

| Endpoint      | Method | Purpose                          | Business value                                                               |
| ------------- | -----: | -------------------------------- | ---------------------------------------------------------------------------- |
| `/`           |    GET | Navigation endpoint              | Helps users find available API routes                                        |
| `/health`     |    GET | Service health check             | Supports deployment and monitoring checks                                    |
| `/model-info` |    GET | Model metadata and limitations   | Makes model assumptions and feature contract visible                         |
| `/predict`    |   POST | In-build quality-risk prediction | Returns predicted class, risk flag, risk probability, and recommended action |

### 15.2 Start the API

First train the selected in-build MVP model if the artifact does not already exist:

```bash
python scripts/train.py \
  --input data/raw/lpbf_titanium_dataset.csv \
  --scenario inbuild \
  --model logistic_regression
```

Then start the API:

```bash
python scripts/serve_api.py
```

Or:

```bash
make serve-api
```

Open:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
```

### 15.3 Run the production smoke test

```bash
python scripts/production_smoke_test.py
```

Or:

```bash
make smoke
```

Expected result:

```text
Production smoke test passed.
```

Business value:

```text
Verifies the operational path:
model artifact -> API schema -> prediction -> production decision response
```

### 15.4 Example prediction request

With the API running, send a request to `/predict`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Alloy_Type": "Ti-6Al-4V",
    "Powder_Morphology": "Spherical",
    "Scan_Strategy": "Stripe",
    "Powder_Size_um": 35.0,
    "Oxygen_Content_percent": 0.12,
    "Laser_Power_W": 280.0,
    "Scan_Speed_mm_s": 950.0,
    "Hatch_Spacing_mm": 0.11,
    "Layer_Thickness_um": 30.0,
    "Preheat_Temp_C": 180.0,
    "Shielding_Gas_Flow_L_min": 28.0,
    "Melt_Pool_Width_um": 115.0,
    "Melt_Pool_Depth_um": 55.0,
    "Melt_Pool_Temp_C": 1850.0,
    "Cooling_Rate_K_s": 850000.0,
    "Energy_Density_J_mm3": 75.0,
    "Thermal_Gradient": 12000.0
  }' | python -m json.tool
```

Example response:

```json
{
  "scenario": "inbuild",
  "model_type": "logistic_regression",
  "model_status": "candidate_for_engineering_validation",
  "predicted_quality_class": "Good",
  "risk_group": "Acceptable",
  "risk_flag": false,
  "risk_probability": 0.052,
  "recommended_action": "Continue normal production monitoring."
}
```

### 15.5 Production interpretation

The API accepts only the 17 leakage-safe in-build features:

```text
material + powder + planned process settings + melt-pool / thermal monitoring
```

It intentionally excludes post-build inspection and mechanical-test features because those are not available during the build.

The service output is designed for operational decision support:

| Model output                | Operational action                                  |
| --------------------------- | --------------------------------------------------- |
| Acceptable / low risk       | Continue normal production monitoring               |
| Elevated risk probability   | Continue build but flag for closer monitoring       |
| Poor or Moderate prediction | Escalate for operator review or inspection planning |
| Invalid input               | Reject request and correct production data          |

The service should not be used for autonomous machine control or final part release without validation on physically coherent production data.

---

## 16. Production smoke test

Run:

```bash
python scripts/production_smoke_test.py
```

Or:

```bash
make smoke
```

Business value:

```text
Confirms that the production-oriented API can load the selected model artifact, validate an in-build request, and return an operational decision response.
```

---

## 17. Makefile command summary

| Command                 | What it does                                       | What it generates                                       | Business value                                      |
| ----------------------- | -------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------- |
| `make install`          | Installs package and dev dependencies              | Editable package environment                            | Reproducible setup                                  |
| `make test`             | Runs all tests                                     | Terminal test result                                    | Confirms workflow contracts                         |
| `make smoke`            | Runs production API smoke test                     | Terminal result                                         | Checks production workflow basics                   |
| `make production-smoke` | Runs production API smoke test                     | Terminal result                                         | Verifies model artifact -> API -> decision response |
| `make serve-api`        | Starts the local FastAPI service                   | API at `127.0.0.1:8000`                                 | Demonstrates Task 3 service integration             |
| `make eda`              | Runs Task 1 standard EDA                           | `reports/eda/*.csv`                                     | Data-quality and realism audit                      |
| `make deep-eda`         | Runs EDA + plots                                   | `reports/figures/*`                                     | Presentation-ready EDA visuals                      |
| `make compare-models`   | Compares all models/scenarios                      | `reports/metrics/comparisons/*.csv`                     | Evidence-based model selection                      |
| `make compare-inbuild`  | Compares models for in-build only                  | comparison CSVs                                         | Focused MVP model comparison                        |
| `make train-inbuild`    | Trains selected in-build Logistic Regression model | `models/inbuild/logistic_regression.joblib` and metrics | Reproducible MVP model artifact                     |
| `make train-all-models` | Trains all registered models                       | `models/<scenario>/*.joblib` and metrics                | Reproducible model artifacts                        |
| `make modeling-report`  | Generates Task 2 Markdown report                   | `reports/metrics/modeling_report.md`                    | Documentation artifact                              |
| `make modeling-plots`   | Generates Task 2 plots                             | `reports/figures/task2/*.png`                           | Presentation visuals                                |
| `make evaluate-inbuild` | Evaluates saved in-build model                     | evaluation metrics CSV/JSON                             | Model artifact validation                           |

---

## 18. Recommended run order from a fresh clone

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest tests -q

python scripts/run_eda.py \
  --input data/raw/lpbf_titanium_dataset.csv \
  --run-deep-eda

python scripts/inspect_feature_sets.py

python scripts/compare_models.py \
  --input data/raw/lpbf_titanium_dataset.csv \
  --scenario all

python scripts/train.py \
  --input data/raw/lpbf_titanium_dataset.csv \
  --scenario inbuild \
  --model logistic_regression

python scripts/production_smoke_test.py

python scripts/generate_modeling_report.py

python scripts/run_modeling_plots.py
```

Equivalent Makefile flow:

```bash
make install
make test
make deep-eda
python scripts/inspect_feature_sets.py
make compare-models
make train-inbuild
make smoke
make modeling-report
make modeling-plots
```

To run the local API after training the in-build model:

```bash
make serve-api
```

Then open:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
```

---

## 19. Expected generated outputs

After running the full workflow, the main generated outputs are:

```text
reports/eda/
reports/eda/challenge/
reports/figures/
reports/figures/business/
reports/figures/task2/
reports/metrics/
reports/metrics/comparisons/
reports/metrics/modeling_report.md
models/inbuild/logistic_regression.joblib
```

If all models are trained, additional artifacts are generated:

```text
models/prebuild/
models/inbuild/
models/postbuild/
```

The `models/` folder contains generated `.joblib` artifacts. These can be regenerated and normally should not be committed unless required.

---

## 20. Feature engineering scope

The current proof of concept intentionally uses the raw, leakage-safe scenario feature sets rather than adding complex domain-specific engineered features.

This is a deliberate engineering choice. Task 1 showed that the dataset is technically clean but does not strongly preserve expected LPBF physics relationships. Because of that, forcing physics-derived features such as volumetric energy density transformations or heat-input ratios into the first modeling baseline could create misleading signals.

The current baseline therefore answers this question first:

```text
How much value can we get from the raw features available at the correct production decision time?
```

For the selected in-build MVP, the API accepts 17 raw in-build features:

```text
material + powder + planned process settings + melt-pool / thermal monitoring
```

Post-build inspection and mechanical-test features are intentionally excluded from the operational API to avoid timing leakage.

### Planned production feature engineering layer

In a real production deployment with validated machine telemetry, the next iteration would add a feature engineering layer after API validation and before model inference:

```text
Incoming JSON payload
        ↓
Pydantic API validation
        ↓
Domain-specific feature engineering
        ↓
Scikit-learn preprocessing
        ↓
Model inference
        ↓
Operational risk decision
```

Potential engineered features include:

| Category                          | Example features                                            | Production value                               |
| --------------------------------- | ----------------------------------------------------------- | ---------------------------------------------- |
| Multi-variable process indicators | Volumetric energy density, `P/v`, heat-input ratios         | Capture melt-pool energy conditions            |
| Thermal proxies                   | Cooling-rate ratios, thermal-gradient interactions          | Capture solidification and microstructure risk |
| Sensor aggregations               | Rolling mean, standard deviation, peak-to-average, kurtosis | Capture transient melt-pool instability        |
| Categorical interactions          | Alloy/process-window interactions                           | Normalize risk by material/process context     |

These features should be added only after validating that production data preserves meaningful LPBF physics and after comparing performance against the current raw-feature baseline.

---

## 21. Production readiness interpretation

The workflow is successful as a proof of concept because it demonstrates:

```text
- leakage-safe feature views
- robust preprocessing
- baseline comparison
- scenario-specific business metrics
- model artifact saving
- auditable metrics
- reporting and plotting
- production API wrapper
- model metadata endpoint
- API contract tests
- production smoke test
- production monitoring design
```

The current trained model and API should not be treated as a trusted production predictor because:

```text
- Task 1 showed weak LPBF physics realism
- post-build diagnostic performance remains weak despite inspection/test features
- production metadata such as build ID, machine ID, operator ID, powder batch, and timestamp is missing
- the dataset may be controlled, pre-cleaned, or synthetic
- the API has not been validated on live machine telemetry
```

Recommendation:

```text
Use the in-build Logistic Regression workflow as the production MVP concept.
Validate on physically coherent production data before deployment.
Deploy only with monitoring for quality drift, process drift, model behavior drift, and human-in-the-loop escalation.
```

---

## 22. External libraries, frameworks, and AI-assisted tooling

### Python libraries and frameworks

| Tool           | Purpose                                                            |
| -------------- | ------------------------------------------------------------------ |
| `pandas`       | Data loading, tabular processing, EDA summaries                    |
| `numpy`        | Numeric operations                                                 |
| `scikit-learn` | Preprocessing pipelines, model training, metrics, train/test split |
| `joblib`       | Saving and loading trained model artifacts                         |
| `matplotlib`   | Plot generation                                                    |
| `FastAPI`      | Production-oriented API service                                    |
| `Pydantic`     | API request and response validation                                |
| `uvicorn`      | Local API server                                                   |
| `pytest`       | Automated testing                                                  |

### AI-assisted tooling

AI-assisted tooling was used to support code drafting, documentation structure, presentation wording, and review of engineering trade-offs. All generated code and documentation were manually reviewed, adapted to the project structure, executed locally, and validated with automated tests.

Current validation status:

```text
python -m pytest tests -q
20 passed

python scripts/production_smoke_test.py
Production smoke test passed.
```

---

## 23. Final project summary

This project demonstrates a complete production-ML proof-of-concept workflow:

```text
Task 1:
Assess whether the data is clean, realistic, and safe to model.

Task 2:
Build leakage-safe scenario-specific models and select the best operational candidate.

Task 3:
Expose the selected in-build model as a tested API service for production decision support.
```

Final conclusion:

```text
The proof of concept is successful as an engineering workflow. The next step is not aggressive tuning of the current weakly physical dataset, but validation on production data, direct binary risk modeling, probability calibration, threshold tuning, and monitoring.
```
