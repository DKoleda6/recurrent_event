# Survival Analysis Pipeline for Recidivism Modeling

## Project Overview

This project implements a modular and extensible survival analysis pipeline for modeling recidivism risk using recurrent-event Cox proportional hazards models.

The system is designed to:

- Train survival models on episodic offender data  
- Perform individual-level train/test splits  
- Compute both global and per-individual survival quality metrics  
- Export structured evaluation results  
- Generate visual performance reports  
- Support easy addition of new models and metrics  

The architecture follows clean software engineering principles and is designed to be research- and thesis-ready.

---

# Project Architecture

```graphql
project/
│
├── data/
│ ├── data_processor.py
│ └── splitter.py
│
├── models/
│ ├── base_model.py
│ └── cox_model.py
│
├── metrics/
│ ├── base_metric.py
│ ├── ibs.py
│ ├── ibs_remain.py
│ └── auprc.py
│
├── pipeline/
│ ├── trainer.py
│ └── evaluator.py
│
├── reports/
│ ├── csv/
│ └── plots/
│
├── main.py
└── README.md
```

---

# Folder and File Descriptions
## data/

### `data_processor.py`

Responsible for:

- Loading the raw dataset  
- Cleaning list-like columns  
- Constructing the long-format survival dataset  
- Building recurrent event structure:
  - entry times  
  - duration times  
  - event indicators  
  - episode identifiers  

Outputs: `cox_df`
Which is the fully prepared survival modeling dataset.
---
### `data_processor.py`
Implements: `SurvivalSplitter`

Performs:
- Individual-level train/test splitting
- Prevents data leakage
- Ensures all episodes of one individual stay in same split
Why individual-level splitting is required:
Each offender has multiple episodes. Splitting by row would cause leakage.
---
## models/
### `base_model.py`
Defines the abstract interface:
```python
fit(df)
predict_survival(df, times)
```
All survival models must follow this interface.
This guarantees:
- Interchangeable models
- Clean experimentation
- Extensibility
---
### `cox_model.py`
Implements:
- Stratified recurrent-event Cox model
- Uses `fit_stratified_cox`
- Accepts feature list at initialization
- Predicts survival functions at arbitrary time grid
Example:
```python
model = CoxModel(features=["age"])
```
---
## metrics/
Defines modular evaluation metrics.
Each metric:
- Inherits from `BaseMetric`
- Implements `compute(...)`
- Can be added to pipeline independently
---
### `ibs.py`
Computes:
- Mean Integrated Brier Score
- IBS over time
---
### `ibs_remain.py`
Computes:
- Per-individual IBS_remain
This measures remaining survival prediction quality per offender.
---
### `auprc.py`
Computes:
- Per-individual AUPRC
- Mean AUPRC
---
## pipeline/
### `trainer.py`
Responsible for:
- Training models
- Generating predictions
- Coordinating evaluation
Keeps model logic separate from orchestration logic.
---
### `evaluator.py`
Handles:
- Per-individual metric aggregation
- CSV export
- Summary metric generation
- IBS over time plotting
- Saving visual reports
Outputs are written to:
```bash
reports/csv/
reports/plots/
```
---
## reports/
Automatically generated evaluation results.
**CSV outputs**:
individual_metrics.csv
ibs_by_time.csv
model_summary.csv

**Plots**:
ibs_time_plot.png
---
## main.py
Main execution script.
Workflow:
1. Load and preprocess data
2. Perform train/test split
3. Prepare survival targets
4. Train model on training set
5. Predict on test set
6. Compute:
    - Mean IBS
    - IBS over time
    - Per-individual IBS_remain
    - Per-individual AUPRC
7. Export reports

---

# Train/Test Methodology
The dataset is split by individual (`name`):
- 80% train
- 20% test
This prevents:
- Episode-level leakage
- Optimistic bias
Evaluation metrics are computed strictly on the test set.

---

# Evaluation Metrics
The following metrics are computed:
**Global metrics**:
- Mean Integrated Brier Score (IBS)
- Mean AUPRC

**Time-dependent metrics**:
- IBS(t)

**Per-individual metrics**:
- IBS_remain
- AUPRC per offender

---

# How to Run
From project root:
```bash
python3 main.py
```
Generated output will appear in: `reports\`

---

# How to Add a New Model
1. Create a new file in models/
2. Inherit from BaseSurvivalModel
3. Implement:
    ```python
    fit()
    predict_survival()
    ```
4. Replace model in main.py
No other changes required.

---

# How to Add a New Metric
1. Create new file in metrics/
2. Inherit from BaseMetric
3. Implement compute()
4. all metric in main.py
No architectural modifications needed.