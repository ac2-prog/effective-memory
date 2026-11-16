# RISKY: 30-day hospital readmission (QH0636 Data Science)

Author: Karina Roade

Predicting whether a diabetic hospital encounter is followed by readmission within 30 days, using the
Diabetes 130-US Hospitals dataset (UCI Machine Learning Repository; Strack et al., 2014).

* **Response variable:** one column, `readmitted`, with three classes (`<30`, `>30`, `NO`).
* **Modelled target:** binary `early_readmission` = 1 if `readmitted == "<30"`, otherwise 0 (11.34% positive among the 100,114 analysed encounters).
* **Models:** Logistic Regression, Random Forest, HistGradientBoostingClassifier (scikit-learn).
* **Live dashboard:** https://risky-readmission.onrender.com/

## Requirements

Python 3.13 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## How to run (from this folder, in this order)

| Step | Command | What it produces |
|---|---|---|
| 1 | `python make_clean_dataset.py` | `data/diabetic_data_cleaned.csv`, `outputs/cleaning_log.csv` |
| 2 | `python train_models.py` | model metrics, test predictions, EDA and evaluation figures in `outputs/` |
| 3 | `python extend_evaluation.py` | `outputs/threshold_sensitivity.csv`, `outputs/model_calibration.png` |
| 4 | `python eda_quality_audit.py` | data-quality tables and extended EDA figures in `outputs/eda_extended/` |
| 5 | `python validate_models.py` | patient-grouped cross-validation in `validation_outputs/` (a few minutes) |
| 6 | `streamlit run dashboard.py` | interactive dashboard in the browser |

`RISKY_workbook.ipynb` repeats the whole pipeline (loading, cleaning, EDA, modelling, evaluation,
threshold sensitivity, calibration, cross-validation) with every result shown inline, and checks that its
numbers match the files in `outputs/`. Open it from this folder and choose *Run All*.

## Folder contents

```
data/diabetic_data.csv            original dataset (101,766 encounters x 50 columns)
data/diabetic_data_cleaned.csv    cleaned dataset produced by make_clean_dataset.py (100,114 encounters)
data/IDS_mapping.csv              code descriptions from UCI (e.g. discharge_disposition_id 11 = Expired)
make_clean_dataset.py             cleaning rules (shared by the notebook)
train_models.py                   split, pipelines, training, test-set evaluation
extend_evaluation.py              threshold sensitivity and reliability diagram
eda_quality_audit.py              data-quality audit and extended EDA
validate_models.py                patient-grouped cross-validation on the training set
dashboard.py                      Streamlit dashboard (Dataset & EDA, Model comparison, Threshold exploration, Calibration)
RISKY_workbook.ipynb              executed notebook with all outputs
outputs/, validation_outputs/     generated results (do not edit by hand)
```

## Key methodological choices

* `?` is treated as missing; `None` in `max_glu_serum` and `A1Cresult` means "test not performed" and is kept as a category.
* `weight` (96.86% missing) is dropped; three `Unknown/Invalid` gender values are treated as missing.
* The 1,652 encounters that ended in the patient's death are removed (readmission is impossible); hospice discharges are kept because 5.6% of them were readmitted within 30 days. Strack et al. (2014) excluded both groups.
* 14 encounters (13 patients) recorded after an encounter coded as death are retained and flagged as a data-quality limitation: the dataset has no dates, so the erroneous record cannot be identified (see the notebook, section 4).
* The train/test split is grouped by patient (`patient_nbr`), so no patient appears in both sets.
* Imputation, scaling and encoding are fitted inside scikit-learn pipelines on training data only.
* Results are reported at the default 0.50 threshold; other thresholds are explored descriptively only.
* Probabilities are not recalibrated. This is an academic analysis, not a clinical tool.

## Second analytical tool

The cleaned dataset was also analysed in Tableau Public (early-readmission rate by prior inpatient visits, age band and length of stay). Tableau reproduced the Python EDA results exactly. See Appendix I of the report.
