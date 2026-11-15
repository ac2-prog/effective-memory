"""RISKY: post-hoc evaluation of the saved held-out test predictions. Author: Karina Roade

Run from the project folder AFTER train_models.py:  python extend_evaluation.py
Input : outputs/test_predictions.csv  (true label + predicted probability per model)
Output: outputs/threshold_sensitivity.csv  (precision, recall, F1 and confusion counts per threshold)
        outputs/model_calibration.png      (reliability diagram, 10 quantile bins)

Nothing is retrained. Thresholds are explored on the test set for description only; no threshold
is selected for clinical use, and the probabilities are not recalibrated.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # write figures to files without opening a window
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
MODELS = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

preds = pd.read_csv(OUT / 'test_predictions.csv')
y = preds['true_label'].to_numpy()

# 1) Threshold sensitivity: how precision/recall and the four confusion counts change
#    when the decision threshold moves away from the default 0.50.
rows = []
for name in MODELS:
    p = preds[name].to_numpy()
    for t in THRESHOLDS:
        z = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, z, labels=[0, 1]).ravel()
        rows.append(dict(model=name, threshold=t,
                         precision=precision_score(y, z, zero_division=0),
                         recall=recall_score(y, z, zero_division=0),
                         f1=f1_score(y, z, zero_division=0),
                         tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn)))
pd.DataFrame(rows).to_csv(OUT / 'threshold_sensitivity.csv', index=False)

# 2) Reliability diagram: mean predicted probability vs observed readmission rate in
#    10 quantile bins (about 2,000 test encounters per bin) for each model.
fig, ax = plt.subplots(figsize=(7, 5))
for name in MODELS:
    p = preds[name].to_numpy()
    observed, estimated = calibration_curve(y, p, n_bins=10, strategy='quantile')
    ax.plot(estimated, observed, marker='o', label=name)
ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
ax.set_xlim(0, .7)
ax.set_ylim(0, .7)
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Observed readmission frequency')
ax.set_title('Test-set reliability diagram (10 quantile bins)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'model_calibration.png', dpi=170)
plt.close(fig)

print('Saved threshold sensitivity and calibration figures')
