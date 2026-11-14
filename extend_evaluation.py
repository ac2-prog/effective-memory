"""Reproducible post-hoc analysis on original held-out test predictions.
No test-set threshold selection for clinical use is claimed.
"""
from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'
preds=pd.read_csv(OUT/'test_predictions.csv'); y=preds['true_label'].to_numpy()
rows=[]
for name in ['Logistic Regression','Random Forest','Gradient Boosting']:
 p=preds[name].to_numpy()
 for t in [0.10,0.20,0.30,0.40,0.50]:
  z=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
  rows.append(dict(model=name,threshold=t,precision=precision_score(y,z,zero_division=0),recall=recall_score(y,z,zero_division=0),f1=f1_score(y,z,zero_division=0),tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn)))
pd.DataFrame(rows).to_csv(OUT/'threshold_sensitivity.csv',index=False)
fig,ax=plt.subplots(figsize=(7,5))
for name in ['Logistic Regression','Random Forest','Gradient Boosting']:
 p=preds[name].to_numpy(); obs,estimated=calibration_curve(y,p,n_bins=10,strategy='quantile');ax.plot(estimated,obs,marker='o',label=name)
ax.plot([0,1],[0,1],'--',label='Perfect calibration');ax.set_xlim(0,.7);ax.set_ylim(0,.7);ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed readmission frequency');ax.set_title('Test-set reliability diagram (10 quantile bins)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'model_calibration.png',dpi=170);plt.close(fig)
print('Saved threshold sensitivity and calibration figures')
