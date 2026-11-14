"""RISKY: transparent, reproducible binary 30-day readmission modelling.
Run from this folder: python train_models.py
Outputs are generated from the real dataset, not manually entered.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, average_precision_score,roc_auc_score,classification_report,confusion_matrix,ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay,precision_score,recall_score,f1_score,brier_score_loss)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
df = pd.read_csv(
    ROOT / 'data' / 'diabetic_data.csv',
    na_values=['?'],
    keep_default_na=False,
    low_memory=False
)

assert df.shape == (101766, 50)
# 3 encounters coded 'Unknown/Invalid' in gender are treated as missing (imputed in-pipeline).
df['gender'] = df['gender'].replace('Unknown/Invalid', np.nan)

# Original source distinguishes <30, >30 and NO. We explicitly map to 0/1.
y=(df['readmitted']=='<30').astype(int)

# encounter and patient identifiers are not predictive features. Patient ID is a group.
exclude={'readmitted','encounter_id','patient_nbr','weight'}
X=df.drop(columns=list(exclude))
cat=X.select_dtypes(include=['object','string']).columns.tolist()
num=[c for c in X.columns if c not in cat]

# Group splitting prevents the same patient from appearing in train and test sets.
tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42).split(X,y,groups=df['patient_nbr']))
Xtr,Xte=X.iloc[tr],X.iloc[te];ytr,yte=y.iloc[tr],y.iloc[te]

# OHE for nominal data; unknown values safely handled. Fit ONLY on training data.
linear_prep=ColumnTransformer([
 ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
 ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=1000))]),cat)
])

# For histogram boosting use ordinal encoding for nominal categories (not interpreted as
# ordered effects); ideally categorical-native modelling is a future extension.
tree_prep=ColumnTransformer([
 ('num',SimpleImputer(strategy='median'),num),
 ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),cat)
])
models={
 'Logistic Regression':Pipeline([('prep',linear_prep),('model',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42))]),
 'Random Forest':Pipeline([('prep',linear_prep),('model',RandomForestClassifier(n_estimators=35,max_depth=12,min_samples_leaf=8,class_weight='balanced_subsample',random_state=42,n_jobs=2))]),
 'Gradient Boosting':Pipeline([('prep',tree_prep),('model',HistGradientBoostingClassifier(max_iter=65,max_leaf_nodes=12,l2_regularization=1.0,random_state=42))])
}
print('Rows',len(df),'Train',len(tr),'Test',len(te),'Train prevalence',round(ytr.mean(),4),'Test prevalence',round(yte.mean(),4),flush=True)
missing=df.isna().sum().sort_values(ascending=False)
missing.head(20).rename('missing_count').to_csv(OUT/'missing_values.csv')
counts=pd.DataFrame({'outcome':['<30','>30','NO'],'count':[int((df.readmitted==v).sum()) for v in ['<30','>30','NO']]})
counts.to_csv(OUT/'original_class_distribution.csv',index=False)
fig,ax=plt.subplots(figsize=(6,3.8));ax.bar(counts['outcome'],counts['count']);ax.set_title('Original readmission classes');ax.set_ylabel('Hospital encounters');fig.tight_layout();fig.savefig(OUT/'eda_class_distribution.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));df.groupby('age')['readmitted'].apply(lambda x:(x=='<30').mean()).plot(kind='bar',ax=ax);ax.set_ylabel('Early readmission rate');ax.set_xlabel('Age band');ax.set_title('Early readmission proportion by age band');fig.tight_layout();fig.savefig(OUT/'eda_age_readmission.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));df.groupby('time_in_hospital')['readmitted'].apply(lambda x:(x=='<30').mean()).plot(ax=ax,marker='o');ax.set_ylabel('Early readmission rate');ax.set_xlabel('Hospital stay (days)');ax.set_title('Association of length of stay with readmission');fig.tight_layout();fig.savefig(OUT/'eda_stay_readmission.png',dpi=160);plt.close(fig)
results=[]; roc_data={}; proba_table=pd.DataFrame({'true_label':yte.values})
for name,pipe in models.items():
 print('Fitting',name,flush=True)
 pipe.fit(Xtr,ytr)
 prob=pipe.predict_proba(Xte)[:,1]; pred=(prob>=0.5).astype(int)
 metrics={'model':name,'accuracy':accuracy_score(yte,pred),'balanced_accuracy':balanced_accuracy_score(yte,pred),
          'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),
          'f1':f1_score(yte,pred,zero_division=0),'roc_auc':roc_auc_score(yte,prob),
          'pr_auc_average_precision':average_precision_score(yte,prob),'brier_score':brier_score_loss(yte,prob)}
 results.append(metrics);roc_data[name]=prob;proba_table[name]=prob
 ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=['No early readmission','<30 days']).plot(values_format='d',cmap='Blues')
 plt.title(name+' — test confusion matrix');plt.tight_layout();plt.savefig(OUT/(name.lower().replace(' ','_')+'_confusion.png'),dpi=150);plt.close()
 print({k:round(v,4) if isinstance(v,float) else v for k,v in metrics.items()},flush=True)
res=pd.DataFrame(results);res.to_csv(OUT/'model_comparison.csv',index=False);proba_table.to_csv(OUT/'test_predictions.csv',index=False)
fig,ax=plt.subplots(figsize=(7,5))
for name,p in roc_data.items():RocCurveDisplay.from_predictions(yte,p,name=name,ax=ax)
ax.set_title('Binary ROC on held-out patients');fig.tight_layout();fig.savefig(OUT/'model_roc_curves.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5))
for name,p in roc_data.items():PrecisionRecallDisplay.from_predictions(yte,p,name=name,ax=ax)
ax.set_title('Precision–recall on held-out patients');fig.tight_layout();fig.savefig(OUT/'model_precision_recall.png',dpi=160);plt.close(fig)
metadata={'data_rows':len(df),'train_rows':len(tr),'test_rows':len(te),'train_positive_rate':float(ytr.mean()),'test_positive_rate':float(yte.mean()),'target':'1 if readmitted is <30; otherwise 0','split':'GroupShuffleSplit by patient_nbr, random_state=42','excluded_features':sorted(exclude),'categorical_features':cat,'numerical_features':num,'threshold':0.5,'notes':'Observational historical dataset; metrics are not a clinical validation; no SHAP/Cox claims.'}
(OUT/'methodology.json').write_text(json.dumps(metadata,indent=2))
print('\nFINAL COMPARISON\n',res.to_string(index=False),flush=True)
