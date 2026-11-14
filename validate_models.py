"""Reproducible additional *training-only* patient-group cross validation for RISKY.

Run: python validate_models.py
This script does not overwrite the main RISKY test predictions or public dashboard.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, balanced_accuracy_score, brier_score_loss

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data'/'diabetic_data.csv'
OUT=ROOT/'validation_outputs'
OUT.mkdir(parents=True, exist_ok=True)
# keep_default_na=False keeps 'None' (lab test not performed) as a category, not missing.
df=pd.read_csv(SOURCE,na_values=['?'],keep_default_na=False,low_memory=False)
assert len(df)==101766
# 3 encounters coded 'Unknown/Invalid' in gender are treated as missing (imputed in-pipeline).
df['gender']=df['gender'].replace('Unknown/Invalid',np.nan)
assert set(df['readmitted'].dropna().unique())=={'<30','>30','NO'}
y=(df['readmitted']=='<30').astype(int)
excluded={'readmitted','encounter_id','patient_nbr','weight'}
X=df.drop(columns=list(excluded))
patients=df['patient_nbr']
tr,te=next(GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=42).split(X,y,groups=patients))
train_patients=set(patients.iloc[tr]); test_patients=set(patients.iloc[te]); assert train_patients.isdisjoint(test_patients)
Xtrain=X.iloc[tr].reset_index(drop=True)
ytrain=y.iloc[tr].reset_index(drop=True)
gtrain=patients.iloc[tr].reset_index(drop=True)
cat=Xtrain.select_dtypes(include=['object','string']).columns.tolist(); num=[c for c in Xtrain if c not in cat]

def prep_linear():
    return ColumnTransformer([
      ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),
      ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',min_frequency=1000))]),cat)])
def prep_ordinal():
    return ColumnTransformer([
      ('num',SimpleImputer(strategy='median'),num),
      ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),cat)])
def models():
    return {
      'Logistic Regression':Pipeline([('prep',prep_linear()),('model',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42))]),
      'Random Forest':Pipeline([('prep',prep_linear()),('model',RandomForestClassifier(n_estimators=35,max_depth=12,min_samples_leaf=8,class_weight='balanced_subsample',random_state=42,n_jobs=2))]),
      'Gradient Boosting':Pipeline([('prep',prep_ordinal()),('model',HistGradientBoostingClassifier(max_iter=65,max_leaf_nodes=12,l2_regularization=1.0,random_state=42))])}

rows=[]
cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
for fold,(it,iv) in enumerate(cv.split(Xtrain,ytrain,groups=gtrain),1):
    assert set(gtrain.iloc[it]).isdisjoint(set(gtrain.iloc[iv]))
    print(f'Fold {fold}: train={len(it)} validation={len(iv)} prevalence={ytrain.iloc[iv].mean():.4f}',flush=True)
    for name,model in models().items():
        print('  Fitting',name,flush=True)
        model.fit(Xtrain.iloc[it],ytrain.iloc[it])
        p=model.predict_proba(Xtrain.iloc[iv])[:,1]
        pred=(p>=.5).astype(int)
        rows.append(dict(fold=fold,model=name,n_validation=len(iv),n_positive=int(ytrain.iloc[iv].sum()),
             roc_auc=roc_auc_score(ytrain.iloc[iv],p),average_precision=average_precision_score(ytrain.iloc[iv],p),
             precision=precision_score(ytrain.iloc[iv],pred,zero_division=0),recall=recall_score(ytrain.iloc[iv],pred,zero_division=0),
             f1=f1_score(ytrain.iloc[iv],pred,zero_division=0),balanced_accuracy=balanced_accuracy_score(ytrain.iloc[iv],pred),
             brier=brier_score_loss(ytrain.iloc[iv],p)))
        pd.DataFrame(rows).to_csv(OUT/'training_group_cv_folds.csv',index=False)
result=pd.DataFrame(rows)
agg=result.groupby('model').agg(roc_auc_mean=('roc_auc','mean'),roc_auc_std=('roc_auc','std'),
     average_precision_mean=('average_precision','mean'),recall_mean=('recall','mean'),
     f1_mean=('f1','mean'),balanced_accuracy_mean=('balanced_accuracy','mean'),
     brier_mean=('brier','mean')).reset_index()
agg.to_csv(OUT/'training_group_cv_summary.csv',index=False)
checks={'n_original_encounters':len(df),'n_unique_patients':int(patients.nunique()),
 'n_training_encounters':len(tr),'n_test_encounters':len(te),'training_test_patient_overlap':len(train_patients&test_patients),
 'target_mapping':{'<30':1,'>30':0,'NO':0},'positive_rate_original':float(y.mean()),
 'validation':'StratifiedGroupKFold 3 folds, on training set only',
 'leakage_cautions':['discharge_disposition_id may be determined only at discharge; check prediction timing',
    'admission_source_id may not be available for every intended use case',
    'splitting by patient is not the same as hospital/time external validation',
    'do not choose hyperparameters or thresholds on held-out test outcomes']}
(OUT/'validation_methodology.json').write_text(json.dumps(checks,indent=2))
print('\nSummary\n',agg.to_string(index=False),flush=True)
