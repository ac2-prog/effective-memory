"""Supplementary RISKY EDA and transparent data-quality audit. Author Karina Roade.
Run in the RISKY project root with: python eda_quality_audit.py
Do not interpret observational group differences as causal effects.
The model training pipelines in train_models.py remain the source of fitted imputations.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'eda_extended'
OUT.mkdir(parents=True,exist_ok=True)
CSV=ROOT/'data'/'diabetic_data.csv'

# keep_default_na=False: 'None' in max_glu_serum/A1Cresult means 'not tested' and is a valid category.
raw=pd.read_csv(CSV,keep_default_na=False,low_memory=False)
df=pd.read_csv(CSV,na_values=['?'],keep_default_na=False,low_memory=False)
assert df.shape==(101766,50), f'Unexpected source dimensions {df.shape}'
unknown_gender=int((df['gender']=='Unknown/Invalid').sum())

# 3 encounters coded 'Unknown/Invalid' in gender are treated as missing (imputed in-pipeline).
df['gender']=df['gender'].replace('Unknown/Invalid',pd.NA)
df['early_readmission']=(df['readmitted']=='<30').astype('int8')

# Data quality BEFORE feature-specific training imputation. Do not drop patients here.
records=[]
for c in raw.columns:
    records.append({'variable':c,
      'missing_before_questionmark_mapping':int(raw[c].isna().sum()),
      'questionmark_placeholders':int(raw[c].astype('string').eq('?').sum()),
      'missing_after_questionmark_mapping':int(df[c].isna().sum()),
      'percent_missing_after':round(float(df[c].isna().mean()*100),2),
      'unique_nonmissing_values':int(df[c].nunique(dropna=True))})
quality=pd.DataFrame(records).sort_values('percent_missing_after',ascending=False)
quality.to_csv(OUT/'data_quality_by_variable.csv',index=False)

summary=pd.DataFrame([{'stage':'Raw CSV','rows':len(raw),'columns':len(raw.columns),'questionmark_placeholders':int(raw.astype('string').eq('?').sum().sum()),'missing_cells':int(raw.isna().sum().sum()),'full_duplicate_rows':int(raw.duplicated().sum()),'duplicate_encounter_ids':int(raw['encounter_id'].duplicated().sum()),'distinct_patients':int(raw['patient_nbr'].nunique())},
{'stage':'After ? mapped to NaN and binary target derived','rows':len(df),'columns':len(df.columns),'questionmark_placeholders':0,'missing_cells':int(df.isna().sum().sum()),'full_duplicate_rows':int(df.drop(columns=['early_readmission']).duplicated().sum()),'duplicate_encounter_ids':int(df['encounter_id'].duplicated().sum()),'distinct_patients':int(df['patient_nbr'].nunique())}])
summary.to_csv(OUT/'quality_overview.csv',index=False)

# Transparent descriptive tables with group sample sizes, no predictive leakage.
def group_table(col):
    t=df.groupby(col,dropna=False,observed=True)['early_readmission'].agg(['size','sum','mean']).reset_index()
    t.columns=[col,'encounters','early_readmissions','early_readmission_rate']
    t.to_csv(OUT/f'eda_{col}_summary.csv',index=False)
    return t

age=group_table('age')
stay=group_table('time_in_hospital')
prior=group_table('number_inpatient')
med=group_table('num_medications')
# numerical distribution table
num_cols=['time_in_hospital','num_medications','num_lab_procedures','number_inpatient','number_diagnoses']
df[num_cols].describe(percentiles=[.25,.5,.75,.9,.95,.99]).T.to_csv(OUT/'numeric_distributions.csv')

fig,ax=plt.subplots(figsize=(8,4.3))
miss=quality.query('percent_missing_after > 0').head(12).iloc[::-1]
ax.barh(miss['variable'],miss['percent_missing_after'])
ax.set_xlabel('Missing values (%)');ax.set_title('Missingness after mapping ? to NaN');fig.tight_layout();fig.savefig(OUT/'fig_missingness.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.3))
ax.bar(age['age'].astype(str),age['early_readmission_rate']*100)
ax.set_ylabel('Early readmission (%)');ax.set_xlabel('Age band');ax.set_title('Observed early-readmission rate by age band');ax.tick_params(axis='x',rotation=35);fig.tight_layout();fig.savefig(OUT/'fig_age_rate.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.3))
ax.plot(stay['time_in_hospital'],stay['early_readmission_rate']*100,marker='o')
ax.set_ylabel('Early readmission (%)');ax.set_xlabel('Days in hospital');ax.set_title('Observed early-readmission rate by length of stay');fig.tight_layout();fig.savefig(OUT/'fig_stay_rate.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.3))
# group unusually high prior admission counts for stable sample-size comparisons
buckets=pd.cut(df['number_inpatient'],[-1,0,1,2,100],labels=['0','1','2','3+'])
t=df.groupby(buckets,observed=True)['early_readmission'].agg(['size','mean'])
ax.bar(t.index.astype(str),t['mean']*100)
ax.set_ylabel('Early readmission (%)');ax.set_xlabel('Prior inpatient visits');ax.set_title('Readmission by previous inpatient utilisation');fig.tight_layout();fig.savefig(OUT/'fig_prior_inpatient.png',dpi=170);plt.close(fig)
t.to_csv(OUT/'eda_prior_inpatient_buckets.csv')

print('QUALITY OVERVIEW')
print(summary.to_string(index=False))
print('\nMISSINGNESS TOP 10')
print(quality[['variable','questionmark_placeholders','percent_missing_after']].head(10).to_string(index=False))
print('\nGender Unknown/Invalid recoded to missing:',unknown_gender)
print('\nBINARY TARGET',df['early_readmission'].value_counts().to_dict())
print('\nOUTPUT',OUT)
