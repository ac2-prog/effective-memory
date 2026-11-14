"""Executable interactive portfolio dashboard. Run: streamlit run dashboard.py"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
st.set_page_config(page_title='RISKY | 30-Day Readmission',layout='wide')
st.title('RISKY | 30-Day Hospital Readmission Analytics')
st.caption('Academic demonstration only • Historical US diabetes encounters • Not a clinical decision tool')
metrics=pd.read_csv(OUT/'model_comparison.csv')
classes=pd.read_csv(OUT/'original_class_distribution.csv')
missing=pd.read_csv(OUT/'missing_values.csv')
preds=pd.read_csv(OUT/'test_predictions.csv')
t1,t2,t3=st.tabs(['Dataset & EDA','Model comparison','Threshold exploration'])
with t1:
 st.subheader('Original dataset and class imbalance')
 st.plotly_chart(px.bar(classes,x='outcome',y='count',title='Readmission outcomes'),use_container_width=True)
 st.dataframe(missing.head(12),use_container_width=True)
 st.image(str(OUT/'eda_age_readmission.png'));st.image(str(OUT/'eda_stay_readmission.png'))
with t2:
 st.subheader('All three models: held-out patient test set')
 st.dataframe(metrics.style.format({c:'{:.3f}' for c in metrics.columns if c!='model'}),use_container_width=True)
 st.plotly_chart(px.bar(metrics,x='model',y=['roc_auc','pr_auc_average_precision','recall','f1'],barmode='group',title='Compare discrimination and minority-class detection'),use_container_width=True)
 st.image(str(OUT/'model_roc_curves.png'));st.image(str(OUT/'model_precision_recall.png'))
with t3:
 st.subheader('Decision threshold and clinical trade-off')
 model=st.selectbox('Model',metrics['model'].tolist())
 threshold=st.slider('Classification threshold',0.05,0.95,0.50,0.05)
 from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
 import numpy as np
 truth=preds['true_label'].to_numpy(); prediction=(preds[model].to_numpy()>=threshold).astype(int)
 p=precision_score(truth,prediction,zero_division=0);r=recall_score(truth,prediction,zero_division=0);f=f1_score(truth,prediction,zero_division=0)
 c1,c2,c3=st.columns(3);c1.metric('Precision',f'{p:.3f}');c2.metric('Recall',f'{r:.3f}');c3.metric('F1',f'{f:.3f}')
 cm=confusion_matrix(truth,prediction)
 st.plotly_chart(px.imshow(cm,x=['No early readmission','<30 days'],y=['No early readmission','<30 days'],text_auto=True,labels={'x':'Predicted','y':'Actual'},title='Held-out confusion matrix'),use_container_width=True)
 st.info('Changing the threshold does not retrain models and does not establish clinical safety. False negatives and false positives have different consequences.')
