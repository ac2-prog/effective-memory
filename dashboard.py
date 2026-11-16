"""Executable interactive portfolio dashboard. Run: streamlit run dashboard.py. Author: Karina Roade

Every number shown is read from files written by the project scripts:
train_models.py (outputs/), eda_quality_audit.py (outputs/eda_extended/) and
extend_evaluation.py (outputs/model_calibration.png, outputs/threshold_sensitivity.csv).
The dashboard does not retrain models or recalibrate probabilities.
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
EXT = OUT / 'eda_extended'

st.set_page_config(page_title='RISKY | 30-Day Readmission', layout='wide')
st.title('RISKY | 30-Day Hospital Readmission Analytics')
st.caption('Academic demonstration only • Historical US diabetes encounters • Not a clinical decision tool')

metrics = pd.read_csv(OUT / 'model_comparison.csv')
classes = pd.read_csv(OUT / 'original_class_distribution.csv')
missing = pd.read_csv(OUT / 'missing_values.csv')
preds = pd.read_csv(OUT / 'test_predictions.csv')


def read_optional(path):
    """Return a DataFrame, or None (with a visible warning) if the file has not been generated."""
    if path.exists():
        return pd.read_csv(path)
    st.warning(f'Missing file: {path.relative_to(ROOT)}. Run the script that generates it.')
    return None


def show_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.warning(f'Missing figure: {path.relative_to(ROOT)}. Run the script that generates it.')


def notes(interpretation, limitation, source):
    """Standard block under every chart: what it shows, what it cannot show, where it comes from."""
    st.markdown(f'**Interpretation.** {interpretation}')
    st.markdown(f'**Limitation.** {limitation}')
    st.caption(f'Source: {source}')


t1, t2, t3, t4 = st.tabs(['Dataset & EDA', 'Model comparison', 'Threshold exploration', 'Calibration'])

# ---------------------------------------------------------------- Dataset & EDA
with t1:
    st.subheader('Original dataset and class imbalance')
    total = int(classes['count'].sum())
    early = int(classes.loc[classes['outcome'] == '<30', 'count'].iloc[0])
    st.plotly_chart(px.bar(classes, x='outcome', y='count', title='Readmission outcomes (original three classes)'),
                    use_container_width=True)
    notes(f'{early:,} of {total:,} encounters ({early / total:.2%}) were readmitted within 30 days, the positive '
          'class of the binary target (<30 = 1; >30 and NO = 0). A model predicting "no early readmission" for '
          f'every encounter would be about {1 - early / total:.0%} accurate while finding no positive case.',
          'Counts are encounters, not patients: one patient can contribute several encounters.',
          'outputs/original_class_distribution.csv (train_models.py)')

    st.subheader('Missing-data analysis')
    overview = read_optional(EXT / 'quality_overview.csv')
    if overview is not None:
        st.dataframe(overview, use_container_width=True, hide_index=True)
    st.dataframe(missing.rename(columns={missing.columns[0]: 'variable'}).head(12),
                 use_container_width=True, hide_index=True)
    show_image(EXT / 'fig_missingness.png')
    quality = read_optional(EXT / 'data_quality_by_variable.csv')
    if quality is not None:
        top = quality.sort_values('percent_missing_after', ascending=False).head(3)
        top_text = ', '.join(f"{r.variable} ({r.percent_missing_after:.2f}%)" for r in top.itertuples())
        notes(f'Missing values were recorded as "?" in the source file. The most incomplete variables are {top_text}. '
              'Weight is excluded from modelling; other gaps are imputed inside the training pipelines, fitted on '
              'training data only. "None" in the glucose-serum and HbA1c columns means the test was not performed '
              'and is kept as a category, not counted as missing. Three gender values coded "Unknown/Invalid" are '
              'treated as missing. The 1,652 encounters that ended in the patient\'s death are excluded from the '
              'patterns and models, because readmission is impossible for them; hospice discharges are kept.',
              'Missingness is unlikely to be random (for example, specialty or payer may be recorded differently '
              'between hospitals), so imputation may introduce bias.',
              'outputs/eda_extended/data_quality_by_variable.csv, quality_overview.csv, fig_missingness.png '
              '(eda_quality_audit.py)')

    st.subheader('Prior inpatient utilisation')
    show_image(EXT / 'fig_prior_inpatient.png')
    prior = read_optional(EXT / 'eda_prior_inpatient_buckets.csv')
    if prior is not None:
        prior = prior.rename(columns={'number_inpatient': 'prior inpatient visits', 'size': 'encounters',
                                      'mean': 'early readmission rate'})
        st.dataframe(prior.style.format({'encounters': '{:,}', 'early readmission rate': '{:.2%}'}),
                     use_container_width=True, hide_index=True)
        lo, hi = prior['early readmission rate'].iloc[0], prior['early readmission rate'].iloc[-1]
        notes(f'The observed early-readmission rate rises from {lo:.2%} with no prior inpatient visits to {hi:.2%} '
              'with three or more. Prior utilisation is the strongest descriptive pattern in the data.',
              'An association, not a causal effect: patients with more admissions are usually sicker, and severity '
              'is not fully measured. Visits of 3 or more are grouped to keep group sizes stable.',
              'outputs/eda_extended/eda_prior_inpatient_buckets.csv, fig_prior_inpatient.png (eda_quality_audit.py)')

    st.subheader('Age groups')
    show_image(EXT / 'fig_age_rate.png')
    age = read_optional(EXT / 'eda_age_summary.csv')
    if age is not None:
        st.dataframe(age.style.format({'encounters': '{:,}', 'early_readmissions': '{:,}',
                                       'early_readmission_rate': '{:.2%}'}),
                     use_container_width=True, hide_index=True)
        small = age.loc[age['encounters'] < 1000, 'age'].tolist()
        notes('Early-readmission rates vary only modestly across most adult age bands. The table shows the number '
              'of encounters behind each bar, so small groups can be identified.',
              'Age is recorded in 10-year bands, so no conclusion can be drawn about specific ages. '
              + (f'Bands with fewer than 1,000 encounters ({", ".join(small)}) give unstable rates. ' if small else '')
              + 'Differences may reflect health status rather than age itself.',
              'outputs/eda_extended/eda_age_summary.csv, fig_age_rate.png (eda_quality_audit.py)')

    st.subheader('Hospital length of stay')
    show_image(EXT / 'fig_stay_rate.png')
    stay = read_optional(EXT / 'eda_time_in_hospital_summary.csv')
    if stay is not None:
        st.dataframe(stay.style.format({'encounters': '{:,}', 'early_readmissions': '{:,}',
                                        'early_readmission_rate': '{:.2%}'}),
                     use_container_width=True, hide_index=True)
        first, last = stay.iloc[0], stay.iloc[-1]
        notes(f'The observed rate is {first.early_readmission_rate:.2%} for {int(first.time_in_hospital)}-day stays '
              f'and {last.early_readmission_rate:.2%} for {int(last.time_in_hospital)}-day stays. Rates rise fairly '
              'steadily up to about 10 days and then fluctuate, so the overall pattern is an association between '
              'longer stays and higher early-readmission rates, not a smooth trend.',
              'Length of stay is known only at discharge and is strongly linked to illness severity, so this is '
              'not evidence that longer stays cause readmission. Long-stay groups contain fewer encounters.',
              'outputs/eda_extended/eda_time_in_hospital_summary.csv, fig_stay_rate.png (eda_quality_audit.py)')

    with st.expander('Original figures from train_models.py'):
        st.image(str(OUT / 'eda_age_readmission.png'))
        st.image(str(OUT / 'eda_stay_readmission.png'))
        st.caption('Same associations as above, plotted as proportions by train_models.py.')

# ---------------------------------------------------------------- Model comparison (unchanged functionality)
with t2:
    st.subheader('All three models: held-out patient test set')
    st.dataframe(metrics.style.format({c: '{:.3f}' for c in metrics.columns if c != 'model'}), use_container_width=True)
    st.plotly_chart(px.bar(metrics, x='model', y=['roc_auc', 'pr_auc_average_precision', 'recall', 'f1'], barmode='group',
                           title='Compare discrimination and minority-class detection'), use_container_width=True)
    best = metrics.loc[metrics['roc_auc'].idxmax()]
    notes(f'{best.model} has the highest ROC-AUC ({best.roc_auc:.3f}), so it ranks encounters best, but at the 0.50 '
          f'threshold its recall is {best.recall:.3f}. Recall, precision and average precision matter more than '
          'accuracy because only about one encounter in nine is positive.',
          'One patient-grouped test split (20% of patients), threshold fixed at 0.50; no external validation on '
          'other hospitals or time periods.',
          'outputs/model_comparison.csv (train_models.py)')
    st.image(str(OUT / 'model_roc_curves.png'))
    notes('Curves above the diagonal show better-than-chance ranking; all three models give modest discrimination.',
          'ROC curves are insensitive to class imbalance and can look favourable even when few positives are detected.',
          'outputs/model_roc_curves.png (train_models.py)')
    st.image(str(OUT / 'model_precision_recall.png'))
    notes('Precision stays low across most recall levels, reflecting the difficulty of finding early readmissions.',
          'The chance level for precision equals the positive rate (about 0.11), not 0.5.',
          'outputs/model_precision_recall.png (train_models.py)')

# ---------------------------------------------------------------- Threshold exploration (unchanged functionality)
with t3:
    st.subheader('Decision threshold and clinical trade-off')
    model = st.selectbox('Model', metrics['model'].tolist())
    threshold = st.slider('Classification threshold', 0.05, 0.95, 0.50, 0.05)
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    truth = preds['true_label'].to_numpy(); prediction = (preds[model].to_numpy() >= threshold).astype(int)
    p = precision_score(truth, prediction, zero_division=0); r = recall_score(truth, prediction, zero_division=0); f = f1_score(truth, prediction, zero_division=0)
    c1, c2, c3 = st.columns(3); c1.metric('Precision', f'{p:.3f}'); c2.metric('Recall', f'{r:.3f}'); c3.metric('F1', f'{f:.3f}')
    cm = confusion_matrix(truth, prediction, labels=[0, 1])
    st.plotly_chart(px.imshow(cm, x=['No early readmission', '<30 days'], y=['No early readmission', '<30 days'], text_auto=True,
                              labels={'x': 'Predicted', 'y': 'Actual'}, title='Held-out confusion matrix'), use_container_width=True)
    notes('Lowering the threshold flags more encounters: recall rises and false positives rise with it.',
          'Thresholds are explored on the test set for illustration only; a deployable threshold would have to be '
          'chosen on separate validation data with clinical input.',
          'outputs/test_predictions.csv (train_models.py); metrics recomputed here for the selected threshold')
    st.info('Changing the threshold does not retrain models and does not establish clinical safety. False negatives and false positives have different consequences.')

# ---------------------------------------------------------------- Calibration
with t4:
    st.subheader('Calibration of predicted probabilities (held-out test set)')
    st.info('This section evaluates how well each model\'s predicted probabilities match observed readmission '
            'rates, using a reliability diagram and Brier scores. Recalibrating the probabilities (for example, '
            'Platt or isotonic scaling) is identified as the next step for future work.')
    show_image(OUT / 'model_calibration.png')
    notes('Gradient Boosting\'s points lie close to the diagonal, so its probabilities are roughly in line with '
          'observed rates. Logistic Regression and Random Forest lie well below it: they predict much higher risk '
          'than is observed, which is expected when models are trained with balanced class weights.',
          'Each model\'s points are 10 quantile bins of about 2,000 test encounters each, so the bins cover different '
          'probability ranges per model. The axes stop at 0.7, so Logistic Regression\'s highest bin is outside the '
          'plot. One test split only, with no confidence intervals.',
          'outputs/model_calibration.png (extend_evaluation.py)')

    prevalence = preds['true_label'].mean()
    reference = prevalence * (1 - prevalence)
    cal = metrics[['model', 'brier_score']].copy()
    cal['mean predicted probability'] = [preds[m].mean() for m in cal['model']]
    cal['observed early-readmission rate'] = prevalence
    st.dataframe(cal.style.format({'brier_score': '{:.3f}', 'mean predicted probability': '{:.3f}',
                                   'observed early-readmission rate': '{:.3f}'}),
                 use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(cal, x='model', y='brier_score', title='Brier score by model (lower is better)')
                    .add_hline(y=reference, line_dash='dash',
                               annotation_text=f'Always predicting the test prevalence: {reference:.3f}'),
                    use_container_width=True)
    notes(f'Brier scores are {", ".join(f"{r.model} {r.brier_score:.3f}" for r in cal.itertuples())}. '
          f'A constant prediction equal to the test prevalence ({prevalence:.3f}) would score {reference:.3f}. '
          'Logistic Regression and Random Forest score much worse than that reference because their average '
          'predicted probability is far above the observed rate; Gradient Boosting is only slightly better than it.',
          'A low Brier score does not mean useful detection: Gradient Boosting has the lowest score but finds very '
          'few readmissions at the 0.50 threshold. The Brier score mixes calibration and discrimination.',
          'brier_score from outputs/model_comparison.csv (train_models.py); mean predicted probability, observed '
          'rate and the reference line computed here from outputs/test_predictions.csv')
