"""RISKY: create the cleaned dataset used for analysis. Author: Karina Roade

Run from the project folder:  python make_clean_dataset.py
Input : data/diabetic_data.csv          (original UCI file, 101,766 encounters x 50 columns)
Output: data/diabetic_data_cleaned.csv  (cleaned file, same encounters)
        outputs/cleaning_log.csv        (what each cleaning step changed)

Cleaning steps (identical to the ones applied inside train_models.py):
  1. "?" placeholders are converted to missing values (NaN).
  2. "None" in max_glu_serum and A1Cresult means "test not performed"; it is kept as a
     category, so pandas' default conversion of the text "None" to NaN is switched off.
  3. gender "Unknown/Invalid" (3 encounters) is recoded as missing.
  4. weight is dropped: 96.86% of its values are missing.
  5. The binary target early_readmission is added: 1 if readmitted == "<30", else 0.
     The original three-class column readmitted is kept for reference.

Imputation is NOT done here on purpose. Filling missing values using the whole dataset would
let information from the test set leak into training, so imputers are fitted inside the
model pipelines on the training data only (see train_models.py).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data' / 'diabetic_data.csv'
CLEAN = ROOT / 'data' / 'diabetic_data_cleaned.csv'
LOG = ROOT / 'outputs' / 'cleaning_log.csv'


def load_raw(path=SOURCE):
    """Read the original CSV with '?' treated as missing and 'None' kept as text."""
    return pd.read_csv(path, na_values=['?'], keep_default_na=False, low_memory=False)


def clean(df):
    """Apply the cleaning steps listed in the module docstring and return (clean_df, log)."""
    log = []
    df = df.copy()
    log.append({'step': '"?" converted to missing', 'detail': f'{int(df.isna().sum().sum()):,} cells now missing'})

    unknown = int((df['gender'] == 'Unknown/Invalid').sum())
    df['gender'] = df['gender'].replace('Unknown/Invalid', pd.NA)
    log.append({'step': 'gender "Unknown/Invalid" recoded as missing', 'detail': f'{unknown} encounters'})

    weight_missing = df['weight'].isna().mean() * 100
    df = df.drop(columns=['weight'])
    log.append({'step': 'weight column dropped', 'detail': f'{weight_missing:.2f}% missing'})

    df['early_readmission'] = (df['readmitted'] == '<30').astype('int8')
    log.append({'step': 'binary target added (<30 = 1; >30 and NO = 0)',
                'detail': f"{int(df['early_readmission'].sum()):,} positives ({df['early_readmission'].mean():.2%})"})

    log.append({'step': 'duplicate checks',
                'detail': f'{int(df.duplicated().sum())} duplicate rows; '
                          f"{int(df['encounter_id'].duplicated().sum())} duplicate encounter IDs; "
                          f"{df['patient_nbr'].nunique():,} distinct patients (repeat patients handled by a patient-grouped split)"})
    return df, pd.DataFrame(log)


if __name__ == '__main__':
    raw = load_raw()
    assert raw.shape == (101766, 50), f'Unexpected source dimensions {raw.shape}'
    cleaned, log = clean(raw)
    LOG.parent.mkdir(exist_ok=True)
    cleaned.to_csv(CLEAN, index=False)
    log.to_csv(LOG, index=False)
    print(log.to_string(index=False))
    print(f'\nSaved {CLEAN.relative_to(ROOT)}: {cleaned.shape[0]:,} rows x {cleaned.shape[1]} columns')
