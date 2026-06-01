"""
CELL 9b — Rebuild master_dataset.csv cleanly and apply fixes
=============================================================
The previous fix script bloated master_dataset.csv from 148 MB to
900 MB by converting all int columns to floats. This script:
  1. Rebuilds master from cohort_with_covariates.csv + PRS files +
     zip3 (from already-saved master) + exposome files
     (no BigQuery needed — all sources are local)
  2. Applies correct smoking fix (concept 40766307, no date filter)
  3. Z-scores PRS columns
  4. Saves compact master_dataset.csv (~150 MB)
"""

import os, gc, subprocess
import pandas as pd
import numpy as np
from google.cloud import bigquery

WORKSPACE    = '/home/dataproc/workspaces/geneexposome'
CDR          = 'wb-silky-artichoke-2408.C2024Q3R9'
BUCKET       = 'gs://rw-migration-aou-rw-6cad436b'
EXPOSOME_DIR = f'{WORKSPACE}/Exposome'
client       = bigquery.Client(project='wb-shining-lemon-5239')

EXCLUDE_ZIP3 = {
    '000','006','007','008','009',
    '967','968','969',
    '995','996','997','998','999',
}

exposome_files = {
    'GEE_Final_3Zip.csv':            'gee',
    'Zip3_Social_Exposome_Final.csv': 'social',
    'Noise_3Zip.csv':                 'noise',
    'SMART_3Zip.csv':                 'smart',
    'Toxins_3Zip.csv':                'toxins',
    'Wildfire_3Zip.csv':              'wildfire',
}

EXTRA_PRS = ['LDLC', 'OBESITY', 'SBP', 'T2D']


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load base cohort
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 1: Load cohort_with_covariates.csv")
master = pd.read_csv(f'{WORKSPACE}/cohort_with_covariates.csv',
                     dtype={'person_id': str})
print(f"  {len(master):,} rows")
gc.collect()

# ── Extra PRS ────────────────────────────────────────────────────────────────
print("\nSTEP 2: Merge extra PRS")
for trait in EXTRA_PRS:
    prs_dir = f'{WORKSPACE}/PRS_Scores/{trait}'
    files = [f for f in os.listdir(prs_dir) if not f.startswith('.')]
    path  = os.path.join(prs_dir, files[0])
    ext   = path.rsplit('.', 1)[-1].lower()
    tmp   = pd.read_csv(path) if ext in ('csv','txt') else pd.read_csv(path, sep='\t')
    tmp.columns = ['person_id', f'prs_{trait.lower()}']
    tmp['person_id'] = tmp['person_id'].astype(str)
    tmp[f'prs_{trait.lower()}'] = tmp[f'prs_{trait.lower()}'].astype('float32')
    master = master.merge(tmp, on='person_id', how='left')
    print(f"  {trait}: matched {master[f'prs_{trait.lower()}'].notna().sum():,}")
    del tmp; gc.collect()

# ── Zip3 (reuse from existing bloated master — just grab person_id+zip3) ────
print("\nSTEP 3: Get zip3 from existing master")
zip3_df = pd.read_csv(f'{WORKSPACE}/master_dataset.csv',
                      usecols=['person_id', 'zip3'],
                      dtype={'person_id': str, 'zip3': str})
master = master.merge(zip3_df, on='person_id', how='left')
del zip3_df; gc.collect()

# Exclude AK/HI/territories
zip3_key = (master['zip3'].astype(str).str.strip()
            .str.extract(r'(\d+)', expand=False).str.zfill(3))
n_before = len(master)
master = master[~zip3_key.isin(EXCLUDE_ZIP3)].copy()
print(f"  Excluded {n_before - len(master):,} AK/HI/territory rows")
print(f"  Final cohort: {len(master):,}")
gc.collect()

# ── Exposome files ────────────────────────────────────────────────────────────
print("\nSTEP 4: Merge exposome files")
master['_zip3_key'] = (master['zip3'].astype(str).str.strip()
                       .str.extract(r'(\d+)', expand=False).str.zfill(3))

for fname, prefix in exposome_files.items():
    fpath = os.path.join(EXPOSOME_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  {fname}: NOT FOUND"); continue
    exp_df  = pd.read_csv(fpath)
    zip_col = next((c for c in exp_df.columns if 'zip' in c.lower()), exp_df.columns[0])
    exp_df['_exp_zip'] = (exp_df[zip_col].astype(str).str.strip()
                          .str.extract(r'(\d+)', expand=False).str.zfill(3))
    exp_df = exp_df.drop(columns=[c for c in exp_df.columns
                                   if 'zip' in c.lower() and c != '_exp_zip'],
                         errors='ignore')
    exp_df = exp_df.rename(columns={c: f'{prefix}_{c}'
                                     for c in exp_df.columns if c != '_exp_zip'})
    for c in exp_df.select_dtypes('float64').columns:
        exp_df[c] = exp_df[c].astype('float32')
    master = master.merge(exp_df, left_on='_zip3_key', right_on='_exp_zip', how='left')
    master = master.drop(columns=['_exp_zip'], errors='ignore')
    print(f"  {prefix}: {master.shape[1]} cols total")
    del exp_df; gc.collect()

master = master.drop(columns=['_zip3_key'], errors='ignore')


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Smoking fix (concept 40766307, no date filter)
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Fix smoking variable")
q_smoke = f"""
WITH ranked AS (
    SELECT
        CAST(o.person_id AS STRING) AS person_id,
        c_val.concept_name AS answer,
        ROW_NUMBER() OVER (
            PARTITION BY o.person_id ORDER BY o.observation_date DESC
        ) AS rn
    FROM `{CDR}.observation` o
    LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
    WHERE o.observation_concept_id = 40766307
      AND o.value_as_concept_id IS NOT NULL
)
SELECT person_id, answer FROM ranked WHERE rn = 1
"""
smoke_df = client.query(q_smoke).to_dataframe()
smoke_df['person_id'] = smoke_df['person_id'].astype(str)
print(f"  People with smoking answer: {len(smoke_df):,}")
print(f"  Answer breakdown:")
print(smoke_df['answer'].value_counts().to_string())

current_ids = set(smoke_df.loc[
    smoke_df['answer'].str.contains('every day|some day', case=False, na=False),
    'person_id'])
master['current_smoker'] = master['person_id'].isin(current_ids).astype('int8')
n_current = master['current_smoker'].sum()
n_ans = master['person_id'].isin(smoke_df['person_id']).sum()
print(f"\n  Answered in cohort : {n_ans:,} ({100*n_ans/len(master):.1f}%)")
print(f"  Current smoker     : {n_current:,} ({100*n_current/len(master):.1f}%)")
del smoke_df, current_ids; gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Z-score PRS
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Z-score PRS columns")
prs_cols = [c for c in master.columns if c.startswith('prs_') or c == 'cad_prs']
for col in prs_cols:
    s  = master[col].dropna().astype('float64')
    mu, sd = float(s.mean()), float(s.std())
    master[col] = ((master[col].astype('float64') - mu) / sd).astype('float32')
    print(f"  {col:<20} mean={mu:>10.3f}  sd={sd:>8.3f}  → z-scored")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save compact CSV
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Save master_dataset.csv")
out_path = f'{WORKSPACE}/master_dataset.csv'
master.to_csv(out_path, index=False, float_format='%.6g')
size_mb = os.path.getsize(out_path) / 1e6
print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
del master; gc.collect()

result = subprocess.run(['gsutil', 'cp', out_path, f'{BUCKET}/master_dataset.csv'],
                        capture_output=True, text=True, timeout=300)
print(f"  {'Copied to bucket.' if result.returncode == 0 else f'gsutil failed: {result.stderr}'}")

print("\nDONE — re-run 07_table1.py to verify")
