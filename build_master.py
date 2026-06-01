"""
build_master.py
===============
Builds master_dataset.csv from scratch.

Steps:
  1. Load cohort_with_covariates.csv (clinical cohort + CAD PRS)
  2. Merge extra PRS scores (LDLC, OBESITY, SBP, T2D)
  3. Get zip3 per person from CDR (observation concept 3043579)
  4. Exclude Alaska, Hawaii, territories, invalid zip3, and no-zip3 rows
  5. Merge 6 exposome files via zip3
  6. Fix smoking variable (concept 40766307, most recent answer)
  7. Z-score all PRS columns
  8. Save master_dataset.csv and copy to bucket
"""

import os, gc, subprocess
import pandas as pd
import numpy as np
from google.cloud import bigquery

# ── Config ────────────────────────────────────────────────────────────────────
WORKSPACE    = '/home/dataproc/workspaces/geneexposome'
CDR          = 'wb-silky-artichoke-2408.C2024Q3R9'
BUCKET       = 'gs://rw-migration-aou-rw-6cad436b'
EXPOSOME_DIR = f'{WORKSPACE}/Exposome'
client       = bigquery.Client(project='wb-shining-lemon-5239')

EXCLUDE_ZIP3 = {
    '000',                               # invalid
    '006', '007', '008', '009',          # Puerto Rico
    '967', '968',                        # Hawaii
    '969',                               # Guam / Pacific territories
    '995', '996', '997', '998', '999',   # Alaska
}

EXPOSOME_FILES = {
    'GEE_Final_3Zip.csv':            'gee',
    'Zip3_Social_Exposome_Final.csv': 'social',
    'Noise_3Zip.csv':                 'noise',
    'SMART_3Zip.csv':                 'smart',
    'Toxins_3Zip.csv':                'toxins',
    'Wildfire_3Zip.csv':              'wildfire',
}

EXTRA_PRS = ['LDLC', 'OBESITY', 'SBP', 'T2D']

BINARY_COLS = [
    'mace3_event', 'has_mi', 'has_stroke', 'has_cvd_death', 'has_any_death',
    'htn', 't2dm', 'obesity_dx', 'cad_prev', 'ckd',
    'statin', 'antihtn', 'metformin', 'antiplatelet', 'current_smoker',
]


def sep(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ── STEP 1: Load base cohort ──────────────────────────────────────────────────
sep("STEP 1: Load cohort_with_covariates.csv")
master = pd.read_csv(f'{WORKSPACE}/cohort_with_covariates.csv',
                     dtype={'person_id': str})
print(f"  {len(master):,} rows × {master.shape[1]} cols")
gc.collect()


# ── STEP 2: Extra PRS ─────────────────────────────────────────────────────────
sep("STEP 2: Merge extra PRS (LDLC, OBESITY, SBP, T2D)")
for trait in EXTRA_PRS:
    d = f'{WORKSPACE}/PRS_Scores/{trait}'
    f = [x for x in os.listdir(d) if not x.startswith('.')][0]
    path = os.path.join(d, f)
    tmp = pd.read_csv(path) if path.endswith(('.csv', '.txt')) else pd.read_csv(path, sep='\t')
    tmp.columns = ['person_id', f'prs_{trait.lower()}']
    tmp['person_id'] = tmp['person_id'].astype(str)
    master = master.merge(tmp, on='person_id', how='left')
    print(f"  {trait}: {master[f'prs_{trait.lower()}'].notna().sum():,} matched")
    del tmp; gc.collect()


# ── STEP 3: Get zip3 from CDR ─────────────────────────────────────────────────
sep("STEP 3: Get zip3 per person (observation concept 3043579)")
q = f"""
WITH ranked AS (
    SELECT
        CAST(o.person_id AS STRING) AS person_id,
        SUBSTR(o.value_as_string, 1, 3) AS zip3,
        ROW_NUMBER() OVER (PARTITION BY o.person_id ORDER BY o.observation_date DESC) AS rn
    FROM `{CDR}.observation` o
    WHERE o.observation_concept_id = 3043579
      AND o.value_as_string IS NOT NULL
)
SELECT person_id, zip3 FROM ranked WHERE rn = 1
"""
zip3_df = client.query(q).to_dataframe()
zip3_df['person_id'] = zip3_df['person_id'].astype(str)
zip3_df = zip3_df.drop_duplicates('person_id')
print(f"  zip3 fetched for {len(zip3_df):,} persons in CDR")

master = master.merge(zip3_df, on='person_id', how='left')
print(f"  zip3 coverage in cohort: {master['zip3'].notna().sum():,} / {len(master):,} "
      f"({100*master['zip3'].notna().mean():.1f}%)")
del zip3_df; gc.collect()


# ── STEP 4: Geographic exclusion ─────────────────────────────────────────────
sep("STEP 4: Exclude AK / HI / territories / missing zip3")
zip3_key = (master['zip3'].astype(str).str.strip()
            .str.extract(r'(\d+)', expand=False).str.zfill(3))
exclude = zip3_key.isin(EXCLUDE_ZIP3) | master['zip3'].isna()
n_before = len(master)
master = master[~exclude].copy()
print(f"  Excluded: {n_before - len(master):,}")
print(f"  Final cohort: {len(master):,}")
gc.collect()


# ── STEP 5: Merge exposome files ──────────────────────────────────────────────
sep("STEP 5: Merge exposome files")
master['_z3'] = (master['zip3'].astype(str).str.strip()
                 .str.extract(r'(\d+)', expand=False).str.zfill(3))

for fname, pfx in EXPOSOME_FILES.items():
    fpath = os.path.join(EXPOSOME_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  {fname}: NOT FOUND — skipping"); continue
    exp = pd.read_csv(fpath)
    zcol = next((c for c in exp.columns if 'zip' in c.lower()), exp.columns[0])
    exp['_ez'] = (exp[zcol].astype(str).str.strip()
                  .str.extract(r'(\d+)', expand=False).str.zfill(3))
    exp = exp.drop(columns=[c for c in exp.columns if 'zip' in c.lower() and c != '_ez'],
                   errors='ignore')
    exp = exp.rename(columns={c: f'{pfx}_{c}' for c in exp.columns if c != '_ez'})
    for c in exp.select_dtypes('float64').columns:
        exp[c] = exp[c].astype('float32')
    master = master.merge(exp, left_on='_z3', right_on='_ez', how='left')
    master = master.drop(columns=['_ez'], errors='ignore')
    n_matched = master[next(c for c in master.columns if c.startswith(f'{pfx}_'))].notna().sum()
    print(f"  {pfx}: {n_matched:,} / {len(master):,} rows matched "
          f"({100*n_matched/len(master):.1f}%)")
    del exp; gc.collect()

master = master.drop(columns=['_z3'], errors='ignore')


# ── STEP 6: Smoking fix ───────────────────────────────────────────────────────
sep("STEP 6: Fix smoking (concept 40766307)")
q_smoke = f"""
WITH ranked AS (
    SELECT
        CAST(o.person_id AS STRING) AS person_id,
        c_val.concept_name AS answer,
        ROW_NUMBER() OVER (PARTITION BY o.person_id ORDER BY o.observation_date DESC) AS rn
    FROM `{CDR}.observation` o
    LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
    WHERE o.observation_concept_id = 40766307
      AND o.value_as_concept_id IS NOT NULL
)
SELECT person_id, answer FROM ranked WHERE rn = 1
"""
smoke = client.query(q_smoke).to_dataframe()
smoke['person_id'] = smoke['person_id'].astype(str)
print(f"  People with answer: {len(smoke):,}")
print(smoke['answer'].value_counts().to_string())

current_ids = set(smoke.loc[
    smoke['answer'].str.contains('every day|some day', case=False, na=False), 'person_id'])
master['current_smoker'] = master['person_id'].isin(current_ids).astype(int)
n_ans = master['person_id'].isin(smoke['person_id']).sum()
n_cur = master['current_smoker'].sum()
print(f"\n  Answered (in cohort): {n_ans:,} ({100*n_ans/len(master):.1f}%)")
print(f"  Current smoker:       {n_cur:,} ({100*n_cur/len(master):.1f}%)")
del smoke, current_ids; gc.collect()


# ── STEP 7: Z-score PRS ───────────────────────────────────────────────────────
sep("STEP 7: Z-score PRS columns")
prs_cols = [c for c in master.columns if c.startswith('prs_') or c == 'cad_prs']
for col in prs_cols:
    s = master[col].dropna().astype('float64')
    mu, sd = float(s.mean()), float(s.std())
    master[col] = ((master[col].astype('float64') - mu) / sd).round(5).astype('float32')
    print(f"  {col:<20} raw mean={mu:>10.3f}  sd={sd:>8.3f}  → z-scored")


# ── STEP 8: Save ──────────────────────────────────────────────────────────────
sep("STEP 8: Save master_dataset.csv")

# Ensure binary columns are written as integers (not floats)
for c in BINARY_COLS:
    if c in master.columns:
        master[c] = master[c].fillna(0).astype(int)

# Round remaining floats to 4 dp to keep file size reasonable
for c in master.select_dtypes(include=['float32', 'float64']).columns:
    master[c] = master[c].round(4)

out_path = f'{WORKSPACE}/master_dataset.csv'
master.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / 1e6
print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
print(f"  Shape: {master.shape}")

del master; gc.collect()

result = subprocess.run(['gsutil', 'cp', out_path, f'{BUCKET}/master_dataset.csv'],
                        capture_output=True, text=True, timeout=300)
print(f"  {'Copied to bucket.' if result.returncode == 0 else f'gsutil failed: {result.stderr}'}")

print("\nDONE — run table1.py to see the summary")
