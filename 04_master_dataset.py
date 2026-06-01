"""
CELL 5 — Assemble master dataset
=================================
Merges:
  1. cohort_with_covariates.csv  (clinical cohort + labs + meds + PRS_CAD)
  2. Extra PRS scores             (LDLC, OBESITY, SBP, T2D)
  3. zip3 linkage                 (from CDR: cb_search_person → person_ext → observation)
  4. Six exposome files           (merged one at a time to limit peak RAM)

Output: WORKSPACE/master_dataset.csv  +  copied to BUCKET

Memory strategy:
  - Float64 → float32, int64 → int32 after each merge
  - gc.collect() after every major step
  - Never keep two large copies of master in RAM simultaneously
  - Exposome files merged one at a time
"""

import os, gc, subprocess
import pandas as pd
import numpy as np
from google.cloud import bigquery

# ── Environment ───────────────────────────────────────────────────────────────
WORKSPACE = '/home/dataproc/workspaces/geneexposome'
CDR       = 'wb-silky-artichoke-2408.C2024Q3R9'
BUCKET    = 'gs://rw-migration-aou-rw-6cad436b'
client    = bigquery.Client(project='wb-shining-lemon-5239')

print(f"CDR:    {CDR}")
print(f"BUCKET: {BUCKET}")
print(f"WORKSPACE: {WORKSPACE}\n")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load cohort with covariates (53 MB on disk)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Load cohort_with_covariates.csv")
print("=" * 60)

cov_path = f'{WORKSPACE}/cohort_with_covariates.csv'
assert os.path.exists(cov_path), f"MISSING: {cov_path}"

# Load with reduced dtypes to limit RAM
master = pd.read_csv(cov_path, dtype={'person_id': str})

# Downcast numerics immediately
for col in master.select_dtypes(include='float64').columns:
    master[col] = master[col].astype('float32')
for col in master.select_dtypes(include='int64').columns:
    master[col] = master[col].astype('int32')

print(f"Loaded: {master.shape[0]:,} rows × {master.shape[1]} cols")
print(f"Columns: {list(master.columns)}")
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load extra PRS scores (LDLC, OBESITY, SBP, T2D)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Merge extra PRS scores")
print("=" * 60)

EXTRA_PRS = ['LDLC', 'OBESITY', 'SBP', 'T2D']

for trait in EXTRA_PRS:
    prs_dir = f'{WORKSPACE}/PRS_Scores/{trait}'
    if not os.path.isdir(prs_dir):
        print(f"  {trait}: directory missing — skipping")
        continue

    files = [f for f in os.listdir(prs_dir) if not f.startswith('.')]
    if not files:
        print(f"  {trait}: no files — skipping")
        continue

    prs_path = os.path.join(prs_dir, files[0])
    ext = prs_path.rsplit('.', 1)[-1].lower()
    if ext == 'parquet':
        prs_tmp = pd.read_parquet(prs_path)
    elif ext in ('csv', 'txt'):
        prs_tmp = pd.read_csv(prs_path)
    else:
        prs_tmp = pd.read_csv(prs_path, sep='\t')

    print(f"  {trait}: {prs_tmp.shape} — columns: {list(prs_tmp.columns)}")

    # Standardise: first col = person_id, second col = score
    prs_tmp.columns = [str(c) for c in prs_tmp.columns]
    pid_col   = prs_tmp.columns[0]
    score_col = prs_tmp.columns[1]
    prs_tmp = prs_tmp[[pid_col, score_col]].rename(columns={
        pid_col:   'person_id',
        score_col: f'prs_{trait.lower()}'
    })
    prs_tmp['person_id'] = prs_tmp['person_id'].astype(str)
    prs_tmp[f'prs_{trait.lower()}'] = prs_tmp[f'prs_{trait.lower()}'].astype('float32')

    before = len(master)
    master = master.merge(prs_tmp, on='person_id', how='left')
    after  = len(master)
    n_matched = master[f'prs_{trait.lower()}'].notna().sum()
    print(f"    Rows before/after: {before:,}/{after:,}  |  matched: {n_matched:,}")

    del prs_tmp
    gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Get zip3 per person from CDR
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Get zip3 from CDR")
print("=" * 60)

# Diagnostic confirmed: zip codes live in observation with concept_id 3043579
# ("Postal code [Location]") — 633k persons have it in this CDR.
# cb_search_person has no zip column; person_ext has no zip column.
# Full-table fetch, then filter to cohort in pandas (avoids 1MB query limit).
zip3_query_obs = f"""
SELECT CAST(o.person_id AS STRING) AS person_id,
       SUBSTR(o.value_as_string, 1, 3) AS zip3
FROM `{CDR}.observation` o
WHERE o.observation_concept_id = 3043579
  AND o.value_as_string IS NOT NULL
"""

zip3_df = None
for label, q in [('observation_3043579', zip3_query_obs)]:
    try:
        tmp = client.query(q).to_dataframe()
        tmp['person_id'] = tmp['person_id'].astype(str)
        # Filter to our cohort in pandas (no query size limit)
        tmp = tmp[tmp['person_id'].isin(master['person_id'])].copy()
        if len(tmp) > 0:
            zip3_df = tmp[['person_id', 'zip3']].drop_duplicates('person_id')
            print(f"  zip3 source: {label}  —  {len(zip3_df):,} unique persons matched")
            break
        else:
            print(f"  {label}: 0 rows matched cohort — trying next source")
    except Exception as e:
        print(f"  {label}: query failed ({e}) — trying next source")

if zip3_df is not None:
    master = master.merge(zip3_df, on='person_id', how='left')
    covered = master['zip3'].notna().sum()
    print(f"  zip3 coverage in master: {covered:,} / {len(master):,} "
          f"({100*covered/len(master):.1f}%)")
    del zip3_df
    gc.collect()
else:
    master['zip3'] = np.nan
    print("  zip3 column added as all-NaN.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3b — Exclude Alaska, Hawaii, territories, and invalid zip3s
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3b: Exclude AK / HI / territories / invalid zip3")
print("=" * 60)

EXCLUDE_ZIP3 = {
    '000',                          # invalid/unknown
    '006', '007', '008', '009',     # Puerto Rico
    '967', '968',                   # Hawaii
    '969',                          # Guam / Pacific territories
    '995', '996', '997', '998', '999',  # Alaska
}

zip3_key = (master['zip3'].astype(str).str.strip()
            .str.extract(r'(\d+)', expand=False).str.zfill(3))
excluded_mask = zip3_key.isin(EXCLUDE_ZIP3)
n_before = len(master)
master = master[~excluded_mask].copy()
gc.collect()
print(f"  Excluded: {n_before - len(master):,}  (AK/HI/territories/invalid)")
print(f"  Final cohort: {len(master):,}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Merge exposome files (one at a time)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Merge exposome files")
print("=" * 60)

# Create a stable zip3 key on master (do once, reuse for all files)
master['_zip3_key'] = (master['zip3']
                        .astype(str)
                        .str.strip()
                        .str.extract(r'(\d+)', expand=False)
                        .str.zfill(3))

EXPOSOME_DIR = f'{WORKSPACE}/Exposome'
exposome_files = {
    'GEE_Final_3Zip.csv':            'gee',
    'Zip3_Social_Exposome_Final.csv': 'social',
    'Noise_3Zip.csv':                 'noise',
    'SMART_3Zip.csv':                 'smart',
    'Toxins_3Zip.csv':                'toxins',
    'Wildfire_3Zip.csv':              'wildfire',
}

for fname, prefix in exposome_files.items():
    fpath = os.path.join(EXPOSOME_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  {fname}: NOT FOUND — skipping")
        continue

    exp_df = pd.read_csv(fpath)
    print(f"\n  {fname}  ({exp_df.shape[0]:,} rows, {exp_df.shape[1]} cols)")
    print(f"    Columns: {list(exp_df.columns[:8])}{'...' if exp_df.shape[1]>8 else ''}")

    # Detect the zip column (first column with 'zip' in the name, case-insensitive)
    zip_candidates = [c for c in exp_df.columns if 'zip' in c.lower()]
    if not zip_candidates:
        # Fall back to first column
        zip_col = exp_df.columns[0]
        print(f"    No 'zip' column found — using first column: {zip_col}")
    else:
        zip_col = zip_candidates[0]
        print(f"    Zip column: {zip_col}")

    # Standardise exposome zip key
    exp_df['_exp_zip'] = (exp_df[zip_col]
                           .astype(str)
                           .str.strip()
                           .str.extract(r'(\d+)', expand=False)
                           .str.zfill(3))

    # Drop ALL zip-like columns from exp_df — we use _exp_zip as the merge key.
    # This prevents 'zip3' in exp_df clashing with master['zip3'].
    all_zip_cols = [c for c in exp_df.columns
                    if 'zip' in c.lower() and c != '_exp_zip']
    exp_df = exp_df.drop(columns=all_zip_cols, errors='ignore')

    # Rename remaining data columns to avoid clashes with master columns
    non_key_cols = [c for c in exp_df.columns if c != '_exp_zip']
    rename_map   = {c: f'{prefix}_{c}' for c in non_key_cols}
    exp_df = exp_df.rename(columns=rename_map)

    # Downcast
    for col in exp_df.select_dtypes(include='float64').columns:
        exp_df[col] = exp_df[col].astype('float32')
    for col in exp_df.select_dtypes(include='int64').columns:
        exp_df[col] = exp_df[col].astype('int32')

    before_cols = master.shape[1]
    master = master.merge(exp_df, left_on='_zip3_key', right_on='_exp_zip', how='left')
    master = master.drop(columns=['_exp_zip'], errors='ignore')

    prefixed_cols = [c for c in master.columns if c.startswith(f'{prefix}_')]
    n_matched = master[prefixed_cols[0]].notna().sum() if prefixed_cols else 0
    print(f"    Rows: {len(master):,} | New cols added: {master.shape[1]-before_cols} "
          f"| Matched rows: {n_matched:,}")

    del exp_df
    gc.collect()

# Clean up the zip3 key column
master = master.drop(columns=['_zip3_key'], errors='ignore')
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save master dataset
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Save master dataset")
print("=" * 60)

out_path = f'{WORKSPACE}/master_dataset.csv'
master.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / 1e6
print(f"Saved to: {out_path}  ({size_mb:.1f} MB)")

# Copy to bucket
bucket_path = f'{BUCKET}/master_dataset.csv'
result = subprocess.run(
    ['gsutil', 'cp', out_path, bucket_path],
    capture_output=True, text=True, timeout=300
)
if result.returncode == 0:
    print(f"Copied to: {bucket_path}")
else:
    print(f"gsutil cp failed: {result.stderr}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Condensed summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MASTER DATASET SUMMARY")
print("=" * 60)

N = len(master)
print(f"\nTotal rows:    {N:,}")
print(f"Total columns: {master.shape[1]}")

# Helper: summarise a block of columns
def _block(label, cols):
    cols = [c for c in cols if c in master.columns]
    if not cols:
        return
    n_missing = master[cols].isna().any(axis=1).sum()
    print(f"\n  [{label}]  {len(cols)} variables  |  "
          f"rows with ≥1 missing: {n_missing:,} ({100*n_missing/N:.1f}%)")
    for c in cols[:5]:
        s = master[c]
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique() <= 2:
                print(f"    {c:<35} mean={s.mean():.3f}  miss={s.isna().mean():.1%}")
            else:
                print(f"    {c:<35} median={s.median():.2f}  "
                      f"IQR=[{s.quantile(.25):.2f},{s.quantile(.75):.2f}]  "
                      f"miss={s.isna().mean():.1%}")
        else:
            top = s.value_counts().index[0] if s.notna().any() else 'N/A'
            print(f"    {c:<35} top='{top}'  n_unique={s.nunique()}  "
                  f"miss={s.isna().mean():.1%}")
    if len(cols) > 5:
        print(f"    ... ({len(cols)-5} more not shown)")

# Clinical block
clinical = ['age_at_landmark', 'sex_at_birth', 'race', 'ethnicity',
            'bmi', 'sbp', 'dbp', 'chol_total', 'ldl', 'hdl',
            'hba1c', 'glucose', 'creatinine']
_block('Demographics + Labs', clinical)

# Outcomes
outcomes = ['mace3_event', 'has_mi', 'has_stroke', 'has_cvd_death',
            'has_any_death', 'mace3_date']
_block('Outcomes', outcomes)

# Comorbidities + meds
comorbid = ['htn', 't2dm', 'obesity_dx', 'cad_prev', 'ckd',
            'statin', 'antihtn', 'metformin', 'antiplatelet', 'current_smoker']
_block('Comorbidities + Meds', comorbid)

# PRS
prs_cols = [c for c in master.columns if c.startswith('prs_') or c == 'cad_prs']
_block('PRS scores', prs_cols)

# zip3
print(f"\n  [ZIP3]  coverage: {master['zip3'].notna().sum():,} / {N:,} "
      f"({100*master['zip3'].notna().mean():.1f}%)")

# Exposome blocks
for prefix in ['gee', 'social', 'noise', 'smart', 'toxins', 'wildfire']:
    cols = [c for c in master.columns if c.startswith(f'{prefix}_')]
    _block(f'Exposome:{prefix}', cols)

print("\n" + "=" * 60)
print("CELL 5 COMPLETE")
print("=" * 60)
