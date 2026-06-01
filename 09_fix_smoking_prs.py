"""
CELL 9 — Fix smoking variable and z-score PRS columns
======================================================
Memory-safe: uses chunked CSV processing so the full 148 MB file is
never loaded into RAM at once.

Steps:
  1. Query CDR for correct smoking status (concept 40766307)
  2. Compute PRS z-score params from a tiny single-column pass
  3. Stream master_dataset.csv through in 50k-row chunks, patching
     current_smoker and PRS columns, writing to a new file
  4. Replace original and copy to bucket
"""

import os, gc, subprocess
import pandas as pd
import numpy as np
from google.cloud import bigquery

WORKSPACE  = '/home/dataproc/workspaces/geneexposome'
CDR        = 'wb-silky-artichoke-2408.C2024Q3R9'
BUCKET     = 'gs://rw-migration-aou-rw-6cad436b'
LANDMARK   = '2018-01-01'
CHUNKSIZE  = 50_000
client     = bigquery.Client(project='wb-shining-lemon-5239')

IN_PATH    = f'{WORKSPACE}/master_dataset.csv'
OUT_PATH   = f'{WORKSPACE}/master_dataset_fixed.csv'


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Get correct smoking status from CDR
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Query smoking status (concept 40766307)")
print("=" * 60)

q_smoke = f"""
WITH ranked AS (
    SELECT
        CAST(o.person_id AS STRING) AS person_id,
        c_val.concept_name AS answer,
        ROW_NUMBER() OVER (
            PARTITION BY o.person_id
            ORDER BY o.observation_date DESC
        ) AS rn
    FROM `{CDR}.observation` o
    LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
    WHERE o.observation_concept_id = 40766307
      AND o.value_as_concept_id IS NOT NULL
)
SELECT person_id, answer
FROM ranked WHERE rn = 1
"""
smoke_df = client.query(q_smoke).to_dataframe()
smoke_df['person_id'] = smoke_df['person_id'].astype(str)
print(f"\nPeople with any smoking answer: {len(smoke_df):,}")
print("\nAnswer breakdown:")
print(smoke_df['answer'].value_counts().to_string())

# Build lookup: person_id → 0/1
smoke_df['current_smoker_new'] = smoke_df['answer'].str.contains(
    'every day|some day', case=False, na=False
).astype('int8')

smoking_map = smoke_df.set_index('person_id')['current_smoker_new']
n_current = smoking_map.sum()
print(f"\nCurrent smokers (every day + some days): {n_current:,} "
      f"({100*n_current/len(smoke_df):.1f}% of those who answered)")

del smoke_df
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute PRS z-score parameters (tiny pass — 6 cols only)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Compute PRS z-score parameters")
print("=" * 60)

all_cols = pd.read_csv(IN_PATH, nrows=0).columns.tolist()
prs_cols = [c for c in all_cols if c.startswith('prs_') or c == 'cad_prs']
print(f"PRS columns: {prs_cols}")

prs_df = pd.read_csv(IN_PATH, usecols=prs_cols, dtype='float32')
prs_params = {}
for col in prs_cols:
    s = prs_df[col].dropna().astype('float64')
    prs_params[col] = (float(s.mean()), float(s.std()))
    print(f"  {col:<20}  mean={prs_params[col][0]:>12.3f}  sd={prs_params[col][1]:>10.3f}")

del prs_df
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Stream CSV in chunks, patch columns, write new file
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Patch master_dataset.csv in chunks")
print("=" * 60)

# Read original dtypes from first chunk to restore them after patching
dtype_sample = pd.read_csv(IN_PATH, nrows=100, dtype={'person_id': str},
                           low_memory=False)
int_cols  = dtype_sample.select_dtypes('int64').columns.tolist()
bool_cols = dtype_sample.select_dtypes('bool').columns.tolist()
del dtype_sample

reader    = pd.read_csv(IN_PATH, dtype={'person_id': str}, chunksize=CHUNKSIZE,
                        low_memory=False)
first     = True
rows_done = 0

for chunk in reader:
    # Patch current_smoker
    new_val = chunk['person_id'].map(smoking_map)
    chunk['current_smoker'] = new_val.fillna(0).astype('int64')

    # Z-score PRS (keep as float, 6 sig figs is plenty)
    for col in prs_cols:
        mu, sd = prs_params[col]
        chunk[col] = (chunk[col].astype('float64') - mu) / sd

    # Restore integer types so they write as 0/1 not 0.0/1.0
    for c in int_cols:
        if c in chunk.columns and c != 'current_smoker':
            chunk[c] = chunk[c].astype('Int64')  # nullable int handles NaN

    # Write with limited float precision to keep file size reasonable
    chunk.to_csv(OUT_PATH, mode='w' if first else 'a',
                 header=first, index=False, float_format='%.6g')
    first      = False
    rows_done += len(chunk)
    print(f"  {rows_done:,} rows written ...", end='\r')

print(f"\n  Done. {rows_done:,} rows total.")

del smoking_map, prs_params
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Replace original file and copy to bucket
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Replace original and upload to bucket")
print("=" * 60)

os.replace(OUT_PATH, IN_PATH)
size_mb = os.path.getsize(IN_PATH) / 1e6
print(f"  master_dataset.csv updated  ({size_mb:.1f} MB)")

result = subprocess.run(
    ['gsutil', 'cp', IN_PATH, f'{BUCKET}/master_dataset.csv'],
    capture_output=True, text=True, timeout=300,
)
if result.returncode == 0:
    print(f"  Copied to bucket.")
else:
    print(f"  gsutil cp failed: {result.stderr}")

print("\n" + "=" * 60)
print("DONE — re-run 07_table1.py to verify")
print("=" * 60)
