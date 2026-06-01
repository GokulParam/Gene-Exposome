"""
CELL 9 — Fix smoking variable and z-score PRS columns
======================================================
1. Re-query smoking using correct concept 40766307
   ("Do you now smoke cigarettes every day, some days, or not at all")
   current_smoker = 1 if most recent answer before 2018-01-01 is
   "Every day" OR "Some days"
2. Z-score all five PRS columns so coefficients are on comparable scales
3. Overwrite master_dataset.csv and copy to bucket
"""

import os, gc, subprocess
import pandas as pd
import numpy as np
from google.cloud import bigquery

WORKSPACE = '/home/dataproc/workspaces/geneexposome'
CDR       = 'wb-silky-artichoke-2408.C2024Q3R9'
BUCKET    = 'gs://rw-migration-aou-rw-6cad436b'
LANDMARK  = '2018-01-01'
client    = bigquery.Client(project='wb-shining-lemon-5239')

# ── Load full master with float32 to halve RAM usage ─────────────────────────
print("Loading master_dataset.csv (float32) ...")
master = pd.read_csv(
    f'{WORKSPACE}/master_dataset.csv',
    dtype={'person_id': str},
    low_memory=False,
)
for c in master.select_dtypes('float64').columns:
    master[c] = master[c].astype('float32')
print(f"  {len(master):,} rows × {master.shape[1]} cols")
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# FIX 1 — Smoking
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FIX 1: Current smoker (concept 40766307)")
print("=" * 60)

# Pull most recent answer per person before landmark
q_smoke = f"""
WITH ranked AS (
    SELECT
        CAST(o.person_id AS STRING) AS person_id,
        c_val.concept_name AS answer,
        o.observation_date,
        ROW_NUMBER() OVER (
            PARTITION BY o.person_id
            ORDER BY o.observation_date DESC
        ) AS rn
    FROM `{CDR}.observation` o
    LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
    WHERE o.observation_concept_id = 40766307
      AND o.observation_date < '{LANDMARK}'
      AND o.value_as_concept_id IS NOT NULL
)
SELECT person_id, answer, observation_date
FROM ranked
WHERE rn = 1
"""
smoke_df = client.query(q_smoke).to_dataframe()
smoke_df['person_id'] = smoke_df['person_id'].astype(str)
print(f"\nPeople with smoking answer before landmark: {len(smoke_df):,}")
print("\nAnswer breakdown (most recent pre-landmark):")
print(smoke_df['answer'].value_counts().to_string())

# Current smoker = "Every day" or "Some days"
current_ids = set(
    smoke_df.loc[
        smoke_df['answer'].str.contains('every day|some day', case=False, na=False),
        'person_id'
    ]
)
master['current_smoker'] = master['person_id'].isin(current_ids).astype('int8')

n_current  = master['current_smoker'].sum()
n_answered = master['person_id'].isin(smoke_df['person_id']).sum()
print(f"\nUpdated current_smoker:")
print(f"  Answered before landmark : {n_answered:>7,}  ({100*n_answered/len(master):.1f}%)")
print(f"  Current smoker           : {n_current:>7,}  ({100*n_current/len(master):.1f}%)")
print(f"  Non/former smoker        : {n_answered-n_current:>7,}")
print(f"  No smoking data (→ 0)    : {len(master)-n_answered:>7,}")

del smoke_df, current_ids
gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# FIX 2 — Z-score PRS columns
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FIX 2: Z-score PRS columns")
print("=" * 60)

prs_cols = [c for c in master.columns if c.startswith('prs_') or c == 'cad_prs']
print(f"\n{'Column':<20} {'raw_mean':>12} {'raw_sd':>12}")
print("─" * 46)
for col in prs_cols:
    s  = master[col].dropna().astype('float64')  # float64 for precision during z-score
    mu = float(s.mean())
    sd = float(s.std())
    master[col] = ((master[col].astype('float64') - mu) / sd).astype('float32')
    print(f"  {col:<18} {mu:>12.3f} {sd:>12.3f}")

print("\nPost-z-score sanity check (mean≈0, sd≈1):")
for col in prs_cols:
    s = master[col].dropna()
    print(f"  {col:<20}  mean={s.mean():+.4f}  sd={s.std():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Saving master_dataset.csv")
print("=" * 60)

out_path = f'{WORKSPACE}/master_dataset.csv'
master.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / 1e6
print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")

del master
gc.collect()

result = subprocess.run(
    ['gsutil', 'cp', out_path, f'{BUCKET}/master_dataset.csv'],
    capture_output=True, text=True, timeout=300,
)
if result.returncode == 0:
    print(f"  Copied to bucket.")
else:
    print(f"  gsutil cp failed: {result.stderr}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
