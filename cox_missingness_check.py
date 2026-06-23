"""
cox_missingness_check.py
========================
Prints a missingness table for all covariates used in cox_nested_models.py.
Memory-safe: computes null counts per column without copying the dataframe,
and simulates dropna shrinkage using boolean masks instead of repeated copies.
Exposome columns (gee_, noise_, smart_, toxins_, wildfire_, social_) are
excluded — we already confirmed ~100% linkage in the pipeline.
"""

import numpy as np
import pandas as pd

WORKSPACE = '/home/dataproc/workspaces/geneexposome'

LAB_COLS = [
    'bmi', 'sbp', 'dbp', 'chol_total', 'ldl', 'hdl', 'hba1c', 'glucose', 'creatinine',
]
COMORBIDITY_COLS = [
    'htn', 't2dm', 't1dm', 'obesity_dx', 'cad_prev', 'ckd', 'fh', 'pad',
    'metabolic_syndrome', 'osa', 'heart_failure', 'afib',
    'ra', 'sle', 'psoriasis', 'ibd', 'crohns', 'ulc_colitis', 'hiv',
    'hypothyroidism', 'hyperthyroidism', 'pcos', 'cushings', 'acromegaly',
    'cardiotoxic_chemo',
    'preeclampsia', 'gest_dm', 'preterm', 'preg_loss',
]
MED_COLS = [
    'statin', 'pcsk9i',
    'ace_inhibitor', 'arb', 'beta_blocker', 'ccb', 'diuretic', 'any_antihtn',
    'aspirin', 'p2y12', 'oral_anticoag', 'arni',
    'metformin', 'sglt2i', 'glp1ra', 'dpp4i', 'sulfonylurea', 'tzd', 'insulin_any',
    'any_dm_med',
]
OTHER_PRS_COLS = ['prs_ldlc', 'prs_obesity', 'prs_sbp', 'prs_t2d']
DEMO_COLS      = ['age_at_landmark', 'sex_at_birth', 'ethnicity', 'cad_prs']
SMOKE_COL      = ['current_smoker']

# All non-exposome covariate columns
ALL_CHECK_COLS = list(dict.fromkeys(
    DEMO_COLS + LAB_COLS + COMORBIDITY_COLS + MED_COLS + SMOKE_COL + OTHER_PRS_COLS
))

# ── Load only these columns (no exposome) ────────────────────────────────────
all_master_cols = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', nrows=0).columns.tolist()
use_cols = [c for c in ALL_CHECK_COLS if c in all_master_cols]

print(f"Loading {len(use_cols)} columns (no exposome) …")
df = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', usecols=use_cols,
                 dtype={c: 'float32' for c in LAB_COLS + OTHER_PRS_COLS if c in use_cols})
N = len(df)
print(f"N = {N:,}   RAM ≈ {df.memory_usage(deep=True).sum()/1e6:.0f} MB\n")

# ── Per-column missingness ────────────────────────────────────────────────────
rows = []
for col in use_cols:
    n_miss = int(df[col].isna().sum())
    domain = ('Lab'          if col in LAB_COLS else
              'Comorbidity'  if col in COMORBIDITY_COLS else
              'Medication'   if col in MED_COLS else
              'Smoking'      if col in SMOKE_COL else
              'Other PRS'    if col in OTHER_PRS_COLS else
              'Demo/PRS')
    rows.append({'column': col, 'domain': domain,
                 'n_missing': n_miss, 'pct_miss': round(100 * n_miss / N, 1)})

miss_df = pd.DataFrame(rows).sort_values('pct_miss', ascending=False)

SEP = '─' * 70
print(f"{'='*70}\n  MISSINGNESS TABLE  (N = {N:,})\n{'='*70}")
print(f"  {'Column':<35} {'Domain':<14} {'N miss':>9}  {'% miss':>7}")
print(SEP)
for _, r in miss_df.iterrows():
    flag = '  *** >50%' if r['pct_miss'] > 50 else ('  * >10%' if r['pct_miss'] > 10 else '')
    print(f"  {r['column']:<35} {r['domain']:<14} {r['n_missing']:>9,}  {r['pct_miss']:>6.1f}%{flag}")

# ── Domain summary ────────────────────────────────────────────────────────────
print(f"\n{'='*70}\n  DOMAIN SUMMARY\n{'='*70}")
print(f"  {'Domain':<18} {'Cols':>5}  {'>10% miss':>10}  {'>50% miss':>10}")
print(SEP)
for domain in ['Demo/PRS', 'Lab', 'Comorbidity', 'Medication', 'Smoking', 'Other PRS']:
    sub = miss_df[miss_df['domain'] == domain]
    if len(sub) == 0:
        continue
    print(f"  {domain:<18} {len(sub):>5}  {(sub['pct_miss']>10).sum():>10}  {(sub['pct_miss']>50).sum():>10}")

# ── Complete-case shrinkage — boolean masks, no copies ────────────────────────
print(f"\n{'='*70}\n  COMPLETE-CASE N IF dropna() APPLIED CUMULATIVELY\n{'='*70}")

def present(lst):
    return [c for c in lst if c in df.columns]

# Build one null-indicator array per column, accumulate a "any null so far" mask
complete_mask = pd.Series(True, index=df.index)   # starts: all rows OK

for label, cols in [
    ('Demographics + PRS',  present(DEMO_COLS)),
    ('+ Labs',              present(LAB_COLS)),
    ('+ Comorbidities',     present(COMORBIDITY_COLS)),
    ('+ Medications',       present(MED_COLS)),
    ('+ Smoking',           present(SMOKE_COL)),
    ('+ Other PRS',         present(OTHER_PRS_COLS)),
]:
    for col in cols:
        complete_mask &= df[col].notna()
    n_complete = int(complete_mask.sum())
    print(f"  {label:<30}  N = {n_complete:>7,}  ({100*n_complete/N:.1f}%)")

print(f"\n  Note: Exposome columns (~100% linked) not shown — add negligible missingness.")
print(f"\n{'='*70}\n  STOP — review before refitting models.\n{'='*70}")
