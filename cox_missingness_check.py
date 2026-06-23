"""
cox_missingness_check.py
========================
Step 1 of fixing the Cox nested models.

Loads master_dataset.csv and prints a sorted missingness table for every
covariate used in M1–M7 across the full 380,780-participant cohort.

DO NOT refit any models until this output has been reviewed.
"""

import numpy as np
import pandas as pd

WORKSPACE = '/home/dataproc/workspaces/geneexposome'

# ── Exact column lists used in cox_nested_models.py ──────────────────────────
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
DEMO_EXTRA     = ['age_at_landmark', 'sex_at_birth', 'ethnicity', 'cad_prs']
SMOKE_COL      = ['current_smoker']

# Exposome prefixes
PHYS_PREFIXES = ('gee_', 'noise_', 'smart_', 'toxins_', 'wildfire_')
SOC_PREFIXES  = ('social_',)

# ── Load only the columns we care about ──────────────────────────────────────
print("Reading column list …")
all_cols = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', nrows=0).columns.tolist()

PHYS_COLS = [c for c in all_cols if c.startswith(PHYS_PREFIXES)]
SOC_COLS  = [c for c in all_cols if c.startswith(SOC_PREFIXES)]

WANT = list(dict.fromkeys(
    DEMO_EXTRA + SMOKE_COL
    + LAB_COLS + COMORBIDITY_COLS + MED_COLS + OTHER_PRS_COLS
    + PHYS_COLS + SOC_COLS
))
use_cols = [c for c in WANT if c in all_cols]

print(f"Loading {len(use_cols)} columns …")
df = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', usecols=use_cols,
                 dtype={'person_id': str})
N = len(df)
print(f"N = {N:,}\n")

# ── Missingness table ─────────────────────────────────────────────────────────
rows = []
for col in use_cols:
    if col not in df.columns:
        continue
    n_miss = int(df[col].isna().sum())
    rows.append({
        'column':    col,
        'domain':    ('Lab'           if col in LAB_COLS else
                      'Comorbidity'   if col in COMORBIDITY_COLS else
                      'Medication'    if col in MED_COLS else
                      'Smoking'       if col in SMOKE_COL else
                      'Other PRS'     if col in OTHER_PRS_COLS else
                      'Demo/PRS'      if col in DEMO_EXTRA else
                      'Phys exposome' if col.startswith(PHYS_PREFIXES) else
                      'Soc exposome'),
        'n_missing': n_miss,
        'pct_miss':  round(100 * n_miss / N, 1),
        'n_present': N - n_miss,
    })

miss_df = pd.DataFrame(rows).sort_values('pct_miss', ascending=False)

SEP = '─' * 72
print(f"\n{'='*72}")
print(f"  MISSINGNESS TABLE  (N = {N:,})")
print(f"{'='*72}")
print(f"  {'Column':<35} {'Domain':<15} {'N missing':>10}  {'% miss':>7}")
print(SEP)
for _, r in miss_df.iterrows():
    flag = '  *** >50%' if r['pct_miss'] > 50 else ('  * >10%' if r['pct_miss'] > 10 else '')
    print(f"  {r['column']:<35} {r['domain']:<15} {r['n_missing']:>10,}  {r['pct_miss']:>6.1f}%{flag}")

# ── Summary by domain ─────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  DOMAIN SUMMARY")
print(f"{'='*72}")
print(f"  {'Domain':<20} {'Cols':>5}  {'Cols >10% miss':>15}  {'Cols >50% miss':>15}")
print(SEP)
for domain in miss_df['domain'].unique():
    sub = miss_df[miss_df['domain'] == domain]
    print(f"  {domain:<20} {len(sub):>5}  {(sub['pct_miss']>10).sum():>15}  {(sub['pct_miss']>50).sum():>15}")

# ── Simulate analytic sample shrinkage ───────────────────────────────────────
# Show how many participants remain as we add each domain's columns via dropna
print(f"\n{'='*72}")
print("  COMPLETE-CASE SAMPLE SIZE IF dropna() APPLIED BY DOMAIN")
print(f"{'='*72}")

present = lambda lst: [c for c in lst if c in df.columns]

cumulative = df.copy()
for label, cols in [
    ('Demographics + PRS',     present(DEMO_EXTRA)),
    ('+ Labs',                 present(LAB_COLS)),
    ('+ Comorbidities',        present(COMORBIDITY_COLS)),
    ('+ Medications',          present(MED_COLS)),
    ('+ Smoking',              present(SMOKE_COL)),
    ('+ Other PRS',            present(OTHER_PRS_COLS)),
    ('+ Physical exposome',    present(PHYS_COLS)),
    ('+ Social exposome',      present(SOC_COLS)),
]:
    cumulative = cumulative.dropna(subset=cols)
    print(f"  {label:<30}  N = {len(cumulative):>7,}  ({100*len(cumulative)/N:.1f}%)")

print(f"\n{'='*72}")
print("  STOP — review the table above before refitting any models.")
print(f"{'='*72}")
