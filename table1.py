"""
table1.py
=========
Loads master_dataset.csv and prints:
  - Cohort exclusion flow
  - Table 1: demographics, labs, comorbidities, medications,
    smoking, PRS scores, MACE outcomes, exposome linkage
"""

import os
import pandas as pd
import numpy as np

WORKSPACE = '/home/dataproc/workspaces/geneexposome'

# ── Exclusion flow: count rows from each saved file cheaply ──────────────────
def nrows(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return sum(1 for _ in f) - 1

n_full       = nrows(f'{WORKSPACE}/cohort_skeleton_full.csv')
n_incident   = nrows(f'{WORKSPACE}/cohort_skeleton_incident.csv')
n_covariates = nrows(f'{WORKSPACE}/cohort_with_covariates.csv')

# ── Load only the columns needed for Table 1 ─────────────────────────────────
all_cols = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', nrows=0).columns.tolist()

CORE = [
    'person_id',
    'age_at_landmark', 'sex_at_birth', 'race', 'ethnicity',
    'bmi', 'sbp', 'dbp', 'chol_total', 'ldl', 'hdl',
    'hba1c', 'glucose', 'creatinine',
    'htn', 't2dm', 'obesity_dx', 'cad_prev', 'ckd',
    'statin', 'antihtn', 'metformin', 'antiplatelet', 'current_smoker',
    'cad_prs', 'prs_ldlc', 'prs_obesity', 'prs_sbp', 'prs_t2d',
    'mace3_event', 'has_mi', 'has_stroke', 'has_cvd_death', 'has_any_death',
    'zip3',
]

# One probe column per exposome prefix for linkage rates
PREFIXES = ['gee', 'social', 'noise', 'smart', 'toxins', 'wildfire']
probe_cols = {}
for pfx in PREFIXES:
    col = next((c for c in all_cols if c.startswith(f'{pfx}_')), None)
    if col:
        probe_cols[pfx] = col
        CORE.append(col)

use_cols = [c for c in CORE if c in all_cols]
master = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', usecols=use_cols,
                     dtype={'person_id': str})
for c in master.select_dtypes('float64').columns:
    master[c] = master[c].astype('float32')

N = len(master)

# ── Formatting helpers ────────────────────────────────────────────────────────
S  = "─" * 70
S2 = "═" * 70

def med_iqr(col):
    s = master[col].dropna()
    return f"{s.median():.1f}  [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]"

def miss(col):
    n = master[col].isna().sum()
    return f"  (missing {n:,}, {100*n/N:.1f}%)" if n > 0 else ""

def pct(col):
    n = int(master[col].sum())
    return f"{n:>7,}  ({100*n/N:5.1f}%)"

def cat_table(col):
    vc = master[col].value_counts(dropna=False)
    lines = []
    for v, c in vc.items():
        label = str(v) if pd.notna(v) else 'Missing'
        lines.append(f"    {label:<50} {c:>7,}  ({100*c/N:5.1f}%)")
    return "\n".join(lines)

def num_row(label, col):
    if col not in master.columns:
        return
    if pd.api.types.is_numeric_dtype(master[col]) and master[col].nunique() > 2:
        print(f"  {label:<42} {med_iqr(col)}{miss(col)}")
    else:
        top = master[col].value_counts().index[0] if master[col].notna().any() else 'N/A'
        print(f"  {label:<42} top='{top}'  n_unique={master[col].nunique()}{miss(col)}")


# ── Print exclusion flow ──────────────────────────────────────────────────────
print(S2)
print("  COHORT EXCLUSION FLOW")
print(S2)
if n_full:
    print(f"  All of Us participants with CAD PRS            {n_full:>7,}")
if n_full and n_incident:
    print(f"  − Prevalent MACE or age <18 before 2018-01-01  {n_full-n_incident:>7,}  (landmark exclusion)")
if n_incident:
    print(f"  = Incident cohort ≥18 at landmark              {n_incident:>7,}")
if n_covariates:
    print(f"  − Alaska / Hawaii / territories / no zip3      {n_covariates-N:>7,}  (no exposome data)")
print(f"  = FINAL ANALYSIS COHORT                        {N:>7,}")
print(S2)


# ── Table 1 ───────────────────────────────────────────────────────────────────
print(f"\n{S2}")
print(f"  TABLE 1 — COHORT CHARACTERISATION")
print(f"  N = {N:,}")
print(S2)

print(f"\nDEMOGRAPHICS\n{S}")
num_row("Age at landmark, median [IQR]", "age_at_landmark")
print(f"\n  Sex at birth:\n{cat_table('sex_at_birth')}")
print(f"\n  Race:\n{cat_table('race')}")
print(f"\n  Ethnicity:\n{cat_table('ethnicity')}")

print(f"\n\nCLINICAL FACTORS (most recent before 2018-01-01)\n{S}")
for col, label in [
    ('bmi',        'BMI, kg/m²'),
    ('sbp',        'Systolic BP, mmHg'),
    ('dbp',        'Diastolic BP, mmHg'),
    ('chol_total', 'Total cholesterol, mg/dL'),
    ('ldl',        'LDL cholesterol, mg/dL'),
    ('hdl',        'HDL cholesterol, mg/dL'),
    ('hba1c',      'HbA1c, %'),
    ('glucose',    'Fasting glucose, mg/dL'),
    ('creatinine', 'Creatinine, mg/dL'),
]:
    num_row(label, col)

print(f"\n\nCOMORBIDITIES (prevalent before 2018-01-01)\n{S}")
for col, label in [
    ('htn',        'Hypertension'),
    ('t2dm',       'Type 2 diabetes'),
    ('obesity_dx', 'Obesity (coded)'),
    ('cad_prev',   'Prior CAD'),
    ('ckd',        'Chronic kidney disease'),
]:
    if col in master.columns:
        print(f"  {label:<42} {pct(col)}")

print(f"\n\nMEDICATIONS (any use before 2018-01-01)\n{S}")
for col, label in [
    ('statin',      'Statin'),
    ('antihtn',     'Antihypertensive'),
    ('metformin',   'Metformin'),
    ('antiplatelet','Antiplatelet'),
]:
    if col in master.columns:
        print(f"  {label:<42} {pct(col)}")

print(f"\n\nSMOKING\n{S}")
if 'current_smoker' in master.columns:
    print(f"  Current smoker                             {pct('current_smoker')}")
    n_no_data = N - master['person_id'].isin(
        master.loc[master['current_smoker'].notna(), 'person_id']).sum()
    print(f"  (note: {N - int(master['current_smoker'].sum()):,} coded 0 = non/former smoker or no survey data)")

print(f"\n\nPOLYGENIC RISK SCORES (z-scored)\n{S}")
for col, label in [
    ('cad_prs',     'CAD PRS'),
    ('prs_ldlc',    'LDL-C PRS'),
    ('prs_obesity', 'Obesity PRS'),
    ('prs_sbp',     'SBP PRS'),
    ('prs_t2d',     'T2D PRS'),
]:
    num_row(label, col)

print(f"\n\nMACE OUTCOMES (incident, after 2018-01-01)\n{S}")
for col, label in [
    ('mace3_event',  '3-point MACE (MI + stroke + CVD death)'),
    ('has_mi',       'Myocardial infarction'),
    ('has_stroke',   'Stroke'),
    ('has_cvd_death','CVD death'),
    ('has_any_death','All-cause death'),
]:
    if col in master.columns:
        print(f"  {label:<42} {pct(col)}")

print(f"\n\nEXPOSOME LINKAGE\n{S}")
LABELS = {
    'gee': 'GEE (air quality / climate)',
    'social': 'Social exposome',
    'noise': 'Noise',
    'smart': 'SMART (built environment)',
    'toxins': 'Toxins',
    'wildfire': 'Wildfire smoke',
}
for pfx, probe in probe_cols.items():
    n = master[probe].notna().sum()
    print(f"  {LABELS.get(pfx,pfx):<42} {n:>7,}  ({100*n/N:5.1f}%)")

print(f"\n{S2}\n  END OF TABLE 1\n{S2}")
