"""
CELL 7 — Table 1: cohort characterisation
==========================================
Loads master_dataset.csv and prints a clean demographic/clinical summary.
Run this as a standalone cell after 04_master_dataset.py has completed.
"""

import pandas as pd
import numpy as np
import os

WORKSPACE = '/home/dataproc/workspaces/geneexposome'

# ── Exclusion cascade (read only person_id — tiny memory footprint) ──────────
def nrows(path):
    if not os.path.exists(path):
        return None
    return len(pd.read_csv(path, usecols=['person_id']))

n_full      = nrows(f'{WORKSPACE}/cohort_skeleton_full.csv')
n_incident  = nrows(f'{WORKSPACE}/cohort_skeleton_incident.csv')
n_covariates= nrows(f'{WORKSPACE}/cohort_with_covariates.csv')

# Read only column names first (no data loaded)
all_cols = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', nrows=0).columns.tolist()

# Core columns needed for Table 1
CORE_COLS = [
    'person_id',
    'age_at_landmark', 'sex_at_birth', 'race', 'ethnicity',
    'bmi', 'sbp', 'dbp', 'chol_total', 'ldl', 'hdl',
    'hba1c', 'glucose', 'creatinine',
    'htn', 't2dm', 'obesity_dx', 'cad_prev', 'ckd',
    'statin', 'antihtn', 'metformin', 'antiplatelet', 'current_smoker',
    'cad_prs', 'prs_ldlc', 'prs_obesity', 'prs_sbp', 'prs_t2d',
    'mace3_event', 'has_mi', 'has_stroke', 'has_cvd_death', 'has_any_death',
]

# Add one probe column per exposome prefix to check linkage rate
PREFIXES = ['gee', 'social', 'noise', 'smart', 'toxins', 'wildfire']
probe_cols = {}
for pfx in PREFIXES:
    col = next((c for c in all_cols if c.startswith(f'{pfx}_')), None)
    if col:
        probe_cols[pfx] = col
        CORE_COLS.append(col)

# Load only what we need, downcast floats to save RAM
use_cols = [c for c in CORE_COLS if c in all_cols]
master = pd.read_csv(
    f'{WORKSPACE}/master_dataset.csv',
    usecols=use_cols,
    dtype={'person_id': str},
)
for c in master.select_dtypes('float64').columns:
    master[c] = master[c].astype('float32')

N = len(master)

# ── Print exclusion cascade ───────────────────────────────────────────────────
SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print("  COHORT EXCLUSION FLOW")
print(SEP2)
if n_full is not None:
    print(f"  All of Us participants with CAD PRS          {n_full:>7,}")
if n_full and n_incident:
    lost_prev = n_full - n_incident
    print(f"  − Prevalent MACE before 2018-01-01           {lost_prev:>7,}  (excluded: MACE before landmark)")
if n_incident is not None:
    print(f"  = Incident cohort (landmark 2018-01-01)      {n_incident:>7,}")
if n_incident and n_covariates and n_covariates != n_incident:
    lost_cov = n_incident - n_covariates
    print(f"  − Other exclusions (age <18 etc.)            {lost_cov:>7,}")
if n_covariates is not None:
    print(f"  = After covariate extraction                 {n_covariates:>7,}")
if n_covariates:
    lost_geo = n_covariates - N
    print(f"  − Alaska / Hawaii / territories / invalid    {lost_geo:>7,}  (no exposome coverage)")
print(f"  = FINAL ANALYSIS COHORT                      {N:>7,}")
print(SEP2)
print()

# ── helpers ──────────────────────────────────────────────────────────────────
def med_iqr(col):
    s = master[col].dropna()
    return f"{s.median():.1f}  [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]"

def pct(col, val=None):
    """% of total N (not of non-missing)."""
    s = master[col]
    if val is not None:
        n = (s == val).sum()
    else:
        n = s.sum()
    return f"{n:>7,}  ({100*n/N:5.1f}%)"

def cat(col):
    """Value counts as % of total N, sorted descending."""
    vc = master[col].value_counts(dropna=False)
    lines = []
    for v, c in vc.items():
        label = str(v) if pd.notna(v) else 'Missing'
        lines.append(f"    {label:<45} {c:>7,}  ({100*c/N:5.1f}%)")
    return "\n".join(lines)

def miss(col):
    n = master[col].isna().sum()
    return f"  (missing {n:,}, {100*n/N:.1f}%)"

SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print(f"  TABLE 1 — COHORT CHARACTERISATION")
print(f"  N = {N:,}")
print(SEP2)

# ── Demographics ─────────────────────────────────────────────────────────────
print(f"\n{'DEMOGRAPHICS':}")
print(SEP)
print(f"  Age at landmark, median [IQR]          {med_iqr('age_at_landmark')}{miss('age_at_landmark')}")

# Sex
print(f"\n  Sex at birth:")
print(cat('sex_at_birth'))

# Race
print(f"\n  Race:")
print(cat('race'))

# Ethnicity
print(f"\n  Ethnicity:")
print(cat('ethnicity'))

# ── Labs / vitals ─────────────────────────────────────────────────────────────
print(f"\n\n{'CLINICAL FACTORS (most recent before 2018-01-01)':}")
print(SEP)
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
    if col in master.columns:
        print(f"  {label:<40} {med_iqr(col)}{miss(col)}")

# ── Comorbidities ─────────────────────────────────────────────────────────────
print(f"\n\n{'COMORBIDITIES (prevalent before 2018-01-01)':}")
print(SEP)
for col, label in [
    ('htn',        'Hypertension'),
    ('t2dm',       'Type 2 diabetes'),
    ('obesity_dx', 'Obesity (coded)'),
    ('cad_prev',   'Prior CAD'),
    ('ckd',        'Chronic kidney disease'),
]:
    if col in master.columns:
        print(f"  {label:<40} {pct(col)}")

# ── Medications ───────────────────────────────────────────────────────────────
print(f"\n\n{'MEDICATIONS (any use before 2018-01-01)':}")
print(SEP)
for col, label in [
    ('statin',      'Statin'),
    ('antihtn',     'Antihypertensive'),
    ('metformin',   'Metformin'),
    ('antiplatelet','Antiplatelet'),
]:
    if col in master.columns:
        print(f"  {label:<40} {pct(col)}")

# ── Smoking ───────────────────────────────────────────────────────────────────
print(f"\n\n{'SMOKING':}")
print(SEP)
if 'current_smoker' in master.columns:
    print(f"  Current smoker                           {pct('current_smoker')}")

# ── PRS scores ────────────────────────────────────────────────────────────────
print(f"\n\n{'POLYGENIC RISK SCORES':}")
print(SEP)
for col, label in [
    ('cad_prs',     'CAD PRS'),
    ('prs_ldlc',    'LDL-C PRS'),
    ('prs_obesity', 'Obesity PRS'),
    ('prs_sbp',     'SBP PRS'),
    ('prs_t2d',     'T2D PRS'),
]:
    if col in master.columns:
        print(f"  {label:<40} {med_iqr(col)}{miss(col)}")

# ── MACE outcomes ─────────────────────────────────────────────────────────────
print(f"\n\n{'MACE OUTCOMES (incident, after 2018-01-01)':}")
print(SEP)
for col, label in [
    ('mace3_event',  '3-point MACE (MI + stroke + CVD death)'),
    ('has_mi',       'Myocardial infarction'),
    ('has_stroke',   'Stroke'),
    ('has_cvd_death','CVD death'),
    ('has_any_death','All-cause death'),
]:
    if col in master.columns:
        print(f"  {label:<40} {pct(col)}")

# ── Exposome coverage ─────────────────────────────────────────────────────────
print(f"\n\n{'EXPOSOME LINKAGE':}")
print(SEP)
LABELS = {
    'gee':      'GEE (air quality / climate)',
    'social':   'Social exposome',
    'noise':    'Noise',
    'smart':    'SMART (built environment)',
    'toxins':   'Toxins',
    'wildfire': 'Wildfire smoke',
}
for pfx, probe in probe_cols.items():
    n_linked = master[probe].notna().sum()
    print(f"  {LABELS.get(pfx, pfx):<40} {n_linked:>7,}  ({100*n_linked/N:5.1f}%)")

print(f"\n{SEP2}")
print(f"  END OF TABLE 1")
print(SEP2)
