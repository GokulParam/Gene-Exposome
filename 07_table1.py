"""
CELL 7 — Table 1: cohort characterisation
==========================================
Loads master_dataset.csv and prints a clean demographic/clinical summary.
Run this as a standalone cell after 04_master_dataset.py has completed.
"""

import pandas as pd
import numpy as np

WORKSPACE = '/home/dataproc/workspaces/geneexposome'
master = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', dtype={'person_id': str},
                     low_memory=False)

N = len(master)

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
for prefix, label in [
    ('gee',      'GEE (air quality / climate)'),
    ('social',   'Social exposome'),
    ('noise',    'Noise'),
    ('smart',    'SMART (built environment)'),
    ('toxins',   'Toxins'),
    ('wildfire', 'Wildfire smoke'),
]:
    probe = next((c for c in master.columns if c.startswith(f'{prefix}_')), None)
    if probe:
        n_linked = master[probe].notna().sum()
        print(f"  {label:<40} {n_linked:>7,}  ({100*n_linked/N:5.1f}%)")

print(f"\n{SEP2}")
print(f"  END OF TABLE 1")
print(SEP2)
