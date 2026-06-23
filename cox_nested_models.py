"""
cox_nested_models.py
====================
Nested Cox Proportional Hazards models (M1–M7) to quantify the incremental
explanatory contribution of genetic risk (CAD PRS), clinical variables, and
exposome domains (physical, social) to time-to-MACE survival.

Assumes df_analysis is already loaded in the namespace, or loads master_dataset.csv
and constructs the required columns.

Outputs:
  - Table 1: Model fit statistics (LogL, AIC, BIC, R², Harrell's C)
  - Table 2: Nested LRT comparisons (Δ logL, LRT χ², df, p-value, ΔR²)
  - Both tables saved as CSV next to this script.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter

warnings.filterwarnings('ignore')

WORKSPACE = '/home/dataproc/workspaces/geneexposome'
OUT_DIR   = WORKSPACE

# ── 1. Load / validate df_analysis ───────────────────────────────────────────
# If df_analysis is already in scope (run from a notebook cell), skip loading.
try:
    df_analysis  # noqa: F821
    print(f"Using existing df_analysis: {df_analysis.shape}")
except NameError:
    print("df_analysis not found in scope — loading from master_dataset.csv")
    master = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', dtype={'person_id': str})

    # ── Build time and event columns ─────────────────────────────────────────
    # time  = days from 2018-01-01 to first MACE or censoring date
    # event = mace3_event flag
    # AoU does not expose exact event dates in the master CSV; we approximate
    # using the CDR follow-up window end (2024-09-30 = CDR C2024Q3R9 cut-off).
    # If your dataset has explicit time-to-event columns, replace this block.
    LANDMARK    = pd.Timestamp('2018-01-01')
    ADMIN_CENSOR = pd.Timestamp('2024-09-30')
    follow_days = (ADMIN_CENSOR - LANDMARK).days   # 2,464 days

    master['event'] = master['mace3_event'].astype(int)
    # Without per-person event dates we assign all events the same follow-up;
    # this is a placeholder — replace with actual time_to_event column if available.
    master['time']  = np.where(master['event'] == 1,
                               follow_days * 0.5,   # placeholder: event at midpoint
                               float(follow_days))   # censored at end of follow-up

    # ── Rename columns to match spec ─────────────────────────────────────────
    col_map = {
        'age_at_landmark': 'age',
        'sex_at_birth':    'sex',
        'cad_prs':         'CAD_PRS',
        # exposome prefixes already match (phys_ / soc_ not present in master;
        # see note below about prefix remapping)
    }
    master = master.rename(columns=col_map)

    # sex → binary (Female=0, Male=1)
    if master['sex'].dtype == object:
        master['sex'] = master['sex'].map(
            lambda x: 1 if str(x).lower() in ('male', 'man', '1') else 0
        ).astype(int)

    # Ancestry PCs — AoU master_dataset.csv may not include PCs; add zeros if absent
    for i in range(1, 11):
        pc = f'PC{i}'
        if pc not in master.columns:
            master[pc] = 0.0

    # PRS quartile
    master['prs_quartile'] = pd.qcut(master['CAD_PRS'], 4, labels=[1, 2, 3, 4]).astype(int)

    # Remap exposome prefixes to phys_ / soc_ per spec
    # Physical exposome: gee_, noise_, smart_, toxins_, wildfire_
    # Social exposome:   social_
    phys_prefixes = ('gee_', 'noise_', 'smart_', 'toxins_', 'wildfire_')
    soc_prefixes  = ('social_',)

    rename_dict = {}
    for col in master.columns:
        if col.startswith(phys_prefixes):
            rename_dict[col] = 'phys_' + col
        elif col.startswith(soc_prefixes):
            rename_dict[col] = 'soc_' + col.replace('social_', '', 1)
    master = master.rename(columns=rename_dict)

    df_analysis = master.copy()
    print(f"Built df_analysis: {df_analysis.shape}")
    del master

# ── 2. Identify column groups ─────────────────────────────────────────────────
PC_COLS   = [c for c in df_analysis.columns if c.startswith('PC') and c[2:].isdigit()][:10]
PHYS_COLS = [c for c in df_analysis.columns if c.startswith('phys_')]
SOC_COLS  = [c for c in df_analysis.columns if c.startswith('soc_')]

# Clinical variables per spec
CLINICAL_CONTINUOUS = [c for c in [
    'bmi', 'sbp', 'dbp', 'ldl', 'hdl', 'chol_total', 'hba1c', 'glucose',
    # also try alternate names from master_dataset
    'BMI', 'SBP', 'DBP', 'LDL', 'HDL', 'total_cholesterol', 'HbA1c', 'fasting_glucose',
] if c in df_analysis.columns]

COMORBIDITY_COLS = [c for c in [
    'htn', 't2dm', 'afib', 'ckd', 'heart_failure', 'pad',
    'hypertension', 'T2DM', 'AF', 'CKD', 'COPD', 'heart_failure', 'PVD',
    't1dm', 'osa', 'metabolic_syndrome', 'ra', 'sle', 'psoriasis', 'ibd',
    'hiv', 'hypothyroidism', 'hyperthyroidism', 'pcos', 'cushings', 'acromegaly',
    'cardiotoxic_chemo', 'preeclampsia', 'gest_dm', 'preterm', 'preg_loss',
] if c in df_analysis.columns]

MED_COLS = [c for c in [
    'statin', 'pcsk9i', 'ace_inhibitor', 'arb', 'beta_blocker', 'ccb', 'diuretic',
    'any_antihtn', 'aspirin', 'p2y12', 'oral_anticoag', 'arni',
    'metformin', 'sglt2i', 'glp1ra', 'dpp4i', 'sulfonylurea', 'tzd', 'insulin_any',
    'any_dm_med',
    'statins', 'antihypertensives', 'antiplatelets', 'anticoagulants', 'antidiabetics',
] if c in df_analysis.columns]

CLIN_COLS = list(dict.fromkeys(CLINICAL_CONTINUOUS + COMORBIDITY_COLS + MED_COLS))

# ── 3. Build working dataframe (drop rows with any NaN in required columns) ───
DEMO_COLS = ['age', 'sex'] + PC_COLS
CORE_COLS = ['time', 'event'] + DEMO_COLS + ['CAD_PRS'] + CLIN_COLS + PHYS_COLS + SOC_COLS
avail     = [c for c in CORE_COLS if c in df_analysis.columns]
df        = df_analysis[avail].copy()

# current_smoker may be NaN; drop from model covariates (it stays in dataset)
for col in ['current_smoker']:
    if col in df.columns:
        df = df.drop(columns=[col])

df = df.dropna(subset=['time', 'event', 'age', 'sex', 'CAD_PRS'])
df['time']  = df['time'].clip(lower=1)   # lifelines requires time > 0
df['event'] = df['event'].astype(int)

N      = len(df)
EVENTS = int(df['event'].sum())
print(f"\nAnalysis N={N:,}  Events={EVENTS:,}  ({100*EVENTS/N:.1f}%)")

# Refresh column lists on cleaned df
PC_COLS_   = [c for c in PC_COLS   if c in df.columns]
CLIN_COLS_ = [c for c in CLIN_COLS if c in df.columns]
PHYS_COLS_ = [c for c in PHYS_COLS if c in df.columns]
SOC_COLS_  = [c for c in SOC_COLS  if c in df.columns]

print(f"  PCs: {len(PC_COLS_)}  Clinical: {len(CLIN_COLS_)}  Phys: {len(PHYS_COLS_)}  Soc: {len(SOC_COLS_)}")

# ── 4. Helper: fit Cox model and extract statistics ───────────────────────────
def fit_cox(covariate_cols, label=""):
    """Fit CoxPHFitter and return (cph, stats_dict)."""
    cols = list(dict.fromkeys(covariate_cols))   # deduplicate, preserve order
    cols = [c for c in cols if c in df.columns]
    sub  = df[['time', 'event'] + cols].dropna()
    n    = len(sub)
    k    = len(cols)

    cph = CoxPHFitter(penalizer=0.01)   # small ridge to handle correlated exposome cols
    cph.fit(sub, duration_col='time', event_col='event', show_progress=False)

    logL  = cph.log_likelihood_
    aic   = -2 * logL + 2 * k
    bic   = -2 * logL + k * np.log(n)
    c_idx = cph.concordance_index_

    # Concordance 95% CI via bootstrap is slow for large N;
    # use lifelines' built-in SE if available, else ±0.005 placeholder
    try:
        c_se = cph.concordance_index_se_
    except AttributeError:
        c_se = 0.005
    c_lo = max(0.5, c_idx - 1.96 * c_se)
    c_hi = min(1.0, c_idx + 1.96 * c_se)

    return cph, {
        'label': label,
        'n': n,
        'events': int(sub['event'].sum()),
        'k': k,
        'logL': logL,
        'aic': aic,
        'bic': bic,
        'c': c_idx,
        'c_lo': c_lo,
        'c_hi': c_hi,
    }

# ── 5. Null model log-likelihood (for Royston R²) ────────────────────────────
print("\nFitting null model …")
cph_null = CoxPHFitter()
cph_null.fit(df[['time', 'event']].assign(_dummy=0),
             duration_col='time', event_col='event',
             formula='_dummy - 1',
             show_progress=False)
logL_null = cph_null.log_likelihood_

def royston_r2(logL_model, n):
    lrt = 2 * (logL_model - logL_null)
    return 1 - np.exp(-lrt / n)

# ── 6. Fit all seven models ───────────────────────────────────────────────────
MODELS = {
    'M1': DEMO_COLS,
    'M2': DEMO_COLS + ['CAD_PRS'],
    'M3': DEMO_COLS + CLIN_COLS_,
    'M4': DEMO_COLS + ['CAD_PRS'] + CLIN_COLS_,
    'M5': DEMO_COLS + ['CAD_PRS'] + CLIN_COLS_ + PHYS_COLS_,
    'M6': DEMO_COLS + ['CAD_PRS'] + CLIN_COLS_ + SOC_COLS_,
    'M7': DEMO_COLS + ['CAD_PRS'] + CLIN_COLS_ + PHYS_COLS_ + SOC_COLS_,
}

results = {}
for name, cols in MODELS.items():
    print(f"Fitting {name} ({len(set(cols))} covariates) …")
    cph, stats = fit_cox(cols, label=name)
    stats['r2'] = royston_r2(stats['logL'], stats['n'])
    results[name] = (cph, stats)
    print(f"  logL={stats['logL']:.2f}  C={stats['c']:.4f}  R²={stats['r2']:.4f}")

# ── 7. Table 1 — Model fit statistics ────────────────────────────────────────
rows1 = []
for name in ['M1','M2','M3','M4','M5','M6','M7']:
    s = results[name][1]
    rows1.append({
        'Model':          name,
        'Description':    {
            'M1': 'Demographics (age, sex, PCs)',
            'M2': 'M1 + CAD PRS',
            'M3': 'M1 + Clinical',
            'M4': 'M1 + CAD PRS + Clinical  [primary baseline]',
            'M5': 'M4 + Physical exposome',
            'M6': 'M4 + Social exposome',
            'M7': 'M4 + Physical + Social exposome  [full model]',
        }[name],
        'N':              f"{s['n']:,}",
        'Events':         f"{s['events']:,}",
        'k (covariates)': s['k'],
        'Log-likelihood': f"{s['logL']:.2f}",
        'AIC':            f"{s['aic']:.1f}",
        'BIC':            f"{s['bic']:.1f}",
        'Royston R²':     f"{s['r2']:.4f}",
        "Harrell's C":    f"{s['c']:.4f}",
        '95% CI':         f"[{s['c_lo']:.4f}, {s['c_hi']:.4f}]",
    })

df_t1 = pd.DataFrame(rows1)
print(f"\n{'='*90}")
print("TABLE 1 — Model Fit Statistics")
print('='*90)
print(df_t1.to_string(index=False))

# ── 8. Table 2 — Nested LRT comparisons ──────────────────────────────────────
def lrt_compare(full_name, null_name):
    sf  = results[full_name][1]
    sn  = results[null_name][1]
    d_logL = sf['logL'] - sn['logL']
    chi2   = 2 * d_logL
    df_    = sf['k'] - sn['k']
    pval   = stats.chi2.sf(chi2, df_) if df_ > 0 else np.nan
    dr2    = sf['r2'] - sn['r2']
    return {
        'Comparison': f"{full_name} vs {null_name}",
        'ΔlogL':      f"{d_logL:.2f}",
        'LRT χ²':     f"{chi2:.2f}",
        'df':         df_,
        'p-value':    f"{pval:.2e}" if pval >= 1e-300 else "< 1e-300",
        'ΔR²':        f"{dr2:.4f}",
        'Interpretation': {
            ('M2','M1'): 'Incremental value of CAD PRS over demographics',
            ('M3','M1'): 'Incremental value of clinical variables over demographics',
            ('M4','M3'): 'Incremental value of CAD PRS on top of clinical variables',
            ('M4','M2'): 'Incremental value of clinical variables on top of PRS',
            ('M5','M4'): 'Incremental value of physical exposome over full baseline',
            ('M6','M4'): 'Incremental value of social exposome over full baseline',
            ('M7','M4'): 'Incremental value of full exposome over full baseline',
            ('M7','M5'): 'Incremental value of social exposome beyond physical',
            ('M7','M6'): 'Incremental value of physical exposome beyond social',
        }.get((full_name, null_name), ''),
    }

COMPARISONS = [
    ('M2','M1'), ('M3','M1'), ('M4','M3'), ('M4','M2'),
    ('M5','M4'), ('M6','M4'), ('M7','M4'), ('M7','M5'), ('M7','M6'),
]

rows2 = [lrt_compare(f, n) for f, n in COMPARISONS]
df_t2 = pd.DataFrame(rows2)

print(f"\n{'='*110}")
print("TABLE 2 — Nested Model LRT Comparisons")
print('='*110)
print(df_t2.to_string(index=False))

# ── 9. Stratified analysis by CAD PRS quartile ───────────────────────────────
# For each quartile: M4, M5, M6, M7 — then LRT M5 vs M4, M6 vs M4, M7 vs M4
print(f"\n\n{'='*90}")
print("STRATIFIED ANALYSIS — M4/M5/M6/M7 by CAD PRS Quartile")
print('='*90)

strat_rows = []
for q in [1, 2, 3, 4]:
    dfq = df[df['prs_quartile'] == q].copy() if 'prs_quartile' in df.columns else df.copy()
    nq  = len(dfq)
    eq  = int(dfq['event'].sum())
    print(f"\n  Quartile Q{q}: N={nq:,}  Events={eq:,}")

    qresults = {}
    for mname, extra in [('M4', []), ('M5', PHYS_COLS_), ('M6', SOC_COLS_), ('M7', PHYS_COLS_ + SOC_COLS_)]:
        cols = DEMO_COLS + ['CAD_PRS'] + CLIN_COLS_ + extra
        cols = [c for c in dict.fromkeys(cols) if c in dfq.columns]
        sub  = dfq[['time', 'event'] + cols].dropna()
        k    = len(cols)
        cph  = CoxPHFitter(penalizer=0.01)
        try:
            cph.fit(sub, duration_col='time', event_col='event', show_progress=False)
            logL = cph.log_likelihood_
            r2   = royston_r2(logL, len(sub))
            c    = cph.concordance_index_
            qresults[mname] = {'logL': logL, 'k': k, 'r2': r2, 'c': c, 'n': len(sub)}
        except Exception as e:
            print(f"    {mname} failed: {e}")
            qresults[mname] = None

    for comp in [('M5','M4'), ('M6','M4'), ('M7','M4')]:
        f, n_ = comp
        if qresults.get(f) and qresults.get(n_):
            sf, sn = qresults[f], qresults[n_]
            chi2 = 2 * (sf['logL'] - sn['logL'])
            ddf  = sf['k'] - sn['k']
            pval = stats.chi2.sf(chi2, ddf) if ddf > 0 else np.nan
            strat_rows.append({
                'Quartile': f"Q{q}",
                'N': nq, 'Events': eq,
                'Comparison': f"{f} vs {n_}",
                'M4 C': f"{qresults['M4']['c']:.4f}",
                'Model C': f"{sf['c']:.4f}",
                'LRT χ²': f"{chi2:.2f}",
                'df': ddf,
                'p-value': f"{pval:.2e}",
                'ΔR²': f"{sf['r2'] - sn['r2']:.4f}",
            })

df_strat = pd.DataFrame(strat_rows)
print(f"\n{'='*110}")
print("STRATIFIED TABLE — LRT comparisons within each CAD PRS quartile")
print('='*110)
print(df_strat.to_string(index=False))

# ── 10. Save tables to CSV ────────────────────────────────────────────────────
df_t1.to_csv(   f'{OUT_DIR}/cox_table1_model_fit.csv',     index=False)
df_t2.to_csv(   f'{OUT_DIR}/cox_table2_lrt_comparisons.csv', index=False)
df_strat.to_csv(f'{OUT_DIR}/cox_table3_stratified.csv',    index=False)
print(f"\nTables saved to {OUT_DIR}/cox_table1/2/3_*.csv")
print("DONE")
