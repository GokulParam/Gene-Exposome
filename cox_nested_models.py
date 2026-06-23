"""
cox_nested_models.py
====================
Nested Cox Proportional Hazards models (M1–M7) quantifying the incremental
contribution of genetic risk (CAD PRS), clinical variables, and the exposome
(physical and social domains) to time-to-incident-MACE survival.

Data source: master_dataset.csv  (380,780 × 282)
             cohort_skeleton_incident.csv  (event dates)

Memory strategy: load only needed columns from each file; downcast to float32/int8;
delete intermediates aggressively; fit one model at a time.
"""

import os, gc, importlib, subprocess, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Install lifelines if absent (AoU Dataproc does not ship it) ───────────────
subprocess.run([sys.executable, '-m', 'pip', 'install', 'lifelines', '-q'], check=True)
importlib.invalidate_caches()
from lifelines import CoxPHFitter  # noqa: E402

warnings.filterwarnings('ignore')

WORKSPACE    = '/home/dataproc/workspaces/geneexposome'
LANDMARK     = pd.Timestamp('2018-01-01')
ADMIN_CENSOR = pd.Timestamp('2024-09-30')
OUT_DIR      = WORKSPACE


# ══════════════════════════════════════════════════════════════════════════════
# 1. Column definitions — everything by name so we load only what we need
# ══════════════════════════════════════════════════════════════════════════════

# Lab columns are excluded from the clinical block.
# Missingness audit (cox_missingness_check.py) found all 9 labs are 64–92% missing
# in AoU — EHR measurements only recorded when a test was ordered.
# Complete-case analysis on labs collapses the cohort from 380,780 → 9,451 (2.5%),
# producing a highly selected, non-representative subset.  Imputation at >50%
# missing is unreliable.  Labs are therefore omitted; the clinical block uses only
# the binary flags (comorbidities, medications, other PRS) which are 0% missing.
LAB_COLS = []   # excluded — see above
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

# Exposome domain prefixes
PHYS_PREFIXES = ('gee_', 'noise_', 'smart_', 'toxins_', 'wildfire_')
SOC_PREFIXES  = ('social_',)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Peek at master columns to build exact usecols list before loading
# ══════════════════════════════════════════════════════════════════════════════
print("Reading master_dataset.csv column list …")
all_cols = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', nrows=0).columns.tolist()

PHYS_COLS_ALL = [c for c in all_cols if c.startswith(PHYS_PREFIXES)]
SOC_COLS_ALL  = [c for c in all_cols if c.startswith(SOC_PREFIXES)]

# Core columns we always need
CORE_LOAD = list(dict.fromkeys(
    ['person_id', 'mace3_event', 'age_at_landmark', 'sex_at_birth', 'ethnicity',
     'cad_prs', 'current_smoker']
    + COMORBIDITY_COLS + MED_COLS + OTHER_PRS_COLS
    + PHYS_COLS_ALL + SOC_COLS_ALL
))
use_cols = [c for c in CORE_LOAD if c in all_cols]
print(f"  Loading {len(use_cols)} / {len(all_cols)} columns")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Load — memory-efficient dtypes
# ══════════════════════════════════════════════════════════════════════════════
print("Loading master_dataset.csv …")
dtype_overrides = {c: 'float32' for c in (OTHER_PRS_COLS + PHYS_COLS_ALL + SOC_COLS_ALL)
                   if c in all_cols}
dtype_overrides.update({c: 'int8' for c in (COMORBIDITY_COLS + MED_COLS)
                         if c in all_cols})
dtype_overrides['person_id'] = str

df = pd.read_csv(
    f'{WORKSPACE}/master_dataset.csv',
    usecols=use_cols,
    dtype=dtype_overrides,
)
print(f"  Loaded: {df.shape}  RAM ≈ {df.memory_usage(deep=True).sum()/1e6:.0f} MB")
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Event dates from skeleton
# ══════════════════════════════════════════════════════════════════════════════
print("Merging event dates from cohort_skeleton_incident.csv …")
skel = pd.read_csv(
    f'{WORKSPACE}/cohort_skeleton_incident.csv',
    usecols=['person_id', 'mace3_date', 'death_date_any'],
    dtype={'person_id': str},
    parse_dates=['mace3_date', 'death_date_any'],
)
df = df.merge(skel[['person_id', 'mace3_date', 'death_date_any']],
              on='person_id', how='left')
del skel; gc.collect()

event_date  = pd.to_datetime(df['mace3_date'],     errors='coerce')
death_date  = pd.to_datetime(df['death_date_any'], errors='coerce')
censor_pp   = death_date.clip(upper=ADMIN_CENSOR).fillna(ADMIN_CENSOR)

df['event'] = df['mace3_event'].astype('int8')
df['time']  = np.where(
    df['event'] == 1,
    (event_date  - LANDMARK).dt.days,
    (censor_pp   - LANDMARK).dt.days,
).clip(1).astype('float32')

df = df.drop(columns=['mace3_date', 'death_date_any', 'mace3_event'], errors='ignore')
gc.collect()
print(f"  Events: {df['event'].sum():,}  Follow-up: {df['time'].min():.0f}–{df['time'].max():.0f} d")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Encode demographics
# ══════════════════════════════════════════════════════════════════════════════

# Sex → binary int8
df['sex_male'] = df['sex_at_birth'].str.lower().map(
    lambda x: 1 if 'male' in str(x) and 'female' not in str(x) else 0
).astype('int8')
df = df.drop(columns=['sex_at_birth'])

# Ethnicity → dummies (drop most common category as reference)
ref_cat  = df['ethnicity'].value_counts().index[0]
eth_dum  = pd.get_dummies(df['ethnicity'], prefix='eth').astype('int8')
eth_cols = [c for c in eth_dum.columns if c != f'eth_{ref_cat}']
df = pd.concat([df.drop(columns=['ethnicity']), eth_dum[eth_cols]], axis=1)
del eth_dum; gc.collect()
print(f"  Ethnicity dummies: {eth_cols}  (ref='{ref_cat}')")

# Smoking — impute NaN → 0.5 (uncertain; avoids dropping ~30% of cohort)
if 'current_smoker' in df.columns:
    df['current_smoker_imp'] = df['current_smoker'].fillna(0.5).astype('float32')
    df = df.drop(columns=['current_smoker'])
    SMOKE_COL = ['current_smoker_imp']
else:
    SMOKE_COL = []

# ══════════════════════════════════════════════════════════════════════════════
# 6. Finalise column groups
# ══════════════════════════════════════════════════════════════════════════════
def present(lst):
    return [c for c in lst if c in df.columns and df[c].nunique() > 1]

DEMO_COLS = present(['age_at_landmark', 'sex_male'] + eth_cols)
CLIN_COLS = present(COMORBIDITY_COLS + MED_COLS + SMOKE_COL + OTHER_PRS_COLS)
PHYS_COLS = present(PHYS_COLS_ALL)
SOC_COLS  = present(SOC_COLS_ALL)

print(f"\nCovariates — Demo:{len(DEMO_COLS)}  Clinical:{len(CLIN_COLS)}  "
      f"Phys:{len(PHYS_COLS)}  Soc:{len(SOC_COLS)}")
print(f"  Note: lab columns excluded (64–92% missing in AoU EHR data)")

# ── Single analytic sample: complete on all variables used across M1–M7 ───────
# Defined ONCE so all models share identical rows — required for valid LRT.
all_model_cols = list(dict.fromkeys(
    ['time', 'event'] + DEMO_COLS + ['cad_prs'] + CLIN_COLS + PHYS_COLS + SOC_COLS
))
all_model_cols = [c for c in all_model_cols if c in df.columns]

# Drop unnecessary columns before dropna to minimise peak RAM
df = df[all_model_cols].dropna()
gc.collect()

N      = len(df)
EVENTS = int(df['event'].sum())
print(f"\nAnalytic sample: N={N:,}  Events={EVENTS:,}  ({100*EVENTS/N:.1f}%)")
print(f"  RAM ≈ {df.memory_usage(deep=True).sum()/1e6:.0f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# 6b. Reduce exposome domains with PCA
#
# Fitting Cox models with 123 physical + 79 social columns on 380k rows
# requires >2 GB for the internal Newton-Raphson matrices and crashes the kernel.
# PCA within each domain compresses to the components explaining 90% of variance
# (typically 15–25 PCs per domain) while preserving the between-domain block
# structure needed for the LRT comparisons.
#
# Scientific rationale: the nested LRT tests whether a *domain* adds incremental
# predictive value — PCA components span the same column space as the raw
# variables, so the test is equivalent.  The n_components selected is printed
# so it can be reported in the methods section.
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

VARIANCE_TARGET = 0.90   # keep enough PCs to explain this fraction of variance

def pca_reduce(raw_cols, prefix, df_in, var_target=VARIANCE_TARGET):
    """Fit PCA on raw_cols of df_in, return (df_pcs, pc_col_names)."""
    X = df_in[raw_cols].values.astype('float32')
    # Standardise: PCA on exposome variables which have different scales/units
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(len(raw_cols), X.shape[0] - 1), svd_solver='full')
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, var_target)) + 1
    n_keep = max(n_keep, 1)
    print(f"  {prefix}: {len(raw_cols)} cols → {n_keep} PCs "
          f"(explain {100*cumvar[n_keep-1]:.1f}% variance)")
    scores = pca.transform(X)[:, :n_keep]
    pc_names = [f'{prefix}_PC{i+1}' for i in range(n_keep)]
    df_pcs = pd.DataFrame(scores.astype('float32'), index=df_in.index, columns=pc_names)
    return df_pcs, pc_names

print("\nReducing exposome domains via PCA …")
phys_pcs, PHYS_PC_COLS = pca_reduce(PHYS_COLS, 'phys', df)
soc_pcs,  SOC_PC_COLS  = pca_reduce(SOC_COLS,  'soc',  df)

# Replace raw exposome columns in df with PCs to free RAM
df = df.drop(columns=PHYS_COLS + SOC_COLS)
df = pd.concat([df, phys_pcs, soc_pcs], axis=1)
del phys_pcs, soc_pcs; gc.collect()

# Update lists used in model definitions
PHYS_COLS = PHYS_PC_COLS
SOC_COLS  = SOC_PC_COLS

print(f"\nFinal working dataframe: {df.shape}  "
      f"RAM ≈ {df.memory_usage(deep=True).sum()/1e6:.0f} MB")

# CAD PRS quartile within the analytic sample
df['prs_q'] = pd.qcut(df['cad_prs'], 4, labels=[1, 2, 3, 4]).astype(int)

# Remove zero-variance cols that survived after dropna (e.g. rare pregnancy flags)
DEMO_COLS = present(DEMO_COLS)
CLIN_COLS = present(CLIN_COLS)
PHYS_COLS = present(PHYS_COLS)
SOC_COLS  = present(SOC_COLS)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Null log-likelihood — analytical Breslow formula
#
# Computed on the SAME analytic sample used for all models.
# For each event time t, contribution = −log(number still at risk at t).
# Uses Breslow approximation for tied event times (matches lifelines default).
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing null log-likelihood on analytic sample …")

def _null_logL(times, events):
    t = np.asarray(times,  dtype=np.float64)
    e = np.asarray(events, dtype=bool)
    t_sorted    = np.sort(t)
    event_times = t[e]
    risk = len(t) - np.searchsorted(t_sorted, event_times, side='left')
    return float(-np.sum(np.log(np.clip(risk, 1, None))))

logL_null = _null_logL(df['time'], df['event'])
print(f"  logL_null = {logL_null:.2f}  (N={N:,}  Events={EVENTS:,})")

def royston_r2(logL_model, n):
    return float(np.clip(1 - np.exp(-2 * (logL_model - logL_null) / n), 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# 8. Cox fitting helper — fits on a sub-dataframe, deletes it after
# ══════════════════════════════════════════════════════════════════════════════
def fit_cox(covariate_cols, src=None, label=""):
    src  = src if src is not None else df
    # Analytic sample is already complete; no dropna needed except for stratified slices
    cols = list(dict.fromkeys(c for c in covariate_cols if c in src.columns))
    sub  = src[['time', 'event'] + cols].dropna()   # dropna handles stratified subsets
    # Drop zero-variance columns (can occur in PRS-quartile strata for rare flags)
    cols = [c for c in cols if sub[c].nunique() > 1]
    sub  = sub[['time', 'event'] + cols].copy()
    n, k = len(sub), len(cols)

    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(sub, duration_col='time', event_col='event', show_progress=False)
    del sub; gc.collect()

    logL = cph.log_likelihood_
    c    = cph.concordance_index_
    try:    c_se = cph.concordance_index_se_
    except: c_se = 0.005

    return {
        'label':  label,
        'n':      n,
        'events': int(sub['event'].sum()),
        'k':      k,
        'logL':   logL,
        'aic':    -2 * logL + 2 * k,
        'bic':    -2 * logL + k * np.log(n),
        'c':      c,
        'c_lo':   max(0.5, c - 1.96 * c_se),
        'c_hi':   min(1.0, c + 1.96 * c_se),
        'r2':     royston_r2(logL, n),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. Fit M1–M7
# ══════════════════════════════════════════════════════════════════════════════
MODELS = {
    'M1': DEMO_COLS,
    'M2': DEMO_COLS + ['cad_prs'],
    'M3': DEMO_COLS + CLIN_COLS,
    'M4': DEMO_COLS + ['cad_prs'] + CLIN_COLS,
    'M5': DEMO_COLS + ['cad_prs'] + CLIN_COLS + PHYS_COLS,
    'M6': DEMO_COLS + ['cad_prs'] + CLIN_COLS + SOC_COLS,
    'M7': DEMO_COLS + ['cad_prs'] + CLIN_COLS + PHYS_COLS + SOC_COLS,
}
DESCRIPTIONS = {
    'M1': 'Demographics  (age, sex, ethnicity)',
    'M2': 'M1 + CAD PRS',
    'M3': 'M1 + All clinical  (labs, PMH, meds, smoking, non-CAD PRS)',
    'M4': 'M1 + CAD PRS + All clinical  [primary baseline]',
    'M5': 'M4 + Physical exposome PCs  (GEE, noise, SMART, toxins, wildfire)',
    'M6': 'M4 + Social exposome PCs',
    'M7': 'M4 + Physical + Social exposome PCs  [full model]',
}

results = {}
for name, cols in MODELS.items():
    u = list(dict.fromkeys(cols))
    print(f"Fitting {name} ({len(u)} covariates) … ", end='', flush=True)
    try:
        s = fit_cox(u, label=name)
        results[name] = s
        print(f"logL={s['logL']:.1f}  C={s['c']:.4f}  R²={s['r2']:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")
        results[name] = None
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# 10. Table 1 — model fit
# ══════════════════════════════════════════════════════════════════════════════
rows1 = []
for name in ['M1','M2','M3','M4','M5','M6','M7']:
    s = results.get(name)
    if s is None: continue
    rows1.append({
        'Model':       name,
        'Description': DESCRIPTIONS[name],
        'N':           f"{s['n']:,}",
        'Events':      f"{s['events']:,}",
        'k':           s['k'],
        'Log-L':       f"{s['logL']:.2f}",
        'AIC':         f"{s['aic']:.1f}",
        'BIC':         f"{s['bic']:.1f}",
        'Royston R²':  f"{s['r2']:.4f}",
        "Harrell C":   f"{s['c']:.4f}",
        '95% CI':      f"[{s['c_lo']:.4f}, {s['c_hi']:.4f}]",
    })

df_t1 = pd.DataFrame(rows1)
SEP = '=' * 110
print(f"\n{SEP}\nTABLE 1 — Model Fit Statistics\n{SEP}")
print(df_t1.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 11. Table 2 — nested LRT
# ══════════════════════════════════════════════════════════════════════════════
INTERP = {
    ('M2','M1'): 'Incremental value of CAD PRS over demographics',
    ('M3','M1'): 'Incremental value of all clinical variables over demographics',
    ('M4','M3'): 'Incremental value of CAD PRS on top of clinical variables',
    ('M4','M2'): 'Incremental value of clinical variables on top of PRS',
    ('M5','M4'): 'Incremental value of physical exposome over full baseline',
    ('M6','M4'): 'Incremental value of social exposome over full baseline',
    ('M7','M4'): 'Incremental value of full exposome over full baseline',
    ('M7','M5'): 'Incremental value of social exposome beyond physical',
    ('M7','M6'): 'Incremental value of physical exposome beyond social',
}

def lrt_row(full, null):
    sf, sn = results.get(full), results.get(null)
    if sf is None or sn is None: return None
    d_logL = sf['logL'] - sn['logL']
    chi2   = 2 * d_logL
    ddf    = sf['k'] - sn['k']
    pval   = scipy_stats.chi2.sf(chi2, max(ddf, 1))
    return {
        'Comparison':    f"{full} vs {null}",
        'ΔlogL':         f"{d_logL:.2f}",
        'LRT χ²':        f"{chi2:.2f}",
        'df':            ddf,
        'p-value':       f"{pval:.2e}" if pval > 1e-300 else "< 1e-300",
        'ΔR²':           f"{sf['r2'] - sn['r2']:.4f}",
        'Interpretation': INTERP.get((full, null), ''),
    }

rows2 = [r for pair in [
    ('M2','M1'), ('M3','M1'), ('M4','M3'), ('M4','M2'),
    ('M5','M4'), ('M6','M4'), ('M7','M4'), ('M7','M5'), ('M7','M6'),
] if (r := lrt_row(*pair)) is not None]

df_t2 = pd.DataFrame(rows2)
print(f"\n{SEP}\nTABLE 2 — Nested LRT Comparisons\n{SEP}")
print(df_t2.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 12. Stratified analysis by CAD PRS quartile
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\nSTRATIFIED — exposome LRT within PRS quartiles\n{SEP}")

strat_rows = []
for q in [1, 2, 3, 4]:
    dfq = df[df['prs_q'] == q]
    nq, eq = len(dfq), int(dfq['event'].sum())
    print(f"\n  Q{q}: N={nq:,}  Events={eq:,}")

    qres = {}
    for mname, extra in [('M4',[]), ('M5',PHYS_COLS), ('M6',SOC_COLS), ('M7',PHYS_COLS+SOC_COLS)]:
        cols = list(dict.fromkeys(DEMO_COLS + ['cad_prs'] + CLIN_COLS + extra))
        try:
            s = fit_cox(cols, src=dfq, label=f"{mname}_Q{q}")
            qres[mname] = s
            print(f"    {mname}: C={s['c']:.4f}  R²={s['r2']:.4f}")
        except Exception as e:
            print(f"    {mname}: FAILED — {e}")
            qres[mname] = None
        gc.collect()

    for full, null in [('M5','M4'), ('M6','M4'), ('M7','M4')]:
        sf, sn = qres.get(full), qres.get(null)
        if sf is None or sn is None: continue
        chi2 = 2 * (sf['logL'] - sn['logL'])
        ddf  = sf['k'] - sn['k']
        pval = scipy_stats.chi2.sf(chi2, max(ddf, 1))
        strat_rows.append({
            'Quartile':   f"Q{q}",
            'N':          nq,
            'Events':     eq,
            'Comparison': f"{full} vs {null}",
            'M4 C':       f"{qres['M4']['c']:.4f}" if qres['M4'] else 'N/A',
            'Fuller C':   f"{sf['c']:.4f}",
            'LRT χ²':     f"{chi2:.2f}",
            'df':         ddf,
            'p-value':    f"{pval:.2e}",
            'ΔR²':        f"{sf['r2'] - sn['r2']:.4f}",
        })

df_strat = pd.DataFrame(strat_rows)
print(f"\n{SEP}\nSTRATIFIED TABLE\n{SEP}")
print(df_strat.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 13. Save
# ══════════════════════════════════════════════════════════════════════════════
df_t1.to_csv(   f'{OUT_DIR}/cox_table1_model_fit.csv',       index=False)
df_t2.to_csv(   f'{OUT_DIR}/cox_table2_lrt_comparisons.csv', index=False)
df_strat.to_csv(f'{OUT_DIR}/cox_table3_prs_stratified.csv',  index=False)
print(f"\nSaved to {OUT_DIR}/cox_table[1-3]_*.csv")
print("DONE")
