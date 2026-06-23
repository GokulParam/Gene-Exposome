"""
cox_nested_models.py
====================
Nested Cox Proportional Hazards models (M1–M7) quantifying the incremental
contribution of genetic risk (CAD PRS), clinical variables, and the exposome
(physical and social domains) to time-to-incident-MACE survival.

Data source: master_dataset.csv (380,780 participants, N=282 cols)
             cohort_skeleton_incident.csv (for per-person event dates)

Column names reflect the actual master_dataset schema built by build_master.py.

Model sequence
──────────────
M1  age + sex + ethnicity dummies                      ← demographic baseline
M2  M1 + CAD_PRS                                       ← adds genetic risk
M3  M1 + ALL clinical variables                        ← adds full clinical picture
M4  M1 + CAD_PRS + ALL clinical variables              ← primary reference model
M5  M4 + physical exposome (gee_, noise_, smart_, toxins_, wildfire_)
M6  M4 + social exposome  (social_)
M7  M4 + physical + social exposome                    ← full model

'ALL clinical variables' = labs + comorbidities + medications + smoking + other PRS.
Everything we measured, except age/sex/ethnicity and the exposome.
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from lifelines import CoxPHFitter

warnings.filterwarnings('ignore')

WORKSPACE    = '/home/dataproc/workspaces/geneexposome'
LANDMARK     = pd.Timestamp('2018-01-01')
ADMIN_CENSOR = pd.Timestamp('2024-09-30')   # CDR C2024Q3R9 cut-off
OUT_DIR      = WORKSPACE


# ── 1. Load master dataset ────────────────────────────────────────────────────
print("Loading master_dataset.csv …")
master = pd.read_csv(f'{WORKSPACE}/master_dataset.csv', dtype={'person_id': str})
print(f"  Shape: {master.shape}")

# ── 2. Merge event dates from cohort skeleton ─────────────────────────────────
# cohort_skeleton_incident.csv has mace3_date and death_date_any per person.
# These were not included in master_dataset to keep file size manageable.
print("Loading cohort_skeleton_incident.csv for event dates …")
skel_cols = ['person_id', 'mace3_date', 'death_date_any']
skel = pd.read_csv(
    f'{WORKSPACE}/cohort_skeleton_incident.csv',
    usecols=lambda c: c in skel_cols,
    dtype={'person_id': str},
    parse_dates=['mace3_date', 'death_date_any'],
)
master = master.merge(skel, on='person_id', how='left')
del skel; gc.collect()

# ── 3. Build time-to-event columns ───────────────────────────────────────────
# event = incident 3-point MACE (already restricted to post-2018 in pipeline)
# time  = days from landmark to first MACE, or to earliest of (death, admin censor)
master['event'] = master['mace3_event'].astype(int)

event_date   = pd.to_datetime(master['mace3_date'],     errors='coerce')
death_date   = pd.to_datetime(master['death_date_any'], errors='coerce')
censor_date  = ADMIN_CENSOR

# Censoring date per person = min(death, admin censor)
censor_per_person = death_date.clip(upper=censor_date).fillna(censor_date)

# time = event date if event, else censor date
time_date = np.where(master['event'] == 1, event_date, censor_per_person)
master['time'] = (pd.to_datetime(time_date) - LANDMARK).dt.days.clip(lower=1)

n_with_date = event_date.notna().sum()
print(f"  Event dates available: {n_with_date:,} / {master['event'].sum():,} events")
print(f"  Follow-up range: {master['time'].min():.0f}–{master['time'].max():.0f} days")

# ── 4. Encode ethnicity as dummy variables (replaces PC1–PC10) ────────────────
# Reference category = most common value (typically "Not Hispanic or Latino")
eth_dummies = pd.get_dummies(master['ethnicity'], prefix='eth', drop_first=False)
ref_cat = master['ethnicity'].value_counts().index[0]
ref_col = 'eth_' + ref_cat
eth_cols = [c for c in eth_dummies.columns if c != ref_col]
eth_dummies = eth_dummies[eth_cols].astype(int)
master = pd.concat([master, eth_dummies], axis=1)
print(f"  Ethnicity dummies: {eth_cols}  (ref='{ref_cat}')")

# ── 5. Sex encoding ───────────────────────────────────────────────────────────
master['sex_male'] = (master['sex_at_birth']
                      .str.lower()
                      .map(lambda x: 1 if 'male' in str(x) and 'female' not in str(x) else 0))

# ── 6. Define column groups ───────────────────────────────────────────────────

# Demographic baseline (M1 covariates)
DEMO_COLS = ['age_at_landmark', 'sex_male'] + eth_cols

# Labs — continuous measurements taken before landmark
LAB_COLS = [c for c in [
    'bmi', 'sbp', 'dbp', 'chol_total', 'ldl', 'hdl', 'hba1c', 'glucose', 'creatinine',
] if c in master.columns]

# All pre-landmark comorbidity flags (binary 0/1)
COMORBIDITY_COLS = [c for c in [
    'htn', 't2dm', 't1dm', 'obesity_dx', 'cad_prev', 'ckd', 'fh', 'pad',
    'metabolic_syndrome', 'osa', 'heart_failure', 'afib',
    'ra', 'sle', 'psoriasis', 'ibd', 'crohns', 'ulc_colitis', 'hiv',
    'hypothyroidism', 'hyperthyroidism', 'pcos', 'cushings', 'acromegaly',
    'cardiotoxic_chemo',
    'preeclampsia', 'gest_dm', 'preterm', 'preg_loss',
] if c in master.columns]

# All pre-landmark medication flags (binary 0/1)
# Include all individual drugs; also composite any_antihtn and any_dm_med
# (penalizer handles overlap with component flags)
MED_COLS = [c for c in [
    'statin', 'pcsk9i',
    'ace_inhibitor', 'arb', 'beta_blocker', 'ccb', 'diuretic', 'any_antihtn',
    'aspirin', 'p2y12', 'oral_anticoag', 'arni',
    'metformin', 'sglt2i', 'glp1ra', 'dpp4i', 'sulfonylurea', 'tzd', 'insulin_any',
    'any_dm_med',
] if c in master.columns]

# Smoking — 1/0/NaN; impute missing as 0.5 (uncertain) for Cox models
# to avoid dropping ~30% of cohort with no survey data
if 'current_smoker' in master.columns:
    master['current_smoker_imp'] = master['current_smoker'].fillna(0.5)
    SMOKE_COL = ['current_smoker_imp']
else:
    SMOKE_COL = []

# Other PRS scores (non-CAD) — RINT-normalised
OTHER_PRS_COLS = [c for c in ['prs_ldlc', 'prs_obesity', 'prs_sbp', 'prs_t2d']
                  if c in master.columns]

# Full clinical block = labs + comorbidities + medications + smoking + non-CAD PRS
CLIN_COLS = LAB_COLS + COMORBIDITY_COLS + MED_COLS + SMOKE_COL + OTHER_PRS_COLS

# Exposome domains — physical and social
PHYS_PREFIXES = ('gee_', 'noise_', 'smart_', 'toxins_', 'wildfire_')
SOC_PREFIXES  = ('social_',)
PHYS_COLS = [c for c in master.columns if c.startswith(PHYS_PREFIXES)]
SOC_COLS  = [c for c in master.columns if c.startswith(SOC_PREFIXES)]

print(f"\nColumn counts:")
print(f"  Demographic (M1): {len(DEMO_COLS)}  [age, sex, {len(eth_cols)} ethnicity dummies]")
print(f"  Labs:             {len(LAB_COLS)}")
print(f"  Comorbidities:    {len(COMORBIDITY_COLS)}")
print(f"  Medications:      {len(MED_COLS)}")
print(f"  Smoking:          {len(SMOKE_COL)}")
print(f"  Other PRS:        {len(OTHER_PRS_COLS)}")
print(f"  Total clinical:   {len(CLIN_COLS)}")
print(f"  Physical exposome:{len(PHYS_COLS)}")
print(f"  Social exposome:  {len(SOC_COLS)}")


# ── 7. Build analysis dataframe ───────────────────────────────────────────────
ALL_NEEDED = (['time', 'event'] + DEMO_COLS + ['cad_prs'] +
              CLIN_COLS + PHYS_COLS + SOC_COLS)
avail = [c for c in ALL_NEEDED if c in master.columns]
df    = master[avail].copy()
del master; gc.collect()

# Drop rows missing time/event/age/sex/PRS (core non-negotiables)
df = df.dropna(subset=['time', 'event', 'age_at_landmark', 'sex_male', 'cad_prs'])
df['time']  = df['time'].clip(lower=1).astype(float)
df['event'] = df['event'].astype(int)

# Refresh column lists against what's actually present and non-constant
def usable_cols(cols):
    return [c for c in cols if c in df.columns and df[c].nunique() > 1]

DEMO_COLS_ = usable_cols(DEMO_COLS)
CLIN_COLS_ = usable_cols(CLIN_COLS)
PHYS_COLS_ = usable_cols(PHYS_COLS)
SOC_COLS_  = usable_cols(SOC_COLS)

N      = len(df)
EVENTS = int(df['event'].sum())
print(f"\nAnalysis dataset: N={N:,}  Events={EVENTS:,}  ({100*EVENTS/N:.1f}%)")
print(f"PRS quartile column: {'present' if 'prs_quartile' in df.columns else 'building now'}")

# CAD PRS quartile (used for stratified analysis)
if 'prs_quartile' not in df.columns:
    df['prs_quartile'] = pd.qcut(df['cad_prs'], 4, labels=[1, 2, 3, 4]).astype(int)


# ── 8. Null model log-likelihood (for Royston R²) ────────────────────────────
print("\nFitting null model …")
_null_df = df[['time', 'event']].copy()
_null_df['_c'] = 0.0
cph_null = CoxPHFitter()
cph_null.fit(_null_df, duration_col='time', event_col='event',
             formula='_c - 1', show_progress=False)
logL_null = cph_null.log_likelihood_

def royston_r2(logL_model, n):
    lrt = 2 * (logL_model - logL_null)
    return float(np.clip(1 - np.exp(-lrt / n), 0, 1))


# ── 9. Cox fitting helper ─────────────────────────────────────────────────────
def fit_cox(covariate_cols, src_df=None, label=""):
    src = src_df if src_df is not None else df
    cols = list(dict.fromkeys(c for c in covariate_cols if c in src.columns))
    sub  = src[['time', 'event'] + cols].dropna()
    n, k = len(sub), len(cols)

    cph = CoxPHFitter(penalizer=0.05)   # ridge: handles correlated exposome columns
    cph.fit(sub, duration_col='time', event_col='event', show_progress=False)

    logL = cph.log_likelihood_
    aic  = -2 * logL + 2 * k
    bic  = -2 * logL + k * np.log(n)
    c    = cph.concordance_index_
    try:
        c_se = cph.concordance_index_se_
    except AttributeError:
        c_se = 0.005
    c_lo = max(0.5, c - 1.96 * c_se)
    c_hi = min(1.0, c + 1.96 * c_se)
    r2   = royston_r2(logL, n)

    return cph, {
        'label': label, 'n': n, 'events': int(sub['event'].sum()),
        'k': k, 'logL': logL, 'aic': aic, 'bic': bic,
        'c': c, 'c_lo': c_lo, 'c_hi': c_hi, 'r2': r2,
    }


# ── 10. Fit M1–M7 ─────────────────────────────────────────────────────────────
MODELS = {
    'M1': DEMO_COLS_,
    'M2': DEMO_COLS_ + ['cad_prs'],
    'M3': DEMO_COLS_ + CLIN_COLS_,
    'M4': DEMO_COLS_ + ['cad_prs'] + CLIN_COLS_,
    'M5': DEMO_COLS_ + ['cad_prs'] + CLIN_COLS_ + PHYS_COLS_,
    'M6': DEMO_COLS_ + ['cad_prs'] + CLIN_COLS_ + SOC_COLS_,
    'M7': DEMO_COLS_ + ['cad_prs'] + CLIN_COLS_ + PHYS_COLS_ + SOC_COLS_,
}

DESCRIPTIONS = {
    'M1': 'Demographics only  (age, sex, ethnicity)',
    'M2': 'M1 + CAD PRS',
    'M3': 'M1 + All clinical  (labs, PMH, medications, smoking, non-CAD PRS)',
    'M4': 'M1 + CAD PRS + All clinical  [primary baseline]',
    'M5': 'M4 + Physical exposome  (GEE, noise, SMART, toxins, wildfire)',
    'M6': 'M4 + Social exposome',
    'M7': 'M4 + Physical + Social exposome  [full model]',
}

results = {}
for name, cols in MODELS.items():
    unique_cols = list(dict.fromkeys(cols))
    print(f"Fitting {name} ({len(unique_cols)} covariates) …", end='  ', flush=True)
    try:
        cph, s = fit_cox(unique_cols, label=name)
        results[name] = (cph, s)
        print(f"logL={s['logL']:.1f}  C={s['c']:.4f}  R²={s['r2']:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")
        results[name] = None


# ── 11. TABLE 1 — Model fit statistics ───────────────────────────────────────
rows1 = []
for name in ['M1','M2','M3','M4','M5','M6','M7']:
    if results[name] is None:
        continue
    s = results[name][1]
    rows1.append({
        'Model':        name,
        'Description':  DESCRIPTIONS[name],
        'N':            f"{s['n']:,}",
        'Events':       f"{s['events']:,}",
        'k':            s['k'],
        'Log-L':        f"{s['logL']:.2f}",
        'AIC':          f"{s['aic']:.1f}",
        'BIC':          f"{s['bic']:.1f}",
        'Royston R²':   f"{s['r2']:.4f}",
        "Harrell's C":  f"{s['c']:.4f}",
        '95% CI C':     f"[{s['c_lo']:.4f}, {s['c_hi']:.4f}]",
    })

df_t1 = pd.DataFrame(rows1)
sep = '=' * 105
print(f"\n{sep}\nTABLE 1 — Model Fit Statistics\n{sep}")
print(df_t1.to_string(index=False))


# ── 12. TABLE 2 — Nested LRT comparisons ─────────────────────────────────────
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
    if results[full] is None or results[null] is None:
        return None
    sf, sn  = results[full][1], results[null][1]
    d_logL  = sf['logL'] - sn['logL']
    chi2    = 2 * d_logL
    ddf     = sf['k'] - sn['k']
    pval    = scipy_stats.chi2.sf(chi2, max(ddf, 1))
    return {
        'Comparison':      f"{full} vs {null}",
        'ΔlogL':           f"{d_logL:.2f}",
        'LRT χ²':          f"{chi2:.2f}",
        'df':              ddf,
        'p-value':         f"{pval:.2e}" if pval > 1e-300 else "< 1e-300",
        'ΔR²':             f"{sf['r2'] - sn['r2']:.4f}",
        'Interpretation':  INTERP.get((full, null), ''),
    }

COMPARISONS = [
    ('M2','M1'), ('M3','M1'), ('M4','M3'), ('M4','M2'),
    ('M5','M4'), ('M6','M4'), ('M7','M4'), ('M7','M5'), ('M7','M6'),
]

rows2 = [r for pair in COMPARISONS if (r := lrt_row(*pair)) is not None]
df_t2 = pd.DataFrame(rows2)
print(f"\n{sep}\nTABLE 2 — Nested LRT Comparisons\n{sep}")
print(df_t2.to_string(index=False))


# ── 13. STRATIFIED ANALYSIS by CAD PRS quartile ───────────────────────────────
print(f"\n{sep}\nSTRATIFIED ANALYSIS — M4 / M5 / M6 / M7 within CAD PRS quartiles\n{sep}")

strat_rows = []
for q in [1, 2, 3, 4]:
    mask = df['prs_quartile'] == q
    dfq  = df[mask].copy()
    nq, eq = len(dfq), int(dfq['event'].sum())
    print(f"\n  Q{q}: N={nq:,}  Events={eq:,}")

    qres = {}
    for mname, extra in [('M4', []), ('M5', PHYS_COLS_), ('M6', SOC_COLS_), ('M7', PHYS_COLS_ + SOC_COLS_)]:
        cols = list(dict.fromkeys(DEMO_COLS_ + ['cad_prs'] + CLIN_COLS_ + extra))
        try:
            _, s = fit_cox(cols, src_df=dfq, label=f"{mname}_Q{q}")
            qres[mname] = s
            print(f"    {mname}: C={s['c']:.4f}  R²={s['r2']:.4f}")
        except Exception as e:
            print(f"    {mname}: FAILED — {e}")
            qres[mname] = None

    for full, null in [('M5','M4'), ('M6','M4'), ('M7','M4')]:
        if qres.get(full) and qres.get(null):
            sf, sn = qres[full], qres[null]
            chi2 = 2 * (sf['logL'] - sn['logL'])
            ddf  = sf['k'] - sn['k']
            pval = scipy_stats.chi2.sf(chi2, max(ddf, 1))
            strat_rows.append({
                'Quartile':  f"Q{q}",
                'N':         nq,
                'Events':    eq,
                'Comparison':f"{full} vs {null}",
                'M4 C':      f"{qres['M4']['c']:.4f}" if qres['M4'] else 'N/A',
                'Fuller C':  f"{sf['c']:.4f}",
                'LRT χ²':    f"{chi2:.2f}",
                'df':        ddf,
                'p-value':   f"{pval:.2e}",
                'ΔR²':       f"{sf['r2'] - sn['r2']:.4f}",
            })

df_strat = pd.DataFrame(strat_rows)
print(f"\n{sep}\nSTRATIFIED TABLE — exposome LRT within each PRS quartile\n{sep}")
print(df_strat.to_string(index=False))


# ── 14. Save ──────────────────────────────────────────────────────────────────
df_t1.to_csv(   f'{OUT_DIR}/cox_table1_model_fit.csv',       index=False)
df_t2.to_csv(   f'{OUT_DIR}/cox_table2_lrt_comparisons.csv', index=False)
df_strat.to_csv(f'{OUT_DIR}/cox_table3_prs_stratified.csv',  index=False)
print(f"\nSaved tables to {OUT_DIR}/cox_table1/2/3_*.csv")
print("DONE")
