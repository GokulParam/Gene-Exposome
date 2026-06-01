"""
build_master.py
===============
Builds master_dataset.csv from scratch.

Steps:
  1.  Load cohort_with_covariates.csv (clinical cohort + CAD PRS)
  2.  Merge extra PRS scores (LDLC, OBESITY, SBP, T2D)
  3.  Get zip3 per person from CDR (observation concept 3043579)
  4.  Exclude Alaska, Hawaii, territories, invalid zip3, and no-zip3 rows
  5.  Merge 6 exposome files via zip3
  6.  Re-query medications from CDR with correct concept IDs
  6b. Additional comorbidities (conditions + cardiotoxic chemo)
  7.  Fix smoking variable (concept 40766307, most recent answer)
  8.  Z-score all PRS columns
  9.  Save master_dataset.csv and copy to bucket
"""

import os, gc, subprocess
import pandas as pd
import numpy as np
from google.cloud import bigquery

# ── Config ────────────────────────────────────────────────────────────────────
WORKSPACE    = '/home/dataproc/workspaces/geneexposome'
CDR          = 'wb-silky-artichoke-2408.C2024Q3R9'
BUCKET       = 'gs://rw-migration-aou-rw-6cad436b'
EXPOSOME_DIR = f'{WORKSPACE}/Exposome'
LANDMARK     = '2018-01-01'
client       = bigquery.Client(project='wb-shining-lemon-5239')

EXCLUDE_ZIP3 = {
    '000',
    '006', '007', '008', '009',
    '967', '968',
    '969',
    '995', '996', '997', '998', '999',
}

EXPOSOME_FILES = {
    'GEE_Final_3Zip.csv':            'gee',
    'Zip3_Social_Exposome_Final.csv': 'social',
    'Noise_3Zip.csv':                 'noise',
    'SMART_3Zip.csv':                 'smart',
    'Toxins_3Zip.csv':                'toxins',
    'Wildfire_3Zip.csv':              'wildfire',
}

EXTRA_PRS = ['LDLC', 'OBESITY', 'SBP', 'T2D']

# ── Medication flags: (column_name, concept_ids)
# All concept IDs are RxNorm ingredient-level; concept_ancestor expands to
# all formulations.  Verified against CDR concept table May 2024.
# Note: rivaroxaban and dabigatran concept IDs need CDR verification —
# they are included here with best-available IDs.
MED_FLAGS = [
    ('statin',        [1545958,  # atorvastatin
                       1510813,  # rosuvastatin
                       1539403,  # simvastatin
                       1551860,  # pravastatin
                       1592085,  # lovastatin
                       1549686,  # fluvastatin
                       40165636]),# pitavastatin
    ('ace_inhibitor', [1308216,  # lisinopril
                       1334456,  # ramipril
                       1341927,  # enalapril
                       1335471,  # benazepril
                       1373928,  # perindopril
                       1340128,  # captopril
                       1331235,  # quinapril
                       1342439]),# fosinopril
    ('arb',           [1367500,  # losartan
                       1308842,  # valsartan
                       40226742, # olmesartan
                       1347384,  # irbesartan
                       1351557,  # candesartan
                       1386957,  # telmisartan
                       44818489]),# azilsartan
    ('beta_blocker',  [1307046,  # metoprolol
                       1314002,  # atenolol
                       1346823,  # carvedilol
                       1338005,  # bisoprolol
                       1353766,  # propranolol
                       1313200]),# nebivolol
    ('ccb',           [1332418,  # amlodipine
                       1318853,  # nifedipine
                       1328165,  # diltiazem
                       1307788,  # verapamil
                       1326012,  # felodipine
                       40220386]),# clevidipine
    ('diuretic',      [974166,   # hydrochlorothiazide
                       1395058,  # chlorthalidone
                       956874,   # furosemide
                       992590,   # spironolactone
                       1326303,  # indapamide
                       942350,   # torsemide
                       932745]), # bumetanide
    ('aspirin',       [1112807]),
    ('p2y12',         [1322184,  # clopidogrel
                       40163924]),# ticagrelor
    ('oral_anticoag', [1310149,  # warfarin
                       43013024, # apixaban
                       1592645,  # rivaroxaban (best-available ID — verify)
                       1599538]),# dabigatran (best-available ID — verify)
    ('metformin',     [1503297]),

    # ── Diabetes medications (beyond metformin) ──────────────────────────────
    # Concept IDs pending concept_id_lookup.py — [0] = skipped at runtime
    ('sglt2i',        [0]),   # empagliflozin, canagliflozin, dapagliflozin, ertugliflozin
    ('glp1ra',        [0]),   # semaglutide, liraglutide, dulaglutide, exenatide, tirzepatide
    ('dpp4i',         [0]),   # sitagliptin, saxagliptin, alogliptin, linagliptin
    ('sulfonylurea',  [0]),   # glipizide, glyburide, glimepiride
    ('tzd',           [0]),   # pioglitazone, rosiglitazone
    ('insulin_any',   [0]),   # glargine, lispro, aspart, detemir, degludec, NPH, regular

    # ── ARNI ────────────────────────────────────────────────────────────────
    ('arni',          [0]),   # sacubitril (valsartan already in arb flag)
]

# Derived composite flags
ANY_ANTIHTN_COLS = ['ace_inhibitor', 'arb', 'beta_blocker', 'ccb', 'diuretic']
ANY_DM_MED_COLS  = ['metformin', 'sglt2i', 'glp1ra', 'dpp4i',
                    'sulfonylurea', 'tzd', 'insulin_any']

# ── Additional comorbidity flags ──────────────────────────────────────────────
# Concept IDs are OMOP standard (SNOMED-based) ancestor concepts; concept_ancestor
# expands to all descendant codes.  Run comorbidities_diagnostic.py first to
# verify IDs against this CDR version before trusting counts.
COND_FLAGS = [
    # Very High Risk
    ('t1dm',             [201254]),            # Type 1 diabetes mellitus
    ('fh',               [4134862]),           # Familial hypercholesterolemia (SNOMED 398036000)
    ('pad',              [321052]),            # Peripheral vascular disease

    # High Risk
    ('metabolic_syndrome', [436940]),          # Metabolic syndrome X (SNOMED 237602007)
    ('osa',              [442588]),            # Obstructive sleep apnea syndrome (SNOMED 78275009)
    ('heart_failure',    [316139]),            # Heart failure
    ('afib',             [313217]),            # Atrial fibrillation

    # Inflammatory / Autoimmune
    ('ra',               [80809]),             # Rheumatoid arthritis
    ('sle',              [257628]),            # Systemic lupus erythematosus (SNOMED 55464009)
    ('psoriasis',        [140168]),            # Psoriasis
    ('crohns',           [201606]),            # Crohn's disease
    ('ulc_colitis',      [81893]),             # Ulcerative colitis (SNOMED 64766004)
    ('hiv',              [439727]),            # Human immunodeficiency virus infection

    # Endocrine / Hormonal
    ('hypothyroidism',   [140673]),            # Hypothyroidism
    ('hyperthyroidism',  [4142479]),           # Hyperthyroidism (SNOMED 34486009)
    ('pcos',             [40443308]),          # Polycystic ovary syndrome (SNOMED 237055002)
    ('cushings',         [195212]),            # Hypercortisolism / Cushing's syndrome (SNOMED 47270006)
    ('acromegaly',       [4253197]),           # Acromegaly (SNOMED 74107003)

    # Pregnancy-related (coded 0 for males)
    ('preeclampsia',     [439393, 443700]),    # Pre-eclampsia (SNOMED 398254007) + eclampsia (15938005)
    ('gest_dm',          [4024659]),           # Gestational diabetes mellitus
    ('preterm',          [4086393]),           # Premature delivery (SNOMED 282020008)
    ('preg_loss',        [4067106]),           # Miscarriage / pregnancy loss
]

CHEMO_FLAGS = [
    # Cardiotoxic chemotherapy — anthracyclines + trastuzumab
    # Daunorubicin and idarubicin not found as standard RxNorm ingredients in this CDR
    ('cardiotoxic_chemo', [1338512,   # doxorubicin  (RxNorm 3639)
                           1344354,   # epirubicin   (RxNorm 3995)
                           1387104]), # trastuzumab  (RxNorm 224905)
]

# IBD composite (Crohn's OR UC) — derived after COND_FLAGS built
IBD_COLS = ['crohns', 'ulc_colitis']

# Binary columns — written as int in final CSV
BINARY_COLS = [
    'mace3_event', 'has_mi', 'has_stroke', 'has_cvd_death', 'has_any_death',
    'htn', 't2dm', 'obesity_dx', 'cad_prev', 'ckd', 'current_smoker',
    # medications
    'statin', 'ace_inhibitor', 'arb', 'beta_blocker', 'ccb', 'diuretic',
    'aspirin', 'p2y12', 'oral_anticoag', 'arni',
    'metformin', 'sglt2i', 'glp1ra', 'dpp4i', 'sulfonylurea', 'tzd', 'insulin_any',
    'any_antihtn', 'any_dm_med',
    # additional comorbidities
    't1dm', 'fh', 'pad',
    'metabolic_syndrome', 'osa', 'heart_failure', 'afib',
    'ra', 'sle', 'psoriasis', 'crohns', 'ulc_colitis', 'ibd', 'hiv',
    'hypothyroidism', 'hyperthyroidism', 'pcos', 'cushings', 'acromegaly',
    'cardiotoxic_chemo',
    'preeclampsia', 'gest_dm', 'preterm', 'preg_loss',
]


def sep(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ── STEP 1: Load base cohort ──────────────────────────────────────────────────
sep("STEP 1: Load cohort_with_covariates.csv")
master = pd.read_csv(f'{WORKSPACE}/cohort_with_covariates.csv',
                     dtype={'person_id': str})
# Drop old incorrect medication columns — will be replaced in Step 6
old_med_cols = ['statin', 'antihtn', 'metformin', 'antiplatelet']
master = master.drop(columns=[c for c in old_med_cols if c in master.columns])
print(f"  {len(master):,} rows × {master.shape[1]} cols (old med flags dropped)")
gc.collect()


# ── STEP 2: Extra PRS ─────────────────────────────────────────────────────────
sep("STEP 2: Merge extra PRS (LDLC, OBESITY, SBP, T2D)")
for trait in EXTRA_PRS:
    d = f'{WORKSPACE}/PRS_Scores/{trait}'
    f = [x for x in os.listdir(d) if not x.startswith('.')][0]
    path = os.path.join(d, f)
    tmp = pd.read_csv(path) if path.endswith(('.csv', '.txt')) else pd.read_csv(path, sep='\t')
    tmp.columns = ['person_id', f'prs_{trait.lower()}']
    tmp['person_id'] = tmp['person_id'].astype(str)
    master = master.merge(tmp, on='person_id', how='left')
    print(f"  {trait}: {master[f'prs_{trait.lower()}'].notna().sum():,} matched")
    del tmp; gc.collect()


# ── STEP 3: Get zip3 from CDR ─────────────────────────────────────────────────
sep("STEP 3: Get zip3 per person (observation concept 3043579)")
q = f"""
WITH ranked AS (
    SELECT
        CAST(o.person_id AS STRING) AS person_id,
        SUBSTR(o.value_as_string, 1, 3) AS zip3,
        ROW_NUMBER() OVER (PARTITION BY o.person_id ORDER BY o.observation_date DESC) AS rn
    FROM `{CDR}.observation` o
    WHERE o.observation_concept_id = 3043579
      AND o.value_as_string IS NOT NULL
)
SELECT person_id, zip3 FROM ranked WHERE rn = 1
"""
zip3_df = client.query(q).to_dataframe()
zip3_df['person_id'] = zip3_df['person_id'].astype(str)
zip3_df = zip3_df.drop_duplicates('person_id')
print(f"  zip3 fetched for {len(zip3_df):,} persons in CDR")

master = master.merge(zip3_df, on='person_id', how='left')
print(f"  zip3 coverage in cohort: {master['zip3'].notna().sum():,} / {len(master):,} "
      f"({100*master['zip3'].notna().mean():.1f}%)")
del zip3_df; gc.collect()


# ── STEP 4: Geographic exclusion ─────────────────────────────────────────────
sep("STEP 4: Exclude AK / HI / territories / missing zip3")
zip3_key = (master['zip3'].astype(str).str.strip()
            .str.extract(r'(\d+)', expand=False).str.zfill(3))
exclude = zip3_key.isin(EXCLUDE_ZIP3) | master['zip3'].isna()
n_before = len(master)
master = master[~exclude].copy()
print(f"  Excluded: {n_before - len(master):,}")
print(f"  Final cohort: {len(master):,}")
gc.collect()


# ── STEP 5: Merge exposome files ──────────────────────────────────────────────
sep("STEP 5: Merge exposome files")
master['_z3'] = (master['zip3'].astype(str).str.strip()
                 .str.extract(r'(\d+)', expand=False).str.zfill(3))

for fname, pfx in EXPOSOME_FILES.items():
    fpath = os.path.join(EXPOSOME_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  {fname}: NOT FOUND — skipping"); continue
    exp = pd.read_csv(fpath)
    zcol = next((c for c in exp.columns if 'zip' in c.lower()), exp.columns[0])
    exp['_ez'] = (exp[zcol].astype(str).str.strip()
                  .str.extract(r'(\d+)', expand=False).str.zfill(3))
    exp = exp.drop(columns=[c for c in exp.columns if 'zip' in c.lower() and c != '_ez'],
                   errors='ignore')
    exp = exp.rename(columns={c: f'{pfx}_{c}' for c in exp.columns if c != '_ez'})
    for c in exp.select_dtypes('float64').columns:
        exp[c] = exp[c].astype('float32')
    master = master.merge(exp, left_on='_z3', right_on='_ez', how='left')
    master = master.drop(columns=['_ez'], errors='ignore')
    probe = next(c for c in master.columns if c.startswith(f'{pfx}_'))
    n_matched = master[probe].notna().sum()
    print(f"  {pfx}: {n_matched:,} / {len(master):,} matched ({100*n_matched/len(master):.1f}%)")
    del exp; gc.collect()

master = master.drop(columns=['_z3'], errors='ignore')


# ── STEP 6: Medications (re-queried with correct concept IDs) ────────────────
sep("STEP 6: Medication flags")
print(f"\n  {'Flag':<16} {'N persons':>10}  {'%':>6}  Drugs included")
print(f"  {'─'*65}")

cohort_ids = master['person_id'].tolist()

for col, concept_ids in MED_FLAGS:
    ids_str = ', '.join(str(i) for i in concept_ids)
    q_med = f"""
    SELECT DISTINCT CAST(de.person_id AS STRING) AS person_id
    FROM `{CDR}.drug_exposure` de
    JOIN `{CDR}.concept_ancestor` ca
        ON de.drug_concept_id = ca.descendant_concept_id
    WHERE ca.ancestor_concept_id IN ({ids_str})
    """
    med_df = client.query(q_med).to_dataframe()
    med_df['person_id'] = med_df['person_id'].astype(str)
    exposed = set(med_df['person_id']) & set(cohort_ids)
    master[col] = master['person_id'].isin(exposed).astype(int)
    n = master[col].sum()
    print(f"  {col:<16} {n:>10,}  {100*n/len(master):>5.1f}%")
    del med_df; gc.collect()

# Composite flags
master['any_antihtn'] = master[ANY_ANTIHTN_COLS].max(axis=1).astype(int)
n = master['any_antihtn'].sum()
print(f"  {'any_antihtn':<16} {n:>10,}  {100*n/len(master):>5.1f}%  (ACE|ARB|BB|CCB|diuretic)")

dm_cols_present = [c for c in ANY_DM_MED_COLS if c in master.columns]
master['any_dm_med'] = master[dm_cols_present].max(axis=1).astype(int)
n = master['any_dm_med'].sum()
print(f"  {'any_dm_med':<16} {n:>10,}  {100*n/len(master):>5.1f}%  (metformin|SGLT2i|GLP1RA|DPP4i|SU|TZD|insulin)")


# ── STEP 6b: Additional comorbidities ────────────────────────────────────────
sep("STEP 6b: Additional comorbidities (conditions + cardiotoxic chemo)")
print(f"\n  {'Flag':<22} {'N persons':>10}  {'%':>6}")
print(f"  {'─'*45}")

for col, concept_ids in COND_FLAGS:
    if concept_ids == [0]:
        master[col] = 0  # pending concept ID — run concept_id_lookup.py
        print(f"  {col:<22} {'PENDING':>10}  (concept ID not yet verified)")
        continue
    ids_str = ', '.join(str(i) for i in concept_ids)
    q_cond = f"""
    SELECT DISTINCT CAST(co.person_id AS STRING) AS person_id
    FROM `{CDR}.condition_occurrence` co
    JOIN `{CDR}.concept_ancestor` ca
        ON co.condition_concept_id = ca.descendant_concept_id
    WHERE ca.ancestor_concept_id IN ({ids_str})
    """
    cond_df = client.query(q_cond).to_dataframe()
    cond_df['person_id'] = cond_df['person_id'].astype(str)
    exposed = set(cond_df['person_id']) & set(cohort_ids)
    master[col] = master['person_id'].isin(exposed).astype(int)
    n = master[col].sum()
    print(f"  {col:<22} {n:>10,}  {100*n/len(master):>5.1f}%")
    del cond_df; gc.collect()

for col, concept_ids in CHEMO_FLAGS:
    if concept_ids == [0]:
        master[col] = 0  # pending concept ID — run concept_id_lookup.py
        print(f"  {col:<22} {'PENDING':>10}  [drug_exposure — concept ID not yet verified]")
        continue
    ids_str = ', '.join(str(i) for i in concept_ids)
    q_chemo = f"""
    SELECT DISTINCT CAST(de.person_id AS STRING) AS person_id
    FROM `{CDR}.drug_exposure` de
    JOIN `{CDR}.concept_ancestor` ca
        ON de.drug_concept_id = ca.descendant_concept_id
    WHERE ca.ancestor_concept_id IN ({ids_str})
    """
    chemo_df = client.query(q_chemo).to_dataframe()
    chemo_df['person_id'] = chemo_df['person_id'].astype(str)
    exposed = set(chemo_df['person_id']) & set(cohort_ids)
    master[col] = master['person_id'].isin(exposed).astype(int)
    n = master[col].sum()
    print(f"  {col:<22} {n:>10,}  {100*n/len(master):>5.1f}%  [drug_exposure]")
    del chemo_df; gc.collect()

# IBD composite
master['ibd'] = master[IBD_COLS].max(axis=1).astype(int)
n = master['ibd'].sum()
print(f"  {'ibd':<22} {n:>10,}  {100*n/len(master):>5.1f}%  (Crohn's | UC)")


# ── STEP 7: Smoking fix ───────────────────────────────────────────────────────
sep("STEP 7: Fix smoking (concept 40766307)")
q_smoke = f"""
WITH ranked AS (
    SELECT
        CAST(o.person_id AS STRING) AS person_id,
        c_val.concept_name AS answer,
        ROW_NUMBER() OVER (PARTITION BY o.person_id ORDER BY o.observation_date DESC) AS rn
    FROM `{CDR}.observation` o
    LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
    WHERE o.observation_concept_id = 40766307
      AND o.value_as_concept_id IS NOT NULL
)
SELECT person_id, answer FROM ranked WHERE rn = 1
"""
smoke = client.query(q_smoke).to_dataframe()
smoke['person_id'] = smoke['person_id'].astype(str)
print(f"  People with answer: {len(smoke):,}")
print(smoke['answer'].value_counts().to_string())

current_ids = set(smoke.loc[
    smoke['answer'].str.contains('every day|some day', case=False, na=False), 'person_id'])
master['current_smoker'] = master['person_id'].isin(current_ids).astype(int)
n_ans = master['person_id'].isin(smoke['person_id']).sum()
n_cur = master['current_smoker'].sum()
print(f"\n  Answered (in cohort): {n_ans:,} ({100*n_ans/len(master):.1f}%)")
print(f"  Current smoker:       {n_cur:,} ({100*n_cur/len(master):.1f}%)")
del smoke, current_ids; gc.collect()


# ── STEP 8: Z-score PRS ───────────────────────────────────────────────────────
sep("STEP 8: Z-score PRS columns")
prs_cols = [c for c in master.columns if c.startswith('prs_') or c == 'cad_prs']
for col in prs_cols:
    s = master[col].dropna().astype('float64')
    mu, sd = float(s.mean()), float(s.std())
    master[col] = ((master[col].astype('float64') - mu) / sd).round(5).astype('float32')
    print(f"  {col:<20} raw mean={mu:>10.3f}  sd={sd:>8.3f}  → z-scored")


# ── STEP 9: Save ──────────────────────────────────────────────────────────────
sep("STEP 9: Save master_dataset.csv")

for c in BINARY_COLS:
    if c in master.columns:
        master[c] = master[c].fillna(0).astype(int)

for c in master.select_dtypes(include=['float32', 'float64']).columns:
    master[c] = master[c].round(4)

out_path = f'{WORKSPACE}/master_dataset.csv'
master.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / 1e6
print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
print(f"  Shape: {master.shape}")

del master; gc.collect()

result = subprocess.run(['gsutil', 'cp', out_path, f'{BUCKET}/master_dataset.csv'],
                        capture_output=True, text=True, timeout=300)
print(f"  {'Copied to bucket.' if result.returncode == 0 else f'gsutil failed: {result.stderr}'}")
print("\nDONE — run table1.py to see the summary")
