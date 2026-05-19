"""
CELL 2 — Define genetic cohort and audit MACE data availability
===============================================================
This cell:
  1. Loads the CAD PRS file to get the list of participants with genetic data
  2. Queries the All of Us CDR (BigQuery) for MACE events in that cohort
  3. Prints a detailed audit so you know exactly what outcome data is available
     before committing to any analysis decisions

MACE = Major Adverse Cardiovascular Events
  - Myocardial infarction (MI)
  - Stroke (ischemic + hemorrhagic combined)
  - Cardiovascular death (CVD death)

All clinical data in All of Us is stored in OMOP CDM tables accessed via BigQuery.
Concept IDs follow OMOP/SNOMED conventions. We use concept_ancestor to capture all
descendant concepts under each parent (e.g. "acute MI" rolls up under "MI").
"""

import os
import pandas as pd
import numpy as np
from google.cloud import bigquery

# ── Paths and clients ──────────────────────────────────────────────────────
WORKSPACE = '/home/jupyter/workspaces/geneexposome'

# WORKSPACE_CDR is the BigQuery dataset name for the current CDR version.
# It looks like: "fc-aou-cdr-prod-ct.C2022Q4R9" — set automatically by AoU.
CDR = os.environ['WORKSPACE_CDR']

# BigQuery client — uses your workspace service account automatically.
client = bigquery.Client()

print(f"CDR dataset: {CDR}")
print(f"Workspace:   {WORKSPACE}\n")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load CAD PRS to define the genetic cohort
# ══════════════════════════════════════════════════════════════════════════
# The PRS file was downloaded in Cell 1. We load it here to get person_ids
# for everyone with a valid CAD polygenic risk score.
# Expected columns: person_id, prs_score (adjust column names to match your file).

print("=" * 60)
print("STEP 1: Loading CAD PRS")
print("=" * 60)

cad_prs_dir = f'{WORKSPACE}/PRS_Scores/CAD'
cad_files = [f for f in os.listdir(cad_prs_dir) if not f.startswith('.')]
print(f"Files found in CAD directory: {cad_files}")

# Load whichever file is present — adjust if you have multiple shards
cad_prs_path = os.path.join(cad_prs_dir, cad_files[0])
print(f"Loading: {cad_prs_path}")

# Read the PRS file — it may be CSV, TSV, or parquet
if cad_prs_path.endswith('.parquet'):
    prs_df = pd.read_parquet(cad_prs_path)
elif cad_prs_path.endswith('.csv'):
    prs_df = pd.read_csv(cad_prs_path)
else:
    prs_df = pd.read_csv(cad_prs_path, sep='\t')

print(f"\nPRS file shape: {prs_df.shape}")
print(f"Columns: {list(prs_df.columns)}")
print(f"\nFirst 5 rows:")
print(prs_df.head())

# Standardise column names — rename to person_id and cad_prs
# Adjust the right-hand side strings to match whatever your file actually uses
prs_df = prs_df.rename(columns={
    prs_df.columns[0]: 'person_id',   # first column assumed to be ID
    prs_df.columns[1]: 'cad_prs'      # second column assumed to be score
})
prs_df['person_id'] = prs_df['person_id'].astype(str)

print(f"\nGenetic cohort size (has CAD PRS): {len(prs_df):,} participants")
print(f"PRS score range: {prs_df['cad_prs'].min():.4f} – {prs_df['cad_prs'].max():.4f}")
print(f"PRS score mean ± SD: {prs_df['cad_prs'].mean():.4f} ± {prs_df['cad_prs'].std():.4f}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Pull basic demographics for the genetic cohort from BigQuery
# ══════════════════════════════════════════════════════════════════════════
# The `person` table has one row per participant: age, sex, race, ethnicity.
# We join to our PRS person_id list to restrict to the genetic cohort.

print("\n" + "=" * 60)
print("STEP 2: Demographics for genetic cohort")
print("=" * 60)

demo_query = f"""
SELECT
    p.person_id,
    p.year_of_birth,
    -- Convert birth year to approximate age at study baseline (2022 CDR)
    (2022 - p.year_of_birth) AS age_approx,
    p.sex_at_birth_concept_id,
    sc.concept_name AS sex_at_birth,
    p.race_concept_id,
    rc.concept_name AS race,
    p.ethnicity_concept_id,
    ec.concept_name AS ethnicity
FROM
    `{CDR}.person` p
LEFT JOIN `{CDR}.concept` sc ON p.sex_at_birth_concept_id = sc.concept_id
LEFT JOIN `{CDR}.concept` rc ON p.race_concept_id = rc.concept_id
LEFT JOIN `{CDR}.concept` ec ON p.ethnicity_concept_id = ec.concept_id
"""

demo_df = client.query(demo_query).to_dataframe()
demo_df['person_id'] = demo_df['person_id'].astype(str)

# Restrict to genetic cohort
demo_genetic = demo_df[demo_df['person_id'].isin(prs_df['person_id'])].copy()

print(f"\nTotal participants in CDR:         {len(demo_df):,}")
print(f"Participants with CAD PRS:         {len(demo_genetic):,}")
print(f"PRS coverage (% of CDR):           {100*len(demo_genetic)/len(demo_df):.1f}%")

print(f"\nAge distribution (genetic cohort):")
print(demo_genetic['age_approx'].describe().round(1))

print(f"\nSex at birth:")
print(demo_genetic['sex_at_birth'].value_counts())

print(f"\nRace:")
print(demo_genetic['race'].value_counts())


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Query MACE events in the genetic cohort
# ══════════════════════════════════════════════════════════════════════════
# OMOP concept IDs used below:
#
#   Myocardial Infarction (MI)
#     4329847  = Myocardial infarction (parent SNOMED concept)
#     All descendant concepts captured via concept_ancestor
#
#   Stroke
#     443454   = Cerebrovascular accident (stroke parent)
#     375557   = Ischemic stroke
#     432923   = Hemorrhagic stroke
#     All descendants captured via concept_ancestor
#
#   Cardiovascular death
#     Uses the `death` table combined with `condition_occurrence`
#     CVD death parent concept: 4185932 (Cardiovascular disease causing death)
#     We also flag any death occurring within 28 days of an MI/stroke code
#     as a pragmatic CVD death proxy (common in EHR research).
#
# NOTE: condition_occurrence captures ICD codes mapped to OMOP concepts.
# condition_start_date is the date the diagnosis was recorded.
# For incident analysis you'll later filter to events AFTER enrollment date.

# ── 3a. Myocardial Infarction ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3a: Myocardial Infarction")
print("=" * 60)

mi_query = f"""
SELECT
    co.person_id,
    co.condition_concept_id,
    c.concept_name AS condition_name,
    co.condition_start_date,
    co.condition_source_value,   -- original ICD code before OMOP mapping
    co.condition_type_concept_id,
    ct.concept_name AS condition_type  -- inpatient, outpatient, EHR claim, etc.
FROM
    `{CDR}.condition_occurrence` co
JOIN `{CDR}.concept` c
    ON co.condition_concept_id = c.concept_id
JOIN `{CDR}.concept` ct
    ON co.condition_type_concept_id = ct.concept_id
-- concept_ancestor expands the MI parent concept to all child concepts
JOIN `{CDR}.concept_ancestor` ca
    ON co.condition_concept_id = ca.descendant_concept_id
WHERE
    ca.ancestor_concept_id = 4329847  -- Myocardial infarction (SNOMED parent)
"""

mi_df = client.query(mi_query).to_dataframe()
mi_df['person_id'] = mi_df['person_id'].astype(str)
mi_genetic = mi_df[mi_df['person_id'].isin(prs_df['person_id'])].copy()

# One row per person (keep earliest date for later incident analysis)
mi_first = (mi_genetic.sort_values('condition_start_date')
                       .groupby('person_id')
                       .first()
                       .reset_index()
                       [['person_id', 'condition_start_date']]
                       .rename(columns={'condition_start_date': 'mi_date'}))

print(f"MI records in genetic cohort (all occurrences): {len(mi_genetic):,}")
print(f"Unique participants with any MI code:           {mi_genetic['person_id'].nunique():,}")
print(f"\nTop MI concept names (most frequent):")
print(mi_genetic['condition_name'].value_counts().head(10))
print(f"\nCondition type breakdown (encounter setting):")
print(mi_genetic['condition_type'].value_counts())
print(f"\nDate range of MI records:")
print(f"  Earliest: {mi_genetic['condition_start_date'].min()}")
print(f"  Latest:   {mi_genetic['condition_start_date'].max()}")
print(f"  Null dates: {mi_genetic['condition_start_date'].isna().sum()}")


# ── 3b. Stroke ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3b: Stroke")
print("=" * 60)

stroke_query = f"""
SELECT
    co.person_id,
    co.condition_concept_id,
    c.concept_name AS condition_name,
    co.condition_start_date,
    co.condition_source_value,
    co.condition_type_concept_id,
    ct.concept_name AS condition_type
FROM
    `{CDR}.condition_occurrence` co
JOIN `{CDR}.concept` c
    ON co.condition_concept_id = c.concept_id
JOIN `{CDR}.concept` ct
    ON co.condition_type_concept_id = ct.concept_id
JOIN `{CDR}.concept_ancestor` ca
    ON co.condition_concept_id = ca.descendant_concept_id
WHERE
    -- 443454 = Cerebrovascular accident (stroke)
    -- Including both ischemic (375557) and hemorrhagic (432923)
    ca.ancestor_concept_id IN (443454, 375557, 432923)
"""

stroke_df = client.query(stroke_query).to_dataframe()
stroke_df['person_id'] = stroke_df['person_id'].astype(str)
stroke_genetic = stroke_df[stroke_df['person_id'].isin(prs_df['person_id'])].copy()

stroke_first = (stroke_genetic.sort_values('condition_start_date')
                               .groupby('person_id')
                               .first()
                               .reset_index()
                               [['person_id', 'condition_start_date']]
                               .rename(columns={'condition_start_date': 'stroke_date'}))

print(f"Stroke records in genetic cohort (all occurrences): {len(stroke_genetic):,}")
print(f"Unique participants with any stroke code:           {stroke_genetic['person_id'].nunique():,}")
print(f"\nTop stroke concept names (most frequent):")
print(stroke_genetic['condition_name'].value_counts().head(10))
print(f"\nCondition type breakdown:")
print(stroke_genetic['condition_type'].value_counts())
print(f"\nDate range of stroke records:")
print(f"  Earliest: {stroke_genetic['condition_start_date'].min()}")
print(f"  Latest:   {stroke_genetic['condition_start_date'].max()}")
print(f"  Null dates: {stroke_genetic['condition_start_date'].isna().sum()}")


# ── 3c. Death (all-cause and CVD-specific) ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 3c: Death records")
print("=" * 60)

# The `death` table has one row per deceased participant.
# cause_concept_id is the OMOP concept for cause of death — this is often
# NULL or coded as "Unknown" in EHR-derived data.
# We separately flag CVD death using concept_ancestor on cardiovascular concepts.

death_query = f"""
SELECT
    d.person_id,
    d.death_date,
    d.death_type_concept_id,
    dt.concept_name AS death_type,       -- EHR, NDI, survey self-report, etc.
    d.cause_concept_id,
    cc.concept_name AS cause_of_death,
    -- Flag if cause of death falls under cardiovascular disease (4185932)
    CASE
        WHEN ca.ancestor_concept_id IS NOT NULL THEN TRUE
        ELSE FALSE
    END AS is_cvd_death
FROM
    `{CDR}.death` d
LEFT JOIN `{CDR}.concept` dt
    ON d.death_type_concept_id = dt.concept_id
LEFT JOIN `{CDR}.concept` cc
    ON d.cause_concept_id = cc.concept_id
-- Left join concept_ancestor to check if cause falls under CVD
LEFT JOIN `{CDR}.concept_ancestor` ca
    ON d.cause_concept_id = ca.descendant_concept_id
    AND ca.ancestor_concept_id = 4185932  -- Cardiovascular disease (SNOMED)
"""

death_df = client.query(death_query).to_dataframe()
death_df['person_id'] = death_df['person_id'].astype(str)
death_genetic = death_df[death_df['person_id'].isin(prs_df['person_id'])].copy()

print(f"Total deaths in genetic cohort:        {len(death_genetic):,}")
print(f"  With a recorded cause concept:       {death_genetic['cause_concept_id'].notna().sum():,}")
print(f"  CVD death (cause maps to CVD):       {death_genetic['is_cvd_death'].sum():,}")
print(f"\nDeath type breakdown (data source):")
print(death_genetic['death_type'].value_counts())
print(f"\nTop causes of death (where recorded):")
print(death_genetic['cause_of_death'].value_counts().head(15))
print(f"\nDate range of death records:")
print(f"  Earliest: {death_genetic['death_date'].min()}")
print(f"  Latest:   {death_genetic['death_date'].max()}")
print(f"  Null dates: {death_genetic['death_date'].isna().sum()}")


# ── 3d. Enrollment date (observation period start) ──────────────────────
# We need enrollment/consent date to define INCIDENT events (post-enrollment only).
# In All of Us the observation_period table captures the span of EHR data
# per person. The earliest observation_period_start_date approximates EHR start.
# The AoU consent date is in the survey / person table.

print("\n" + "=" * 60)
print("STEP 3d: Observation period (EHR span) for genetic cohort")
print("=" * 60)

obs_query = f"""
SELECT
    person_id,
    MIN(observation_period_start_date) AS ehr_start,
    MAX(observation_period_end_date)   AS ehr_end,
    COUNT(*) AS n_periods
FROM `{CDR}.observation_period`
GROUP BY person_id
"""

obs_df = client.query(obs_query).to_dataframe()
obs_df['person_id'] = obs_df['person_id'].astype(str)
obs_genetic = obs_df[obs_df['person_id'].isin(prs_df['person_id'])].copy()

print(f"Participants with any observation period:  {len(obs_genetic):,}")
print(f"Missing obs period (no EHR):               {len(prs_df) - len(obs_genetic):,}")
print(f"\nEHR start date distribution:")
print(obs_genetic['ehr_start'].describe())
print(f"\nEHR end date distribution:")
print(obs_genetic['ehr_end'].describe())


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Build the 3-point MACE summary table
# ══════════════════════════════════════════════════════════════════════════
# Merge all outcome flags onto the genetic cohort so you can see how many
# participants have each combination of events.

print("\n" + "=" * 60)
print("STEP 4: MACE summary for genetic cohort")
print("=" * 60)

# Start with all PRS participants
cohort = prs_df[['person_id', 'cad_prs']].copy()

# Merge MI flag + date
cohort = cohort.merge(mi_first, on='person_id', how='left')
cohort['has_mi'] = cohort['mi_date'].notna()

# Merge stroke flag + date
cohort = cohort.merge(stroke_first, on='person_id', how='left')
cohort['has_stroke'] = cohort['stroke_date'].notna()

# Merge CVD death flag + date
cvd_death_first = (death_genetic[death_genetic['is_cvd_death']]
                   [['person_id', 'death_date']]
                   .rename(columns={'death_date': 'cvd_death_date'}))
cohort = cohort.merge(cvd_death_first, on='person_id', how='left')
cohort['has_cvd_death'] = cohort['cvd_death_date'].notna()

# Also keep all-cause death date (useful for censoring in survival analysis)
all_death = death_genetic[['person_id', 'death_date']].rename(
    columns={'death_date': 'death_date_any'})
cohort = cohort.merge(all_death, on='person_id', how='left')
cohort['has_any_death'] = cohort['death_date_any'].notna()

# 3-point MACE: any of the three
cohort['has_mace'] = cohort['has_mi'] | cohort['has_stroke'] | cohort['has_cvd_death']

# Convert all date columns to datetime64 before taking row-wise min.
# BigQuery returns datetime.date objects; NaN fills (from left-merge misses) are
# float, so numpy's <= comparison blows up unless everything is the same dtype.
for col in ['mi_date', 'stroke_date', 'cvd_death_date']:
    cohort[col] = pd.to_datetime(cohort[col], errors='coerce')

# Earliest MACE date (for survival analysis)
cohort['mace_date'] = cohort[['mi_date', 'stroke_date', 'cvd_death_date']].min(axis=1)

# ── Print the summary ─────────────────────────────────────────────────────
n = len(cohort)
print(f"\nGenetic cohort N = {n:,}\n")

print(f"{'Outcome':<35} {'N':>8} {'%':>7}")
print("-" * 52)
print(f"{'Any 3-pt MACE':<35} {cohort['has_mace'].sum():>8,} {100*cohort['has_mace'].mean():>6.1f}%")
print(f"{'  Myocardial Infarction (MI)':<35} {cohort['has_mi'].sum():>8,} {100*cohort['has_mi'].mean():>6.1f}%")
print(f"{'  Stroke':<35} {cohort['has_stroke'].sum():>8,} {100*cohort['has_stroke'].mean():>6.1f}%")
print(f"{'  CVD Death':<35} {cohort['has_cvd_death'].sum():>8,} {100*cohort['has_cvd_death'].mean():>6.1f}%")
print(f"{'All-cause death':<35} {cohort['has_any_death'].sum():>8,} {100*cohort['has_any_death'].mean():>6.1f}%")

print(f"\nMACE component overlap:")
mi_and_stroke    = (cohort['has_mi'] & cohort['has_stroke']).sum()
mi_and_cvddeath  = (cohort['has_mi'] & cohort['has_cvd_death']).sum()
str_and_cvddeath = (cohort['has_stroke'] & cohort['has_cvd_death']).sum()
all_three        = (cohort['has_mi'] & cohort['has_stroke'] & cohort['has_cvd_death']).sum()
print(f"  MI + Stroke:          {mi_and_stroke:,}")
print(f"  MI + CVD death:       {mi_and_cvddeath:,}")
print(f"  Stroke + CVD death:   {str_and_cvddeath:,}")
print(f"  All three:            {all_three:,}")

print(f"\nDate availability for MACE events:")
print(f"  MI with date:        {cohort['mi_date'].notna().sum():,} / {cohort['has_mi'].sum():,}")
print(f"  Stroke with date:    {cohort['stroke_date'].notna().sum():,} / {cohort['has_stroke'].sum():,}")
print(f"  CVD death with date: {cohort['cvd_death_date'].notna().sum():,} / {cohort['has_cvd_death'].sum():,}")

print(f"\nMACE date range (for survival analysis feasibility):")
print(f"  Earliest MACE:  {cohort['mace_date'].min()}")
print(f"  Latest MACE:    {cohort['mace_date'].max()}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Additional data availability checks
# ══════════════════════════════════════════════════════════════════════════
# Check for data domains that will feed into downstream analyses:
# labs, medications, smoking status, BMI.

print("\n" + "=" * 60)
print("STEP 5: Additional data availability in genetic cohort")
print("=" * 60)

# ── 5a. Key measurements (labs + vitals) ─────────────────────────────────
# Concept IDs: LDL=3007070, SBP=3004249, DBP=3012888, BMI=3038553, HbA1c=3004410
lab_check_query = f"""
SELECT
    m.measurement_concept_id,
    c.concept_name,
    COUNT(DISTINCT m.person_id) AS n_participants,
    COUNT(*) AS n_records,
    ROUND(AVG(m.value_as_number), 2) AS mean_value,
    MIN(m.measurement_date) AS earliest,
    MAX(m.measurement_date) AS latest
FROM `{CDR}.measurement` m
JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
WHERE m.measurement_concept_id IN (
    3007070,  -- LDL cholesterol
    3004249,  -- Systolic blood pressure
    3012888,  -- Diastolic blood pressure
    3038553,  -- Body mass index
    3004410,  -- HbA1c
    3027114   -- Total cholesterol
)
GROUP BY m.measurement_concept_id, c.concept_name
ORDER BY n_participants DESC
"""

lab_df = client.query(lab_check_query).to_dataframe()
print("\nKey labs/vitals available (all CDR, will subset to genetic cohort in cleaning step):")
print(lab_df.to_string(index=False))

# ── 5b. Smoking status ────────────────────────────────────────────────────
# Smoking is captured both in observation table (survey) and condition_occurrence
smoking_query = f"""
SELECT
    c.concept_name AS smoking_status,
    COUNT(DISTINCT o.person_id) AS n_participants
FROM `{CDR}.observation` o
JOIN `{CDR}.concept` c ON o.value_as_concept_id = c.concept_id
JOIN `{CDR}.concept_ancestor` ca
    ON o.observation_concept_id = ca.descendant_concept_id
WHERE ca.ancestor_concept_id = 4041306  -- Tobacco smoking behavior (SNOMED)
GROUP BY c.concept_name
ORDER BY n_participants DESC
"""

smoking_df = client.query(smoking_query).to_dataframe()
print(f"\nSmoking status records (observation table):")
print(smoking_df.to_string(index=False))

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
print("""
Next decisions based on these numbers:
  1. Is MACE prevalence high enough for XGBoost? (target: >2,000 events)
  2. What % of MACE events have a usable date? (need >80% for survival analysis)
  3. How far back does EHR data go? (affects prevalent vs incident case definition)
  4. Are CVD deaths captured well enough, or should 2-pt MACE (MI+stroke) be primary?
""")

# ── Save cohort skeleton for next steps ──────────────────────────────────
save_path = f'{WORKSPACE}/cohort_skeleton.parquet'
cohort.to_parquet(save_path, index=False)
print(f"Cohort skeleton saved to: {save_path}")
print(f"Shape: {cohort.shape}")
print(f"Columns: {list(cohort.columns)}")
