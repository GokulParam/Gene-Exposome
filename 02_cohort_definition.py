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

# CVD death ascertainment strategy:
# The original concept_ancestor query using a single parent (4185932) only
# returned 1 CVD death because the OMOP hierarchy in this CDR is incomplete —
# concepts like "Cardiac arrest", "Heart failure", and "Cardiogenic shock"
# are not connected as descendants of that parent.
#
# We use a two-pronged approach instead:
#   (a) Explicit CVD cause concepts — a curated list of cardiovascular
#       cause-of-death concept IDs that covers the concepts visibly present
#       in this CDR's death table
#   (b) 30-day post-MI/stroke proxy — any death within 30 days of a coded
#       MI or stroke, regardless of recorded cause
#
# CVD death = (a) OR (b)
# This is the 3rd component of 3-point MACE.
#
# LIMITATION (to state in paper): cause-of-death coding captures only a
# fraction of true CVD deaths in EHR-derived data. The majority of deaths
# (est. ~75%) have no cause recorded. Our CVD death component therefore
# relies primarily on the 30-day proximity proxy, which is a standard
# approach in EHR-based cardiovascular studies when NDI linkage is absent.

# Cardiovascular cause-of-death concept IDs present in this CDR.
# Identified empirically from the cause_of_death breakdown in Step 3c.
# Covers: cardiac arrest, heart failure, cardiogenic shock, sudden cardiac
# death, hypertensive heart disease, other CVD, cerebrovascular disease.
CVD_CAUSE_CONCEPTS = (
    321042,    # Cardiac arrest
    4059796,   # Sudden cardiac death
    316139,    # Heart failure
    40479586,  # Cardiogenic shock
    4108812,   # Other specified heart disease
    312927,    # Hypertensive heart disease
    4108814,   # Ischemic heart disease
    443454,    # Cerebrovascular accident
    432923,    # Hemorrhagic stroke
    375557,    # Ischemic stroke
    4185932,   # Cardiovascular disease (keep original, catches any mapped descendants)
)

cvd_concepts_sql = ', '.join(str(c) for c in CVD_CAUSE_CONCEPTS)

death_query = f"""
SELECT
    d.person_id,
    d.death_date,
    d.death_type_concept_id,
    dt.concept_name AS death_type,
    d.cause_concept_id,
    cc.concept_name AS cause_of_death,
    -- (a) Explicit CVD cause: cause concept is in our curated CVD list
    --     OR is a descendant of any concept in that list
    CASE
        WHEN ca.ancestor_concept_id IS NOT NULL THEN TRUE
        ELSE FALSE
    END AS is_cvd_death_coded
FROM `{CDR}.death` d
LEFT JOIN `{CDR}.concept` dt  ON d.death_type_concept_id = dt.concept_id
LEFT JOIN `{CDR}.concept` cc  ON d.cause_concept_id = cc.concept_id
LEFT JOIN `{CDR}.concept_ancestor` ca
    ON d.cause_concept_id = ca.descendant_concept_id
    AND ca.ancestor_concept_id IN ({cvd_concepts_sql})
"""

death_df = client.query(death_query).to_dataframe()
death_df['person_id'] = death_df['person_id'].astype(str)
death_genetic = death_df[death_df['person_id'].isin(prs_df['person_id'])].copy()

print(f"Total deaths in genetic cohort:           {len(death_genetic):,}")
print(f"  With any cause concept recorded:        {death_genetic['cause_concept_id'].notna().sum():,}")
print(f"  With NULL cause (no coding):            {death_genetic['cause_concept_id'].isna().sum():,}")
print(f"  CVD death (explicit cause coding):      {death_genetic['is_cvd_death_coded'].sum():,}")
print(f"\nTop causes of death (where recorded):")
print(death_genetic['cause_of_death'].value_counts().head(15))
print(f"\nDate range of death records:")
print(f"  Earliest: {death_genetic['death_date'].min()}")
print(f"  Latest:   {death_genetic['death_date'].max()}")


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
# STEP 4 — Build outcome flags and the cohort skeleton
# ══════════════════════════════════════════════════════════════════════════
#
# PRIMARY OUTCOME — 3-point MACE (MI, stroke, or CVD death)
#   CVD death = explicitly coded CVD cause (from curated concept list)
#               OR death within 30 days of a coded MI/stroke (proxy)
#   Rationale: combines all identifiable CVD deaths from a CDR where
#   cause-of-death coding is incomplete (~75% of deaths have no cause).
#   The 30-day proxy is the dominant contributor; explicitly coded CVD
#   deaths add the remainder. Stated as a limitation in the paper.

print("\n" + "=" * 60)
print("STEP 4: Outcome definition and cohort skeleton")
print("=" * 60)

cohort = prs_df[['person_id', 'cad_prs']].copy()

# ── Merge MI ──────────────────────────────────────────────────────────────
cohort = cohort.merge(mi_first, on='person_id', how='left')
cohort['has_mi'] = cohort['mi_date'].notna()

# ── Merge stroke ──────────────────────────────────────────────────────────
cohort = cohort.merge(stroke_first, on='person_id', how='left')
cohort['has_stroke'] = cohort['stroke_date'].notna()

# ── CVD death: explicitly coded ──────────────────────────────────────────
cvd_death_coded = (
    death_genetic[death_genetic['is_cvd_death_coded']]
    [['person_id', 'death_date']]
    .rename(columns={'death_date': 'cvd_death_date'})
)
cohort = cohort.merge(cvd_death_coded, on='person_id', how='left')
cohort['has_cvd_death_coded'] = cohort['cvd_death_date'].notna()

# ── All-cause death (censoring anchor) ────────────────────────────────────
all_death = (death_genetic[['person_id', 'death_date']]
             .rename(columns={'death_date': 'death_date_any'}))
cohort = cohort.merge(all_death, on='person_id', how='left')
cohort['has_any_death'] = cohort['death_date_any'].notna()

# ── Convert all date columns to datetime64 ────────────────────────────────
for col in ['mi_date', 'stroke_date', 'cvd_death_date', 'death_date_any']:
    cohort[col] = pd.to_datetime(cohort[col], errors='coerce')

# ── 30-day post-MI/stroke death proxy ────────────────────────────────────
# Used when no CVD cause is coded — death within 30 days of MI/stroke
# is almost certainly cardiovascular.
mace2_date_temp = cohort[['mi_date', 'stroke_date']].min(axis=1)
days_to_death   = (cohort['death_date_any'] - mace2_date_temp).dt.days
cohort['proxy_cvd_death'] = (
    cohort['has_any_death'] &
    (mace2_date_temp.notna()) &
    (days_to_death >= 0) &
    (days_to_death <= 30)
)

# ── PRIMARY: 3-point MACE ─────────────────────────────────────────────────
cohort['has_cvd_death'] = cohort['has_cvd_death_coded'] | cohort['proxy_cvd_death']
cohort['cvd_death_date_final'] = cohort[['cvd_death_date', 'death_date_any']].apply(
    lambda row: row['death_date_any'] if (pd.isna(row['cvd_death_date']) and cohort.loc[row.name, 'proxy_cvd_death'])
    else row['cvd_death_date'], axis=1
)

cohort['mace3_event'] = cohort['has_mi'] | cohort['has_stroke'] | cohort['has_cvd_death']
cohort['mace3_date']  = cohort[['mi_date', 'stroke_date', 'cvd_death_date_final']].min(axis=1)
cohort.loc[~cohort['mace3_event'], 'mace3_date'] = pd.NaT

# ── Print the outcome summary ─────────────────────────────────────────────
n = len(cohort)
print(f"\nGenetic cohort N = {n:,}\n")

print(f"{'Outcome':<50} {'N':>8} {'%':>7}")
print("-" * 67)
print(f"{'3-pt MACE (MI + stroke + CVD death)':<50} "
      f"{cohort['mace3_event'].sum():>8,} "
      f"{100*cohort['mace3_event'].mean():>6.1f}%")
print(f"{'  Myocardial Infarction (MI)':<50} "
      f"{cohort['has_mi'].sum():>8,} "
      f"{100*cohort['has_mi'].mean():>6.1f}%")
print(f"{'  Stroke':<50} "
      f"{cohort['has_stroke'].sum():>8,} "
      f"{100*cohort['has_stroke'].mean():>6.1f}%")
print(f"{'  CVD Death (coded + 30-day proxy)':<50} "
      f"{cohort['has_cvd_death'].sum():>8,} "
      f"{100*cohort['has_cvd_death'].mean():>6.1f}%")
print(f"{'    of which: explicitly coded CVD cause':<50} "
      f"{cohort['has_cvd_death_coded'].sum():>8,} "
      f"{100*cohort['has_cvd_death_coded'].mean():>6.1f}%")
print(f"{'    of which: 30-day post-MI/stroke proxy':<50} "
      f"{cohort['proxy_cvd_death'].sum():>8,} "
      f"{100*cohort['proxy_cvd_death'].mean():>6.1f}%")
print(f"{'  MI + Stroke overlap (counted once)':<50} "
      f"{(cohort['has_mi'] & cohort['has_stroke']).sum():>8,} "
      f"{100*(cohort['has_mi'] & cohort['has_stroke']).mean():>6.1f}%")
print()
print(f"{'All-cause death (any timing)':<50} "
      f"{cohort['has_any_death'].sum():>8,} "
      f"{100*cohort['has_any_death'].mean():>6.1f}%")
      f"{100*cohort['proxy_cvd_death'].mean():>6.1f}%")
print()
print(f"{'All-cause death (any timing)':<45} "
      f"{cohort['has_any_death'].sum():>8,} "
      f"{100*cohort['has_any_death'].mean():>6.1f}%")

print(f"\nDate completeness:")
print(f"  PRIMARY events with a date:     "
      f"{cohort.loc[cohort['mace2_event'], 'mace2_date'].notna().sum():,} / "
      f"{cohort['mace2_event'].sum():,}")
print(f"  SENSITIVITY events with a date: "
      f"{cohort.loc[cohort['mace3s_event'], 'mace3s_date'].notna().sum():,} / "
      f"{cohort['mace3s_event'].sum():,}")

print(f"\nPrimary outcome date range:")
mace_dates = cohort.loc[cohort['mace3_event'], 'mace3_date']
print(f"  Earliest: {mace_dates.min().date()}")
print(f"  Latest:   {mace_dates.max().date()}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4b — Apply incident-only filter (landmark date Jan 1 2018)
# ══════════════════════════════════════════════════════════════════════════
# Landmark = Jan 1 2018 (start of the exposome measurement window).
#
# ONLY exclusion criterion: coded MACE event before the landmark.
# These are genuinely prevalent cases — the MI or stroke predates the
# entire study window, so including them would mix prevalent disease
# with the exposome exposure in a way that's not interpretable.
#
# We do NOT require EHR data to be present before 2018.
# Reason: observation_period_start_date reflects when a health system
# began submitting to AoU, not when the person first had healthcare.
# A participant whose system joined AoU in 2021 may have no pre-2018
# EHR records, but also no coded prior MACE — treating them as incident
# is correct and conservative (if anything, we'll miss a small number
# of truly prevalent cases whose prior events aren't in AoU, biasing
# toward the null rather than inflating the effect).

LANDMARK = pd.Timestamp('2018-01-01')

print("\n" + "=" * 60)
print("STEP 4b: Incident-only filter — landmark Jan 1 2018")
print("=" * 60)

n_before     = len(cohort)
events_before = cohort['mace3_event'].sum()

# Participants whose earliest coded MACE is before the landmark are excluded
prevalent = cohort['mace3_event'] & (cohort['mace3_date'] < LANDMARK)

print(f"\nBefore filtering:")
print(f"  Total cohort:            {n_before:,}")
print(f"  Any MACE (all time):     {events_before:,}  ({100*events_before/n_before:.1f}%)")
print(f"  Prevalent cases (<2018): {prevalent.sum():,}  "
      f"({100*prevalent.sum()/events_before:.1f}% of all MACE events)")

# Year-by-year breakdown
print(f"\nMACE events by year:")
mace_by_year = (cohort[cohort['mace3_event']]
                .assign(year=cohort.loc[cohort['mace3_event'], 'mace3_date'].dt.year)
                ['year'].value_counts().sort_index())
for year, count in mace_by_year.items():
    marker = '  ← excluded (prevalent)' if year < 2018 else ''
    print(f"  {year}: {count:,}{marker}")

# ── Age at landmark ───────────────────────────────────────────────────────
# Use year_of_birth from the demographics pull (Step 2).
# Age is calculated at the landmark date (Jan 1 2018), not current age,
# because that's the eligibility point. year_of_birth is the only
# resolution available in AoU, so age = 2018 - year_of_birth.
# Participants born in 2000 turn 18 during 2018 — we keep them since
# we only have birth year, not birth date.
cohort = cohort.merge(
    demo_genetic[['person_id', 'year_of_birth']].assign(
        person_id=demo_genetic['person_id'].astype(str)
    ),
    on='person_id', how='left'
)
cohort['age_at_landmark'] = LANDMARK.year - cohort['year_of_birth']
under_18 = cohort['age_at_landmark'] < 18

print(f"\nAge at landmark (Jan 1 2018):")
print(f"  Participants <18:  {under_18.sum():,}  ({100*under_18.mean():.2f}%)")
print(f"  Min age in cohort: {cohort['age_at_landmark'].min()}")
print(f"  Median age:        {cohort['age_at_landmark'].median():.0f}")

# Apply both filters: remove prevalent cases AND under-18s
cohort_incident = cohort[~prevalent & ~under_18].copy()

n_after      = len(cohort_incident)
events_after = cohort_incident['mace3_event'].sum()

print(f"\nAfter incident + age filter:")
print(f"  Total cohort:            {n_after:,}  (lost {n_before - n_after:,})")
print(f"    of which prevalent:    {prevalent.sum():,}")
print(f"    of which <18 at 2018:  {under_18.sum():,}")
print(f"  Incident 3-pt MACE:      {events_after:,}  ({100*events_after/n_after:.1f}%)")

print("\n" + "=" * 60)
print("STEP 5: Additional data availability in genetic cohort")
print("=" * 60)

# ── 5a. Key measurements (labs + vitals) ─────────────────────────────────
# IMPORTANT — these labs are COVARIATES, not inclusion criteria.
# A participant missing BMI or cholesterol stays in the cohort; their
# time-to-event data is still valid. Missingness will be handled in the
# cleaning step (most-recent-value imputation + missingness indicator flags).
#
# This query is descriptive only — it tells us:
#   (a) what the true value distributions look like (using range-filtered median)
#   (b) what fraction of the genetic cohort has at least one usable reading
#
# Range filtering here is purely to get honest summary statistics.
# OMOP mixes units across health systems so AVG is meaningless without it.
# Hard ranges per concept (values outside = unit errors or mapping garbage):
#   BMI:               10 – 80    (kg/m²)
#   SBP:               60 – 250   (mmHg)
#   DBP:               30 – 150   (mmHg)
#   Total cholesterol: 50 – 500   (mg/dL)
#   HDL cholesterol:   10 – 150   (mg/dL)  — normal HDL is 35–65, so 10–50 is valid
#   HbA1c:              3 – 20    (%)

lab_check_query = f"""
WITH filtered AS (
    SELECT
        m.measurement_concept_id,
        c.concept_name,
        m.person_id,
        m.value_as_number,
        m.measurement_date,
        m.unit_source_value
    FROM `{CDR}.measurement` m
    JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
    WHERE m.measurement_concept_id IN (
        3038553,  -- BMI
        3004249,  -- SBP
        3012888,  -- DBP
        3027114,  -- Total cholesterol
        3007070,  -- HDL cholesterol
        3004410   -- HbA1c
    )
    AND m.value_as_number IS NOT NULL
    AND CASE m.measurement_concept_id
        WHEN 3038553 THEN m.value_as_number BETWEEN 10  AND 80
        WHEN 3004249 THEN m.value_as_number BETWEEN 60  AND 250
        WHEN 3012888 THEN m.value_as_number BETWEEN 30  AND 150
        WHEN 3027114 THEN m.value_as_number BETWEEN 50  AND 500
        WHEN 3007070 THEN m.value_as_number BETWEEN 10  AND 150
        WHEN 3004410 THEN m.value_as_number BETWEEN 3   AND 20
        ELSE TRUE
    END
)
SELECT
    measurement_concept_id,
    concept_name,
    COUNT(DISTINCT person_id)                                      AS n_with_valid_reading,
    COUNT(*)                                                       AS n_valid_records,
    ROUND(APPROX_QUANTILES(value_as_number, 100)[OFFSET(5)],  1)  AS p5,
    ROUND(APPROX_QUANTILES(value_as_number, 100)[OFFSET(50)], 1)  AS median,
    ROUND(APPROX_QUANTILES(value_as_number, 100)[OFFSET(95)], 1)  AS p95,
    APPROX_TOP_COUNT(unit_source_value, 1)[OFFSET(0)].value        AS dominant_unit
FROM filtered
GROUP BY measurement_concept_id, concept_name
ORDER BY n_with_valid_reading DESC
"""

lab_df = client.query(lab_check_query).to_dataframe()

# Show coverage relative to genetic cohort size
genetic_n = len(prs_df)
lab_df['pct_of_genetic_cohort'] = (100 * lab_df['n_with_valid_reading'] / genetic_n).round(1)

print(f"\nKey labs/vitals — range-filtered summary (N cohort = {genetic_n:,})")
print("Values outside plausible ranges excluded from stats only; people are NOT dropped.")
print()
print(lab_df[['concept_name', 'n_with_valid_reading', 'pct_of_genetic_cohort',
              'n_valid_records', 'p5', 'median', 'p95', 'dominant_unit']].to_string(index=False))
print()
print("pct_of_genetic_cohort = % of PRS participants with ≥1 valid reading.")
print("Those without a reading will receive imputed values in the cleaning step.")

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
Outcome column reference for downstream scripts:
  mace3_event           bool  — PRIMARY: 3-pt MACE (MI, stroke, or CVD death)
  mace3_date            date  — date of first MACE event
  has_mi                bool  — MI component
  mi_date               date  — date of first MI
  has_stroke            bool  — stroke component
  stroke_date           date  — date of first stroke
  has_cvd_death         bool  — CVD death component (coded OR 30-day proxy)
  has_cvd_death_coded   bool  — explicitly coded CVD cause only
  proxy_cvd_death       bool  — 30-day post-MI/stroke death proxy
  has_any_death    bool  — any death recorded
  death_date_any   date  — date of death (for censoring)
""")

# ── Save both skeletons ───────────────────────────────────────────────────
# full_cohort_skeleton: everyone with a CAD PRS (pre-filter) — keep for
#   reference and for the XGBoost model which doesn't need incident-only
# incident_cohort_skeleton: landmark-filtered, used for Cox / KM survival

save_full     = f'{WORKSPACE}/cohort_skeleton_full.csv'
save_incident = f'{WORKSPACE}/cohort_skeleton_incident.csv'

cohort.to_csv(save_full, index=False)
cohort_incident.to_csv(save_incident, index=False)

print(f"Full cohort saved to:     {save_full}  {cohort.shape}")
print(f"Incident cohort saved to: {save_incident}  {cohort_incident.shape}")
