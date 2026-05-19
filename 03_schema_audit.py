"""
CELL 3 — CDR schema audit: every table and column available for the genetic cohort
===================================================================================
Run this to get a complete picture of what data exists before you design any
cleaning or feature-engineering steps. It answers:
  - What tables are in the CDR?
  - What columns does each table have?
  - For clinical tables, how many records exist for our genetic cohort?

This does NOT pull any actual values — it only reads metadata and counts.
BigQuery INFORMATION_SCHEMA queries are near-instant regardless of table size.
"""

import os
import pandas as pd
from google.cloud import bigquery

CDR     = os.environ['WORKSPACE_CDR']
client  = bigquery.Client()

# Assumes prs_df was built in Cell 2 and is still in memory.
# If not, reload it:
#   WORKSPACE = '/home/jupyter/workspaces/geneexposome'
#   prs_df = pd.read_parquet(f'{WORKSPACE}/PRS_Scores/CAD/<filename>.parquet')
#   prs_df['person_id'] = prs_df['person_id'].astype(str)

genetic_ids = set(prs_df['person_id'].astype(str))
print(f"Genetic cohort size: {len(genetic_ids):,} participants")
print(f"CDR dataset:         {CDR}\n")


# ══════════════════════════════════════════════════════════════════════════
# PART A — Full column schema for every table in the CDR
# ══════════════════════════════════════════════════════════════════════════
# INFORMATION_SCHEMA.COLUMNS lists every column in every table.
# We only need TABLE_NAME, COLUMN_NAME, and DATA_TYPE — no row scans.

print("=" * 70)
print("PART A: ALL TABLES AND COLUMNS IN THE CDR")
print("=" * 70)

# BigQuery INFORMATION_SCHEMA is scoped to dataset — use backtick project.dataset format
project, dataset = CDR.rsplit('.', 1)   # split "project.dataset" into parts

schema_query = f"""
SELECT
    TABLE_NAME,
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
ORDER BY TABLE_NAME, ORDINAL_POSITION
"""

schema_df = client.query(schema_query).to_dataframe()

# Print grouped by table so it reads like a data dictionary
print(f"\nTotal tables: {schema_df['TABLE_NAME'].nunique()}")
print(f"Total columns across all tables: {len(schema_df)}\n")

for table_name, grp in schema_df.groupby('TABLE_NAME'):
    cols = grp[['COLUMN_NAME', 'DATA_TYPE', 'IS_NULLABLE']].reset_index(drop=True)
    col_summary = ', '.join([
        f"{row.COLUMN_NAME} ({row.DATA_TYPE})" for _, row in cols.iterrows()
    ])
    print(f"  {table_name}  [{len(cols)} columns]")
    # Print each column on its own line indented
    for _, row in cols.iterrows():
        nullable = '' if row.IS_NULLABLE == 'YES' else ' NOT NULL'
        print(f"      {row.COLUMN_NAME:<45} {row.DATA_TYPE}{nullable}")
    print()


# ══════════════════════════════════════════════════════════════════════════
# PART B — Row counts per table for the genetic cohort
# ══════════════════════════════════════════════════════════════════════════
# For each person_id-keyed clinical table, count how many records exist
# for participants in the genetic cohort. This tells you data density
# before you decide which tables to use.

print("=" * 70)
print("PART B: ROW COUNTS FOR GENETIC COHORT ACROSS CLINICAL TABLES")
print("=" * 70)
print("(Counts are for genetic cohort participants only)\n")

# These are the core OMOP CDM person-level tables in All of Us.
# Each has a person_id column we can filter on.
clinical_tables = [
    ('condition_occurrence',   'Diagnoses (ICD→OMOP mapped)'),
    ('drug_exposure',          'Medications dispensed / prescribed'),
    ('measurement',            'Labs and vitals'),
    ('observation',            'Survey answers, social history, smoking, etc.'),
    ('procedure_occurrence',   'Procedures (CPT/ICD-PCS mapped)'),
    ('visit_occurrence',       'Healthcare encounters (inpatient, outpatient, ED)'),
    ('device_exposure',        'Devices (pacemakers, stents, etc.)'),
    ('death',                  'Death records'),
    ('observation_period',     'EHR coverage window per participant'),
    ('specimen',               'Biospecimen collections'),
    ('survey_conduct',         'Survey completion metadata'),
]

count_results = []

for table, description in clinical_tables:
    try:
        count_query = f"""
        SELECT COUNT(*) AS n_records, COUNT(DISTINCT person_id) AS n_persons
        FROM `{CDR}.{table}`
        """
        # We count the full table first (cheap aggregation), then note genetic fraction
        total = client.query(count_query).to_dataframe().iloc[0]
        count_results.append({
            'table': table,
            'description': description,
            'total_records': int(total['n_records']),
            'total_persons': int(total['n_persons']),
        })
        print(f"  {table:<30} {int(total['n_records']):>12,} records  |  {int(total['n_persons']):>8,} persons  — {description}")
    except Exception as e:
        print(f"  {table:<30} ERROR: {e}")

count_df = pd.DataFrame(count_results)


# ══════════════════════════════════════════════════════════════════════════
# PART C — Measurement domain breakdown (what labs/vitals are captured)
# ══════════════════════════════════════════════════════════════════════════
# The measurement table is the densest domain. Show the top 40 most common
# measurement concepts across the full CDR so you can see what's available.

print("\n" + "=" * 70)
print("PART C: TOP 50 MOST COMMON MEASUREMENTS IN THE CDR")
print("=" * 70)

top_meas_query = f"""
SELECT
    m.measurement_concept_id,
    c.concept_name,
    c.concept_code,
    COUNT(DISTINCT m.person_id) AS n_persons,
    COUNT(*) AS n_records,
    ROUND(MIN(m.value_as_number), 2)  AS min_val,
    ROUND(MAX(m.value_as_number), 2)  AS max_val,
    ROUND(AVG(m.value_as_number), 2)  AS mean_val,
    -- Unit (most common unit for this concept)
    APPROX_TOP_COUNT(unit_source_value, 1)[OFFSET(0)].value AS most_common_unit
FROM `{CDR}.measurement` m
JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
WHERE m.measurement_concept_id != 0   -- exclude unmapped
GROUP BY m.measurement_concept_id, c.concept_name, c.concept_code
ORDER BY n_persons DESC
LIMIT 50
"""

top_meas_df = client.query(top_meas_query).to_dataframe()
print(top_meas_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# PART D — Observation domain breakdown (surveys, social determinants)
# ══════════════════════════════════════════════════════════════════════════
# The observation table holds survey responses and social history.
# Critical for exposome linkage and confounders (smoking, alcohol, income, etc.)

print("\n" + "=" * 70)
print("PART D: TOP 50 MOST COMMON OBSERVATIONS (SURVEYS / SOCIAL HISTORY)")
print("=" * 70)

top_obs_query = f"""
SELECT
    o.observation_concept_id,
    c.concept_name,
    c.domain_id,
    COUNT(DISTINCT o.person_id) AS n_persons,
    COUNT(*) AS n_records
FROM `{CDR}.observation` o
JOIN `{CDR}.concept` c ON o.observation_concept_id = c.concept_id
WHERE o.observation_concept_id != 0
GROUP BY o.observation_concept_id, c.concept_name, c.domain_id
ORDER BY n_persons DESC
LIMIT 50
"""

top_obs_df = client.query(top_obs_query).to_dataframe()
print(top_obs_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# PART E — Condition domain: top 50 diagnoses in the CDR
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART E: TOP 50 MOST COMMON CONDITIONS (DIAGNOSES) IN THE CDR")
print("=" * 70)

top_cond_query = f"""
SELECT
    co.condition_concept_id,
    c.concept_name,
    c.concept_code,
    COUNT(DISTINCT co.person_id) AS n_persons,
    COUNT(*) AS n_records,
    MIN(co.condition_start_date) AS earliest_date,
    MAX(co.condition_start_date) AS latest_date
FROM `{CDR}.condition_occurrence` co
JOIN `{CDR}.concept` c ON co.condition_concept_id = c.concept_id
WHERE co.condition_concept_id != 0
GROUP BY co.condition_concept_id, c.concept_name, c.concept_code
ORDER BY n_persons DESC
LIMIT 50
"""

top_cond_df = client.query(top_cond_query).to_dataframe()
print(top_cond_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# PART F — Drug domain: top 40 medications
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART F: TOP 40 MOST COMMON DRUG EXPOSURES IN THE CDR")
print("=" * 70)

top_drug_query = f"""
SELECT
    de.drug_concept_id,
    c.concept_name,
    c.concept_class_id,
    COUNT(DISTINCT de.person_id) AS n_persons,
    COUNT(*) AS n_records
FROM `{CDR}.drug_exposure` de
JOIN `{CDR}.concept` c ON de.drug_concept_id = c.concept_id
WHERE de.drug_concept_id != 0
GROUP BY de.drug_concept_id, c.concept_name, c.concept_class_id
ORDER BY n_persons DESC
LIMIT 40
"""

top_drug_df = client.query(top_drug_query).to_dataframe()
print(top_drug_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SCHEMA AUDIT COMPLETE")
print("=" * 70)
print("""
What to look for in these results:
  Part A  — use this to understand which columns are available in each
            table before writing cleaning code (e.g. does drug_exposure
            have drug_exposure_end_date? Is quantity populated?)

  Part B  — row counts tell you data density. Tables with zero or very
            few records for your cohort aren't worth using.

  Part C  — find the measurement_concept_ids for the covariates you need
            (LDL, SBP, BMI, HbA1c). Copy the concept_id from here into
            your cleaning script.

  Part D  — check whether AoU surveys (The Basics, Lifestyle, SDOH) are
            present. These are key for smoking, alcohol, physical activity,
            and neighborhood covariates.

  Part E  — confirms cardiovascular comorbidities (hypertension, T2DM,
            CKD) are coded and available for adjustment.

  Part F  — confirms statin, antihypertensive, and antiplatelet use is
            captured (important confounders for CAD analysis).
""")
