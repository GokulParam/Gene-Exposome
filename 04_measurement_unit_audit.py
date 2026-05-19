"""
CELL 4 — Measurement unit audit
================================
Diagnoses exactly what unit strings and unit concept IDs are present for
each key lab/vital, and what the value distribution looks like within each
unit group. Run this before writing any cleaning logic so you know:
  - Which unit_source_value strings to keep vs. drop
  - Whether any unit group needs a conversion factor applied
  - What's causing anomalous means (e.g. the BMI = 1522 issue)
"""

import os
import pandas as pd
from google.cloud import bigquery

CDR    = os.environ['WORKSPACE_CDR']
client = bigquery.Client()

# Concepts to audit — add or remove as needed
CONCEPTS = {
    3038553: 'BMI',
    3004249: 'SBP',
    3012888: 'DBP',
    3027114: 'Total cholesterol',
    3007070: 'HDL cholesterol',
    3004410: 'HbA1c',
}


# ══════════════════════════════════════════════════════════════════════════
# PART A — Unit breakdown per concept
# ══════════════════════════════════════════════════════════════════════════
# For each concept, show every distinct (unit_source_value, unit_concept_id)
# combination with record count and value distribution.
# This is the ground truth of what the CDR actually contains.

print("=" * 80)
print("PART A: Unit breakdown per measurement concept")
print("=" * 80)
print("For each concept: every unit string present, how many records use it,")
print("and the p5/median/p95 of values within that unit group.\n")

unit_query = f"""
SELECT
    m.measurement_concept_id,
    -- The raw unit string as recorded by the contributing health system
    COALESCE(m.unit_source_value, '[NULL]')              AS unit_source_value,
    -- The OMOP-standardised unit concept (often 0 = unmapped)
    m.unit_concept_id,
    COALESCE(uc.concept_name, 'unmapped')                AS unit_concept_name,
    COUNT(*)                                             AS n_records,
    COUNT(DISTINCT m.person_id)                          AS n_persons,
    -- Count how many records have a numeric value vs. not
    COUNTIF(m.value_as_number IS NOT NULL)               AS n_with_value,
    COUNTIF(m.value_as_number IS NULL)                   AS n_null_value,
    -- Distribution of numeric values within this unit group
    ROUND(MIN(m.value_as_number), 2)                     AS min_val,
    ROUND(APPROX_QUANTILES(m.value_as_number, 100)[OFFSET(5)],  2) AS p5,
    ROUND(APPROX_QUANTILES(m.value_as_number, 100)[OFFSET(50)], 2) AS median,
    ROUND(APPROX_QUANTILES(m.value_as_number, 100)[OFFSET(95)], 2) AS p95,
    ROUND(MAX(m.value_as_number), 2)                     AS max_val
FROM `{CDR}.measurement` m
LEFT JOIN `{CDR}.concept` uc ON m.unit_concept_id = uc.concept_id
WHERE m.measurement_concept_id IN ({', '.join(str(k) for k in CONCEPTS)})
GROUP BY
    m.measurement_concept_id,
    m.unit_source_value,
    m.unit_concept_id,
    uc.concept_name
ORDER BY
    m.measurement_concept_id,
    n_records DESC
"""

unit_df = client.query(unit_query).to_dataframe()

# Print grouped by concept so it's readable
for concept_id, label in CONCEPTS.items():
    grp = unit_df[unit_df['measurement_concept_id'] == concept_id].copy()
    total_records = grp['n_records'].sum()

    print(f"\n{'─' * 80}")
    print(f"  {label}  (concept_id={concept_id})  —  {total_records:,} total records")
    print(f"{'─' * 80}")

    if grp.empty:
        print("  No records found.")
        continue

    grp['pct'] = (100 * grp['n_records'] / total_records).round(1)
    display_cols = [
        'unit_source_value', 'unit_concept_name',
        'n_records', 'pct',
        'n_null_value', 'min_val', 'p5', 'median', 'p95', 'max_val'
    ]
    print(grp[display_cols].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# PART B — BMI deep dive: what do the extreme values look like?
# ══════════════════════════════════════════════════════════════════════════
# Pull a sample of BMI records with values > 80 (impossible as kg/m²) to
# understand what data is actually stored there — is it weight in lbs,
# height in cm, or genuine entry errors?

print("\n" + "=" * 80)
print("PART B: BMI anomaly deep-dive — sample of records with value > 80")
print("=" * 80)
print("Values >80 cannot be valid kg/m². Examining what's stored there.\n")

bmi_outlier_query = f"""
SELECT
    m.person_id,
    m.measurement_date,
    m.value_as_number,
    COALESCE(m.unit_source_value, '[NULL]')  AS unit_source_value,
    m.unit_concept_id,
    COALESCE(uc.concept_name, 'unmapped')    AS unit_concept_name,
    m.value_source_value,   -- the raw value string before OMOP mapping
    m.measurement_source_value  -- original concept string from source system
FROM `{CDR}.measurement` m
LEFT JOIN `{CDR}.concept` uc ON m.unit_concept_id = uc.concept_id
WHERE m.measurement_concept_id = 3038553   -- BMI
  AND m.value_as_number > 80
ORDER BY m.value_as_number DESC
LIMIT 50
"""

bmi_outlier_df = client.query(bmi_outlier_query).to_dataframe()
print(f"Sample of {len(bmi_outlier_df)} records with BMI > 80:\n")
print(bmi_outlier_df.to_string(index=False))

# Also show the full value distribution in 10-unit buckets to see the shape
print("\n" + "─" * 60)
print("BMI value distribution (bucket counts):")
print("─" * 60)

bmi_dist_query = f"""
SELECT
    FLOOR(value_as_number / 10) * 10   AS bucket_start,
    COUNT(*)                           AS n_records,
    ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM `{CDR}.measurement`
WHERE measurement_concept_id = 3038553
  AND value_as_number IS NOT NULL
  AND value_as_number > 0
GROUP BY bucket_start
ORDER BY bucket_start
"""

bmi_dist_df = client.query(bmi_dist_query).to_dataframe()
print(bmi_dist_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# PART C — Cholesterol unit audit (mg/dL vs mmol/L)
# ══════════════════════════════════════════════════════════════════════════
# Cholesterol >500 in a dataset almost always means mmol/L values stored
# without conversion. 1 mmol/L = 38.67 mg/dL, so 5 mmol/L ≈ 193 mg/dL.
# Show the count of records in each range to quantify mixing.

print("\n" + "=" * 80)
print("PART C: Cholesterol value range audit (mg/dL vs mmol/L signal)")
print("=" * 80)

chol_range_query = f"""
SELECT
    measurement_concept_id,
    CASE
        WHEN value_as_number IS NULL       THEN 'null'
        WHEN value_as_number <= 0          THEN 'zero_or_negative'
        WHEN value_as_number < 15          THEN '0–15    (mmol/L range)'
        WHEN value_as_number < 50          THEN '15–50   (ambiguous / low mg/dL)'
        WHEN value_as_number <= 500        THEN '50–500  (plausible mg/dL)'
        ELSE                                    '>500    (impossible mg/dL — likely unit error)'
    END AS value_range,
    COUNT(*) AS n_records,
    ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY measurement_concept_id), 1) AS pct
FROM `{CDR}.measurement`
WHERE measurement_concept_id IN (3027114, 3007070)  -- total chol, HDL
GROUP BY measurement_concept_id, value_range
ORDER BY measurement_concept_id, n_records DESC
"""

chol_range_df = client.query(chol_range_query).to_dataframe()

for cid, label in {3027114: 'Total cholesterol', 3007070: 'HDL cholesterol'}.items():
    grp = chol_range_df[chol_range_df['measurement_concept_id'] == cid]
    print(f"\n{label}:")
    print(grp[['value_range', 'n_records', 'pct']].to_string(index=False))

print("""
Interpretation guide:
  '0–15 (mmol/L range)'   → these are almost certainly mmol/L values stored
                             as-is; multiply by 38.67 to convert to mg/dL
  '>500 (impossible mg/dL)' → unit entry errors; safest to drop
  '50–500 (plausible mg/dL)' → keep as-is if dominant unit is mg/dL
""")
