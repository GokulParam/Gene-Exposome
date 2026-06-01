"""
CELL 5b — ZIP3 diagnostic
==========================
The exposome merge produced 100% NaN because zip3 coverage = 0%.
This script probes the CDR to find where zip codes actually live.
Run this, paste the output here, and we'll fix the merge query.
"""

import pandas as pd
from google.cloud import bigquery

CDR    = 'wb-silky-artichoke-2408.C2024Q3R9'
client = bigquery.Client(project='wb-shining-lemon-5239')

def run(label, sql):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    try:
        df = client.query(sql).to_dataframe()
        print(df.to_string(index=False))
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ── 1. Does cb_search_person exist and what columns does it have? ──────────
run("cb_search_person columns", f"""
SELECT column_name, data_type
FROM `{CDR}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'cb_search_person'
ORDER BY ordinal_position
""")


# ── 2. Does person_ext exist and what columns? ─────────────────────────────
run("person_ext columns", f"""
SELECT column_name, data_type
FROM `{CDR}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'person_ext'
ORDER BY ordinal_position
""")


# ── 3. Sample cb_search_person — look for anything zip-like ───────────────
run("cb_search_person sample (10 rows)", f"""
SELECT *
FROM `{CDR}.cb_search_person`
LIMIT 10
""")


# ── 4. Observation concepts that contain 'zip' in the name ────────────────
run("Observation concepts with 'zip' in name", f"""
SELECT DISTINCT o.observation_concept_id,
       c.concept_name,
       COUNT(DISTINCT o.person_id) AS n_persons
FROM `{CDR}.observation` o
JOIN `{CDR}.concept` c ON o.observation_concept_id = c.concept_id
WHERE LOWER(c.concept_name) LIKE '%zip%'
   OR LOWER(c.concept_name) LIKE '%postal%'
GROUP BY 1, 2
ORDER BY n_persons DESC
LIMIT 20
""")


# ── 5. Observation value_as_string samples for concept 1585250 ────────────
run("Observation concept 1585250 samples (zip survey)", f"""
SELECT o.value_as_string,
       COUNT(*) AS n
FROM `{CDR}.observation` o
WHERE o.observation_concept_id = 1585250
  AND o.value_as_string IS NOT NULL
GROUP BY 1
ORDER BY n DESC
LIMIT 20
""")


# ── 6. Any table in CDR with a column named like 'zip' ────────────────────
run("All CDR tables with a zip-like column", f"""
SELECT table_name, column_name, data_type
FROM `{CDR}.INFORMATION_SCHEMA.COLUMNS`
WHERE LOWER(column_name) LIKE '%zip%'
   OR LOWER(column_name) LIKE '%postal%'
ORDER BY table_name, ordinal_position
""")


print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE — paste this output to fix the zip3 merge")
print("=" * 60)
