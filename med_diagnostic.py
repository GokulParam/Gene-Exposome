"""
CELL — Medication breakdown diagnostic
=======================================
Shows exactly which drugs are captured by the current concept ID lists,
and proposes separate flags for individual drug classes.
"""

from google.cloud import bigquery
import pandas as pd

CDR    = 'wb-silky-artichoke-2408.C2024Q3R9'
client = bigquery.Client(project='wb-shining-lemon-5239')
LANDMARK = '2018-01-01'

SEP = "─" * 70

# ── 1. What do the current concept IDs actually map to? ────────────────────
print("=" * 70)
print("1. Current concept ID → drug name mapping")
print("=" * 70)

all_ids = [1539403,1592085,1510813,1549686,1551860,1581987,   # statins
           1308216,1335471,1346823,1353766,1395058,1340128,   # antihtn
           1503297,                                            # metformin
           1115008,1154343,1301025,1322184]                   # antiplatelet

q = f"""
SELECT concept_id, concept_name, concept_class_id, vocabulary_id
FROM `{CDR}.concept`
WHERE concept_id IN ({','.join(str(i) for i in all_ids)})
ORDER BY concept_class_id, concept_name
"""
print(client.query(q).to_dataframe().to_string(index=False))

# ── 2. Antiplatelet breakdown: counts per ingredient ───────────────────────
print(f"\n\n{'='*70}")
print("2. Antiplatelet / anticoagulant drug counts (pre-landmark)")
print("   Using concept_ancestor to capture all formulations per ingredient")
print("=" * 70)

# Key cardiovascular antiplatelet/anticoagulant ingredients
antiplatelets = {
    1112807: 'Aspirin',
    1322184: 'Warfarin',
    1154343: 'Clopidogrel',
    40163924:'Ticagrelor',
    1592085: 'Prasugrel',   # note: also in statin list? check
    1115008: 'Dipyridamole',
    43013024:'Rivaroxaban',
    1310149: 'Apixaban',
    1351461: 'Dabigatran',
}

rows = []
for cid, name in antiplatelets.items():
    q2 = f"""
    SELECT COUNT(DISTINCT de.person_id) AS n_persons
    FROM `{CDR}.drug_exposure` de
    JOIN `{CDR}.concept_ancestor` ca
        ON de.drug_concept_id = ca.descendant_concept_id
    WHERE ca.ancestor_concept_id = {cid}
      AND de.drug_exposure_start_date < '{LANDMARK}'
    """
    try:
        n = client.query(q2).to_dataframe().iloc[0,0]
        rows.append({'drug': name, 'concept_id': cid, 'n_persons_pre_landmark': n})
    except Exception as e:
        rows.append({'drug': name, 'concept_id': cid, 'n_persons_pre_landmark': f'ERROR: {e}'})

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# ── 3. What's inside the current antiplatelet concept IDs specifically ──────
print(f"\n\n{'='*70}")
print("3. Exact drugs captured by current antiplatelet IDs (1115008, 1154343, 1301025, 1322184)")
print("=" * 70)

q3 = f"""
SELECT
    ca.ancestor_concept_id,
    c_anc.concept_name AS ancestor_name,
    c_drug.concept_name AS drug_name,
    COUNT(DISTINCT de.person_id) AS n_persons
FROM `{CDR}.drug_exposure` de
JOIN `{CDR}.concept` c_drug ON de.drug_concept_id = c_drug.concept_id
JOIN `{CDR}.concept_ancestor` ca ON de.drug_concept_id = ca.descendant_concept_id
JOIN `{CDR}.concept` c_anc ON ca.ancestor_concept_id = c_anc.concept_id
WHERE ca.ancestor_concept_id IN (1115008, 1154343, 1301025, 1322184)
  AND de.drug_exposure_start_date < '{LANDMARK}'
GROUP BY 1, 2, 3
ORDER BY ca.ancestor_concept_id, n_persons DESC
LIMIT 40
"""
print(client.query(q3).to_dataframe().to_string(index=False))

print("\n\nDIAGNOSTIC COMPLETE")
