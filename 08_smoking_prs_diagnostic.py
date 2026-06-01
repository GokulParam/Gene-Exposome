"""
CELL 8 — Diagnose smoking variable and T2D PRS
===============================================
1. Find correct smoking concepts in this CDR
2. Inspect T2D PRS distribution vs other PRS scores
"""

import os, pandas as pd, numpy as np
from google.cloud import bigquery

WORKSPACE = '/home/dataproc/workspaces/geneexposome'
CDR       = 'wb-silky-artichoke-2408.C2024Q3R9'
client    = bigquery.Client(project='wb-shining-lemon-5239')

SEP = "─" * 70


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — SMOKING DIAGNOSIS
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: SMOKING VARIABLE DIAGNOSIS")
print("=" * 70)

# 1a. What observation concepts relate to smoking?
print(f"\n{SEP}")
print("1a. All observation concepts with 'smok' in name (top 20 by person count)")
print(SEP)

q = f"""
SELECT
    o.observation_concept_id,
    c.concept_name,
    COUNT(DISTINCT o.person_id) AS n_persons,
    COUNT(*) AS n_records
FROM `{CDR}.observation` o
JOIN `{CDR}.concept` c ON o.observation_concept_id = c.concept_id
WHERE LOWER(c.concept_name) LIKE '%smok%'
GROUP BY 1, 2
ORDER BY n_persons DESC
LIMIT 20
"""
df = client.query(q).to_dataframe()
print(df.to_string(index=False))

# 1b. For the top smoking concept, what values are recorded?
if len(df) > 0:
    top_concept_id = df.iloc[0]['observation_concept_id']
    top_concept_name = df.iloc[0]['concept_name']
    print(f"\n{SEP}")
    print(f"1b. Values recorded for top concept: {top_concept_id} ({top_concept_name})")
    print(SEP)

    q2 = f"""
    SELECT
        o.value_as_concept_id,
        c_val.concept_name AS value_concept_name,
        o.value_as_string,
        COUNT(DISTINCT o.person_id) AS n_persons
    FROM `{CDR}.observation` o
    LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
    WHERE o.observation_concept_id = {top_concept_id}
    GROUP BY 1, 2, 3
    ORDER BY n_persons DESC
    LIMIT 20
    """
    df2 = client.query(q2).to_dataframe()
    print(df2.to_string(index=False))

# 1c. Check concept 1585870 (AoU PPI tobacco/smoking survey question)
print(f"\n{SEP}")
print("1c. Values for AoU PPI smoking concepts 1585870, 1585873, 903096")
print(SEP)

q3 = f"""
SELECT
    o.observation_concept_id,
    c_obs.concept_name AS question,
    o.value_as_concept_id,
    c_val.concept_name AS answer,
    COUNT(DISTINCT o.person_id) AS n_persons
FROM `{CDR}.observation` o
JOIN `{CDR}.concept` c_obs ON o.observation_concept_id = c_obs.concept_id
LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
WHERE o.observation_concept_id IN (
    1585870,  -- Tobacco Use: Do you currently smoke?
    1585873,  -- Tobacco: cigarettes per day
    903096,   -- Tobacco smoker
    1585876,  -- Tobacco: smokeless
    4041306   -- Tobacco smoking behavior (SNOMED parent)
)
GROUP BY 1, 2, 3, 4
ORDER BY o.observation_concept_id, n_persons DESC
LIMIT 40
"""
df3 = client.query(q3).to_dataframe()
print(df3.to_string(index=False))

# 1d. Check what concept the existing current_smoker flag used
#     (look at Cell 4 output — smoking used concept_ancestor 4041306)
print(f"\n{SEP}")
print("1d. concept_ancestor of 4041306 — what did Cell 4 actually match?")
print(SEP)

q4 = f"""
SELECT
    o.observation_concept_id,
    c.concept_name,
    COUNT(DISTINCT o.person_id) AS n_persons
FROM `{CDR}.observation` o
JOIN `{CDR}.concept` c ON o.observation_concept_id = c.concept_id
JOIN `{CDR}.concept_ancestor` ca ON o.observation_concept_id = ca.descendant_concept_id
WHERE ca.ancestor_concept_id = 4041306
GROUP BY 1, 2
ORDER BY n_persons DESC
LIMIT 20
"""
df4 = client.query(q4).to_dataframe()
print(df4.to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — T2D PRS DISTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("PART 2: T2D PRS DISTRIBUTION")
print("=" * 70)

prs_dir = f'{WORKSPACE}/PRS_Scores/T2D'
files   = [f for f in os.listdir(prs_dir) if not f.startswith('.')]
print(f"Files in T2D PRS dir: {files}")

prs_file = os.path.join(prs_dir, files[0])
ext = prs_file.rsplit('.', 1)[-1].lower()
t2d = pd.read_csv(prs_file) if ext == 'csv' else pd.read_csv(prs_file, sep='\t')

print(f"\nShape: {t2d.shape}")
print(f"Columns: {list(t2d.columns)}")
score_col = t2d.columns[1]
s = t2d[score_col]
print(f"\nT2D PRS raw distribution:")
print(f"  min    {s.min():.4f}")
print(f"  p1     {s.quantile(.01):.4f}")
print(f"  p5     {s.quantile(.05):.4f}")
print(f"  p25    {s.quantile(.25):.4f}")
print(f"  median {s.median():.4f}")
print(f"  p75    {s.quantile(.75):.4f}")
print(f"  p95    {s.quantile(.95):.4f}")
print(f"  p99    {s.quantile(.99):.4f}")
print(f"  max    {s.max():.4f}")
print(f"  mean   {s.mean():.4f}")
print(f"  sd     {s.std():.4f}")

# Compare all PRS score distributions side by side
print(f"\n{SEP}")
print("All PRS score distributions (raw values)")
print(SEP)
print(f"{'Trait':<12} {'min':>12} {'median':>12} {'max':>12} {'sd':>12}")
print(SEP)
for trait in ['CAD', 'LDLC', 'OBESITY', 'SBP', 'T2D']:
    d = f'{WORKSPACE}/PRS_Scores/{trait}'
    if not os.path.isdir(d):
        continue
    ff = [f for f in os.listdir(d) if not f.startswith('.')]
    if not ff:
        continue
    ext = ff[0].rsplit('.', 1)[-1].lower()
    tmp = pd.read_csv(os.path.join(d, ff[0])) if ext == 'csv' else pd.read_csv(os.path.join(d, ff[0]), sep='\t')
    sc  = tmp.iloc[:, 1]
    print(f"  {trait:<10} {sc.min():>12.3f} {sc.median():>12.3f} {sc.max():>12.3f} {sc.std():>12.3f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
