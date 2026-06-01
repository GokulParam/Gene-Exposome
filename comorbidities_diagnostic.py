"""
comorbidities_diagnostic.py
============================
Verifies OMOP concept IDs for all planned comorbidity flags and prints
pre-landmark prevalence counts.  Run this before build_master.py to
confirm concept IDs are correct for this CDR version.
"""

from google.cloud import bigquery
import pandas as pd

CDR      = 'wb-silky-artichoke-2408.C2024Q3R9'
LANDMARK = '2018-01-01'
client   = bigquery.Client(project='wb-shining-lemon-5239')

# ── Proposed concept IDs ──────────────────────────────────────────────────────
# Format: flag_name → [(concept_id, expected_name), ...]
COND_IDS = {
    # Very High Risk
    't1dm':              [(201254,  'Type 1 diabetes mellitus')],
    'fh':                [(314522,  'Familial hypercholesterolaemia')],
    'pad':               [(321052,  'Peripheral vascular disease')],

    # High Risk
    'metabolic_syndrome':[(4028741, 'Metabolic syndrome')],
    'osa':               [(4173505, 'Obstructive sleep apnea syndrome')],
    'heart_failure':     [(316139,  'Heart failure')],
    'afib':              [(313217,  'Atrial fibrillation')],

    # Inflammatory / Autoimmune
    'ra':                [(80809,   'Rheumatoid arthritis')],
    'sle':               [(201606,  'Systemic lupus erythematosus')],
    'psoriasis':         [(140168,  'Psoriasis')],
    'crohns':            [(4052776, "Crohn's disease")],
    'ulc_colitis':       [(4059478, 'Ulcerative colitis')],
    'hiv':               [(439727,  'Human immunodeficiency virus infection')],

    # Endocrine / Hormonal
    'hypothyroidism':    [(140673,  'Hypothyroidism')],
    'hyperthyroidism':   [(4058243, 'Hyperthyroidism')],
    'pcos':              [(4070454, 'Polycystic ovary syndrome')],
    'cushings':          [(197961,  "Cushing's syndrome")],
    'acromegaly':        [(4023722, 'Acromegaly')],

    # Pregnancy-related
    'preeclampsia':      [(4060985, 'Pre-eclampsia'), (4024244, 'Eclampsia')],
    'gest_dm':           [(4024659, 'Gestational diabetes mellitus')],
    'preterm':           [(4163838, 'Preterm delivery')],
    'preg_loss':         [(4067106, 'Spontaneous abortion / pregnancy loss')],
}

CHEMO_IDS = {
    'cardiotoxic_chemo': [
        (1350066, 'Doxorubicin'),
        (1396797, 'Epirubicin'),
        (1313411, 'Daunorubicin'),
        (1336941, 'Idarubicin'),
        (1336825, 'Trastuzumab'),
    ],
}

SEP = '─' * 70

# ── 1. Verify concept names ───────────────────────────────────────────────────
all_cond_ids = [cid for lst in COND_IDS.values() for cid, _ in lst]
all_drug_ids = [cid for lst in CHEMO_IDS.values() for cid, _ in lst]

print('=' * 70)
print('1. Concept ID → name verification')
print('=' * 70)

q_cond = f"""
SELECT concept_id, concept_name, domain_id, vocabulary_id, standard_concept
FROM `{CDR}.concept`
WHERE concept_id IN ({','.join(str(i) for i in all_cond_ids)})
ORDER BY concept_id
"""
cond_df = client.query(q_cond).to_dataframe()
print('\nCondition concepts:')
print(cond_df.to_string(index=False))

q_drug = f"""
SELECT concept_id, concept_name, domain_id, vocabulary_id, standard_concept
FROM `{CDR}.concept`
WHERE concept_id IN ({','.join(str(i) for i in all_drug_ids)})
ORDER BY concept_id
"""
drug_df = client.query(q_drug).to_dataframe()
print('\nDrug concepts (chemo):')
print(drug_df.to_string(index=False))

# ── 2. Cross-check expected vs actual names ───────────────────────────────────
print('\n\n' + '=' * 70)
print('2. Flag-level ID check (MISMATCH = wrong concept ID)')
print('=' * 70)

id_to_name = dict(zip(cond_df['concept_id'], cond_df['concept_name']))
id_to_name.update(dict(zip(drug_df['concept_id'], drug_df['concept_name'])))

for flag, entries in {**COND_IDS, **CHEMO_IDS}.items():
    for cid, expected in entries:
        actual = id_to_name.get(cid, '*** NOT FOUND ***')
        match = '✓' if expected.lower() in actual.lower() or actual.lower() in expected.lower() else '✗ MISMATCH'
        print(f"  {flag:<22} {cid:>10,}  {match}  actual='{actual}'")

# ── 3. Pre-landmark prevalence ────────────────────────────────────────────────
print('\n\n' + '=' * 70)
print(f'3. Pre-landmark condition counts (before {LANDMARK})')
print(f"  {'Flag':<22} {'N persons':>10}  Concept IDs used")
print(SEP)

def count_cond(concept_ids):
    ids = ','.join(str(i) for i in concept_ids)
    q = f"""
    SELECT COUNT(DISTINCT co.person_id) AS n
    FROM `{CDR}.condition_occurrence` co
    JOIN `{CDR}.concept_ancestor` ca
        ON co.condition_concept_id = ca.descendant_concept_id
    WHERE ca.ancestor_concept_id IN ({ids})
      AND co.condition_start_date < '{LANDMARK}'
    """
    return int(client.query(q).to_dataframe().iloc[0, 0])

def count_drug(concept_ids):
    ids = ','.join(str(i) for i in concept_ids)
    q = f"""
    SELECT COUNT(DISTINCT de.person_id) AS n
    FROM `{CDR}.drug_exposure` de
    JOIN `{CDR}.concept_ancestor` ca
        ON de.drug_concept_id = ca.descendant_concept_id
    WHERE ca.ancestor_concept_id IN ({ids})
      AND de.drug_exposure_start_date < '{LANDMARK}'
    """
    return int(client.query(q).to_dataframe().iloc[0, 0])

for flag, entries in COND_IDS.items():
    ids = [cid for cid, _ in entries]
    try:
        n = count_cond(ids)
        print(f"  {flag:<22} {n:>10,}  {ids}")
    except Exception as e:
        print(f"  {flag:<22}  ERROR: {e}")

print(SEP)
for flag, entries in CHEMO_IDS.items():
    ids = [cid for cid, _ in entries]
    try:
        n = count_drug(ids)
        print(f"  {flag:<22} {n:>10,}  {ids}  [drug_exposure]")
    except Exception as e:
        print(f"  {flag:<22}  ERROR: {e}")

print('\n\nDIAGNOSTIC COMPLETE')
print('Review any ✗ MISMATCH lines above and correct concept IDs in build_master.py')
