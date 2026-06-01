"""
concept_id_lookup.py
====================
Looks up correct OMOP concept IDs by SNOMED / RxNorm concept code —
much more reliable than guessing the internal OMOP integer ID.

Run this, paste the output, and the correct concept_ids will be
updated in build_master.py.
"""

from google.cloud import bigquery
import pandas as pd

CDR    = 'wb-silky-artichoke-2408.C2024Q3R9'
client = bigquery.Client(project='wb-shining-lemon-5239')

# ── Conditions: search by SNOMED concept code ─────────────────────────────────
# SNOMED codes are the published ontology identifiers, stable across OMOP versions
SNOMED_LOOKUP = {
    'fh':               ('398036000', 'Familial hypercholesterolaemia'),
    'metabolic_syndrome': ('237602007','Metabolic syndrome X'),
    'osa':              ('78275009',  'Obstructive sleep apnea syndrome'),
    'sle':              ('55464009',  'Systemic lupus erythematosus'),
    'ulc_colitis':      ('64766004',  'Ulcerative colitis'),
    'hyperthyroidism':  ('34486009',  'Hyperthyroidism'),
    'pcos':             ('69878008',  'Polycystic ovary syndrome'),
    'cushings':         ('47270006',  "Cushing's syndrome"),
    'acromegaly':       ('74107003',  'Acromegaly'),
    'preeclampsia':     ('398254007', 'Pre-eclampsia'),
    'eclampsia':        ('15938005',  'Eclampsia'),
    'preterm':          ('282020008', 'Preterm delivery'),
}

# ── Drugs: search by RxNorm concept code (CUI) ───────────────────────────────
RXNORM_LOOKUP = {
    'doxorubicin':  ('3151',   'Doxorubicin'),
    'epirubicin':   ('41867',  'Epirubicin'),
    'daunorubicin': ('3002',   'Daunorubicin'),
    'idarubicin':   ('27340',  'Idarubicin'),
    'trastuzumab':  ('224905', 'Trastuzumab'),
}

SEP = '─' * 70

# ── 1. Condition concept lookup ───────────────────────────────────────────────
print('=' * 70)
print('CONDITION concept IDs (by SNOMED concept code, standard_concept = S)')
print('=' * 70)

snomed_codes = [code for code, _ in SNOMED_LOOKUP.values()]
q_snomed = f"""
SELECT
    concept_id,
    concept_name,
    concept_code   AS snomed_code,
    domain_id,
    standard_concept
FROM `{CDR}.concept`
WHERE vocabulary_id = 'SNOMED'
  AND concept_code IN ({','.join(f"'{c}'" for c in snomed_codes)})
  AND standard_concept = 'S'
ORDER BY concept_code
"""
snomed_df = client.query(q_snomed).to_dataframe()
print(snomed_df.to_string(index=False))

print(f'\n{SEP}')
print('Flag → SNOMED code → OMOP concept_id (use these in build_master.py)')
print(SEP)
code_to_row = {r['snomed_code']: r for _, r in snomed_df.iterrows()}
for flag, (code, expected) in SNOMED_LOOKUP.items():
    row = code_to_row.get(code)
    if row is not None:
        print(f"  {flag:<22} SNOMED {code:<12} → concept_id {int(row['concept_id']):>10,}  '{row['concept_name']}'")
    else:
        print(f"  {flag:<22} SNOMED {code:<12} → *** NOT FOUND — try name search below ***")

# ── 2. Name fallback for any not found by code ───────────────────────────────
not_found = [flag for flag, (code, _) in SNOMED_LOOKUP.items() if code not in code_to_row]
if not_found:
    print(f'\n{SEP}')
    print('Name-based fallback search for unfound conditions:')
    print(SEP)
    terms = ' OR '.join([f"LOWER(concept_name) LIKE '%{SNOMED_LOOKUP[f][1].lower()[:20]}%'"
                         for f in not_found])
    q_name = f"""
    SELECT concept_id, concept_name, concept_code, vocabulary_id, standard_concept, domain_id
    FROM `{CDR}.concept`
    WHERE standard_concept = 'S'
      AND domain_id = 'Condition'
      AND ({terms})
    ORDER BY concept_name
    """
    print(client.query(q_name).to_dataframe().to_string(index=False))

# ── 3. Drug concept lookup ────────────────────────────────────────────────────
print(f'\n{"=" * 70}')
print('DRUG concept IDs (by RxNorm concept code, Ingredient class, standard = S)')
print('=' * 70)

rxnorm_codes = [code for code, _ in RXNORM_LOOKUP.values()]
q_rx = f"""
SELECT
    concept_id,
    concept_name,
    concept_code   AS rxnorm_cui,
    concept_class_id,
    standard_concept
FROM `{CDR}.concept`
WHERE vocabulary_id = 'RxNorm'
  AND concept_code IN ({','.join(f"'{c}'" for c in rxnorm_codes)})
  AND concept_class_id = 'Ingredient'
  AND standard_concept = 'S'
ORDER BY concept_name
"""
rx_df = client.query(q_rx).to_dataframe()
print(rx_df.to_string(index=False))

print(f'\n{SEP}')
print('Flag → RxNorm CUI → OMOP concept_id (use these in build_master.py)')
print(SEP)
rx_to_row = {r['rxnorm_cui']: r for _, r in rx_df.iterrows()}
for flag, (code, expected) in RXNORM_LOOKUP.items():
    row = rx_to_row.get(code)
    if row is not None:
        print(f"  {flag:<16} RxNorm CUI {code:<8} → concept_id {int(row['concept_id']):>10,}  '{row['concept_name']}'")
    else:
        print(f"  {flag:<16} RxNorm CUI {code:<8} → *** NOT FOUND ***")

# ── 4. Name fallback for drugs not found by CUI ──────────────────────────────
rx_not_found = [flag for flag, (code, _) in RXNORM_LOOKUP.items() if code not in rx_to_row]
if rx_not_found:
    print(f'\n{SEP}')
    print('Name-based fallback for unfound drugs:')
    print(SEP)
    drug_names = [RXNORM_LOOKUP[f][1].lower() for f in rx_not_found]
    drug_terms = ' OR '.join([f"LOWER(concept_name) LIKE '%{n}%'" for n in drug_names])
    q_drug_name = f"""
    SELECT concept_id, concept_name, concept_code, vocabulary_id, concept_class_id, standard_concept
    FROM `{CDR}.concept`
    WHERE standard_concept = 'S'
      AND domain_id = 'Drug'
      AND concept_class_id = 'Ingredient'
      AND ({drug_terms})
    ORDER BY concept_name
    """
    print(client.query(q_drug_name).to_dataframe().to_string(index=False))

print('\n\nLOOKUP COMPLETE — paste output to get corrected concept_ids for build_master.py')
