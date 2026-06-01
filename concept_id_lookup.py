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
# Grouped by drug class; all entries use the stable RxNorm CUI as concept_code
RXNORM_LOOKUP = {
    # ── Previously queried (chemo) ──────────────────────────────────────────
    'doxorubicin':    ('3639',   'Doxorubicin'),    # use 3639 (name fallback confirmed)
    'epirubicin':     ('3995',   'Epirubicin'),     # use 3995 (name fallback confirmed)
    'trastuzumab':    ('224905', 'Trastuzumab'),

    # ── SGLT2 inhibitors ────────────────────────────────────────────────────
    'empagliflozin':  ('1545653', 'Empagliflozin'),
    'canagliflozin':  ('1373458', 'Canagliflozin'),
    'dapagliflozin':  ('1488564', 'Dapagliflozin'),
    'ertugliflozin':  ('2003113', 'Ertugliflozin'),

    # ── GLP-1 receptor agonists ─────────────────────────────────────────────
    'semaglutide':    ('2200644', 'Semaglutide'),
    'liraglutide':    ('475968',  'Liraglutide'),
    'dulaglutide':    ('1551297', 'Dulaglutide'),
    'exenatide':      ('60548',   'Exenatide'),
    'tirzepatide':    ('2395481', 'Tirzepatide'),

    # ── DPP4 inhibitors ─────────────────────────────────────────────────────
    'sitagliptin':    ('593411',  'Sitagliptin'),
    'saxagliptin':    ('857974',  'Saxagliptin'),
    'alogliptin':     ('1240995', 'Alogliptin'),
    'linagliptin':    ('1100699', 'Linagliptin'),

    # ── Sulfonylureas ───────────────────────────────────────────────────────
    'glipizide':      ('4782',    'Glipizide'),
    'glyburide':      ('4815',    'Glyburide'),
    'glimepiride':    ('25789',   'Glimepiride'),

    # ── Thiazolidinediones ──────────────────────────────────────────────────
    'pioglitazone':   ('33738',   'Pioglitazone'),
    'rosiglitazone':  ('213051',  'Rosiglitazone'),

    # ── Insulins (individual ingredients; concept_ancestor captures formulations)
    'insulin_glargine':  ('274783', 'Insulin glargine'),
    'insulin_lispro':    ('86009',  'Insulin lispro'),
    'insulin_aspart':    ('86705',  'Insulin aspart'),
    'insulin_detemir':   ('564992', 'Insulin detemir'),
    'insulin_degludec':  ('1992693','Insulin degludec'),
    'insulin_nph':       ('5856',   'Insulin NPH'),
    'insulin_regular':   ('5755',   'Insulin regular'),

    # ── ARNI ────────────────────────────────────────────────────────────────
    'sacubitril':     ('1657973', 'Sacubitril'),
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
print('Flag → SNOMED code → OMOP concept_id')
print(SEP)
code_to_row = {r['snomed_code']: r for _, r in snomed_df.iterrows()}
for flag, (code, expected) in SNOMED_LOOKUP.items():
    row = code_to_row.get(code)
    if row is not None:
        print(f"  {flag:<22} SNOMED {code:<12} → concept_id {int(row['concept_id']):>10,}  '{row['concept_name']}'")
    else:
        print(f"  {flag:<22} SNOMED {code:<12} → *** NOT FOUND ***")

not_found_cond = [flag for flag, (code, _) in SNOMED_LOOKUP.items() if code not in code_to_row]
if not_found_cond:
    print(f'\n{SEP}\nName-based fallback for unfound conditions:\n{SEP}')
    terms = ' OR '.join([f"LOWER(concept_name) LIKE '%{SNOMED_LOOKUP[f][1].lower()[:20]}%'"
                         for f in not_found_cond])
    q_name = f"""
    SELECT concept_id, concept_name, concept_code, vocabulary_id, standard_concept, domain_id
    FROM `{CDR}.concept`
    WHERE standard_concept = 'S' AND domain_id = 'Condition' AND ({terms})
    ORDER BY concept_name
    """
    print(client.query(q_name).to_dataframe().to_string(index=False))

# ── 2. Drug concept lookup ────────────────────────────────────────────────────
print(f'\n{"=" * 70}')
print('DRUG concept IDs (by RxNorm concept code / CUI, Ingredient class)')
print('=' * 70)

rxnorm_codes = [code for code, _ in RXNORM_LOOKUP.values()]
q_rx = f"""
SELECT
    concept_id,
    concept_name,
    concept_code   AS rxnorm_code,
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
print('Drug → RxNorm code → OMOP concept_id')
print(SEP)
rx_to_row = {r['rxnorm_code']: r for _, r in rx_df.iterrows()}
for flag, (code, expected) in RXNORM_LOOKUP.items():
    row = rx_to_row.get(code)
    if row is not None:
        print(f"  {flag:<20} RxNorm {code:<8} → concept_id {int(row['concept_id']):>10,}  '{row['concept_name']}'")
    else:
        print(f"  {flag:<20} RxNorm {code:<8} → *** NOT FOUND ***")

rx_not_found = [flag for flag, (code, _) in RXNORM_LOOKUP.items() if code not in rx_to_row]
if rx_not_found:
    print(f'\n{SEP}\nName-based fallback for unfound drugs:\n{SEP}')
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

print('\n\nLOOKUP COMPLETE — paste output to update build_master.py')
