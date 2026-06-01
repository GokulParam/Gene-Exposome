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

SEP = '─' * 70

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — MACE / outcome concept ID verification
# Checks the concept IDs used in the cohort-building cell (CELL 2)
# ══════════════════════════════════════════════════════════════════════════════

MACE_CONCEPT_IDS = {
    # MACE events
    'MI ancestor':                4329847,
    'CVA (stroke, broad)':        443454,
    'Ischaemic stroke':           375557,
    'Haemorrhagic stroke':        432923,
    # CVD death cause concepts
    'Cardiac arrest':             321042,
    'Sudden cardiac death':       4059796,
    'Heart failure':              316139,   # confirmed correct in earlier lookup
    'Cardiogenic shock':          40479586,
    'Other specified heart dis.': 4108812,
    'Hypertensive heart disease': 312927,
    'Ischaemic heart disease':    4108814,
    'Cerebrovascular disease':    4185932,
}

print('=' * 70)
print('SECTION A — MACE / CVD outcome concept ID verification')
print('=' * 70)

mace_ids = list(MACE_CONCEPT_IDS.values())
q_mace = f"""
SELECT concept_id, concept_name, domain_id, vocabulary_id, standard_concept
FROM `{CDR}.concept`
WHERE concept_id IN ({','.join(str(i) for i in mace_ids)})
ORDER BY concept_id
"""
mace_df = client.query(q_mace).to_dataframe()
id_to_name = dict(zip(mace_df['concept_id'], mace_df['concept_name']))

print(f'\n{SEP}')
print(f"  {'Label':<32} {'ID':>10}  {'Status':<12}  Actual name in CDR")
print(SEP)
for label, cid in MACE_CONCEPT_IDS.items():
    actual = id_to_name.get(cid, '*** NOT FOUND ***')
    ok = '✓' if cid in id_to_name else '✗ NOT FOUND'
    print(f"  {label:<32} {cid:>10,}  {ok:<12}  '{actual}'")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Conditions: search by SNOMED concept code
# ══════════════════════════════════════════════════════════════════════════════

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
    # ── Anticoagulants / antiplatelets (unverified in previous run) ──────────
    'rivaroxaban':    ('1114195', 'Rivaroxaban'),
    'dabigatran':     ('1037042', 'Dabigatran'),
    'ticagrelor':     ('1116632', 'Ticagrelor'),

    # ── Previously queried (chemo — re-confirm) ──────────────────────────────
    'doxorubicin':    ('3639',   'Doxorubicin'),
    'epirubicin':     ('3995',   'Epirubicin'),
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

    # ── Insulins ────────────────────────────────────────────────────────────
    'insulin_glargine':  ('274783', 'Insulin glargine'),
    'insulin_lispro':    ('86009',  'Insulin lispro'),
    'insulin_aspart':    ('86705',  'Insulin aspart'),
    'insulin_detemir':   ('564992', 'Insulin detemir'),
    'insulin_degludec':  ('1992693','Insulin degludec'),

    # ── ARNI ────────────────────────────────────────────────────────────────
    'sacubitril':     ('1657973', 'Sacubitril'),
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Condition lookup by SNOMED code
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n\n{"=" * 70}')
print('SECTION B — Condition concept IDs (by SNOMED code)')
print('=' * 70)

snomed_codes = [code for code, _ in SNOMED_LOOKUP.values()]
q_snomed = f"""
SELECT concept_id, concept_name, concept_code AS snomed_code, domain_id, standard_concept
FROM `{CDR}.concept`
WHERE vocabulary_id = 'SNOMED'
  AND concept_code IN ({','.join(f"'{c}'" for c in snomed_codes)})
  AND standard_concept = 'S'
ORDER BY concept_code
"""
snomed_df = client.query(q_snomed).to_dataframe()
print(snomed_df.to_string(index=False))

print(f'\n{SEP}')
code_to_row = {r['snomed_code']: r for _, r in snomed_df.iterrows()}
for flag, (code, expected) in SNOMED_LOOKUP.items():
    row = code_to_row.get(code)
    if row is not None:
        print(f"  {flag:<22} SNOMED {code:<12} → {int(row['concept_id']):>10,}  '{row['concept_name']}'")
    else:
        print(f"  {flag:<22} SNOMED {code:<12} → *** NOT FOUND ***")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Drug lookup by RxNorm code
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n\n{"=" * 70}')
print('SECTION C — Drug concept IDs (by RxNorm code, Ingredient class)')
print('=' * 70)

rxnorm_codes = [code for code, _ in RXNORM_LOOKUP.values()]
q_rx = f"""
SELECT concept_id, concept_name, concept_code AS rxnorm_code, concept_class_id, standard_concept
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
rx_to_row = {r['rxnorm_code']: r for _, r in rx_df.iterrows()}
for flag, (code, expected) in RXNORM_LOOKUP.items():
    row = rx_to_row.get(code)
    if row is not None:
        print(f"  {flag:<20} RxNorm {code:<8} → {int(row['concept_id']):>10,}  '{row['concept_name']}'")
    else:
        print(f"  {flag:<20} RxNorm {code:<8} → *** NOT FOUND ***")

rx_not_found = [flag for flag, (code, _) in RXNORM_LOOKUP.items() if code not in rx_to_row]
if rx_not_found:
    print(f'\n{SEP}\nName-based fallback for unfound drugs:\n{SEP}')
    drug_names = [RXNORM_LOOKUP[f][1].lower() for f in rx_not_found]
    drug_terms = ' OR '.join([f"LOWER(concept_name) LIKE '%{n}%'" for n in drug_names])
    q_fallback = f"""
    SELECT concept_id, concept_name, concept_code, vocabulary_id, concept_class_id, standard_concept
    FROM `{CDR}.concept`
    WHERE standard_concept = 'S' AND domain_id = 'Drug' AND concept_class_id = 'Ingredient'
      AND ({drug_terms})
    ORDER BY concept_name
    """
    print(client.query(q_fallback).to_dataframe().to_string(index=False))

print('\n\nLOOKUP COMPLETE')
print('Section A: verify MACE concept names are as expected')
print('Section C: update rivaroxaban/dabigatran/ticagrelor in build_master.py if IDs change')
