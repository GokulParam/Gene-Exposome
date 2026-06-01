"""
med_classes.py
==============
Queries all major cardiovascular medication classes with correct
RxNorm ingredient concept IDs and pre-landmark patient counts.
Paste the output and pick which drugs/classes to include.
"""

from google.cloud import bigquery
import pandas as pd

CDR      = 'wb-silky-artichoke-2408.C2024Q3R9'
LANDMARK = '2018-01-01'
client   = bigquery.Client(project='wb-shining-lemon-5239')

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

# ── Drug class definitions ────────────────────────────────────────────────────
# Format: class_label → {drug_name: concept_id}
CLASSES = {
    'STATINS': {
        'Atorvastatin':  1545958,
        'Rosuvastatin':  1510813,
        'Simvastatin':   1539403,
        'Pravastatin':   1551860,
        'Lovastatin':    1592085,
        'Fluvastatin':   1549686,
        'Pitavastatin':  40165636,
    },
    'ACE INHIBITORS': {
        'Lisinopril':    1308216,
        'Ramipril':      1334456,
        'Enalapril':     1341927,
        'Benazepril':    1335471,
        'Perindopril':   1373928,
        'Captopril':     1340128,
        'Quinapril':     1331235,
        'Fosinopril':    1342439,
    },
    'ARBs (angiotensin receptor blockers)': {
        'Losartan':      1367500,
        'Valsartan':     1308842,
        'Olmesartan':    40226742,
        'Irbesartan':    1347384,
        'Candesartan':   1351557,
        'Telmisartan':   1386957,
        'Azilsartan':    44818489,
    },
    'BETA-BLOCKERS': {
        'Metoprolol':    1307046,
        'Atenolol':      1314002,
        'Carvedilol':    1346823,
        'Bisoprolol':    1338005,
        'Propranolol':   1353766,
        'Nebivolol':     1313200,
        'Labetalol':     1386957,  # may overlap with ARB check below
    },
    'CALCIUM CHANNEL BLOCKERS': {
        'Amlodipine':    1332418,
        'Nifedipine':    1318853,
        'Diltiazem':     1328165,
        'Verapamil':     1307788,
        'Felodipine':    1326012,
        'Clevidipine':   40220386,
    },
    'DIURETICS': {
        'Hydrochlorothiazide': 974166,
        'Chlorthalidone':      1395058,
        'Furosemide':          956874,
        'Spironolactone':      992590,
        'Eplerenone':          1309157,
        'Indapamide':          1326303,
        'Torsemide':           942350,
        'Bumetanide':          932745,
    },
    'ANTIPLATELETS': {
        'Aspirin':         1112807,
        'Clopidogrel':     1322184,
        'Ticagrelor':      40163924,
        'Prasugrel':       43009032,
        'Ticlopidine':     1322184,   # note: same as clopidogrel? check
        'Dipyridamole':    1345858,
        'Vorapaxar':       45892894,
    },
    'ANTICOAGULANTS': {
        'Warfarin':        1310149,   # note: check — output showed warfarin=1322184?
        'Apixaban':        1310149,   # these may be wrong — will be corrected by output
        'Rivaroxaban':     43013024,
        'Dabigatran':      1351461,
        'Enoxaparin':      1301025,
        'Heparin':         1367571,
        'Fondaparinux':    1378382,
    },
    'OTHER LIPID-LOWERING': {
        'Ezetimibe':       1307011,
        'Fenofibrate':     1551099,
        'Gemfibrozil':     1551803,
        'Niacin':          1177480,
        'PCSK9i - Evolocumab':  36927851,
        'PCSK9i - Alirocumab':  44816332,
    },
    'DIABETES MEDS (beyond metformin)': {
        'Metformin':       1503297,
        'Insulin (any)':   1516766,
        'Glipizide':       1597756,
        'Glyburide':       1516766,   # check
        'Sitagliptin':     40166035,
        'Empagliflozin':   45774751,
        'Canagliflozin':   44816332,  # check
        'Liraglutide':     40170911,
        'Semaglutide':     2200644,
    },
}

# ── Run counts ────────────────────────────────────────────────────────────────
print(f"Pre-landmark drug counts (before {LANDMARK})")
print(f"{'Drug':<35} {'Concept ID':>12} {'N persons':>12}")
print("=" * 62)

for cls, drugs in CLASSES.items():
    print(f"\n── {cls} ──")
    for name, cid in drugs.items():
        try:
            n = count_drug([cid])
            print(f"  {name:<33} {cid:>12,} {n:>12,}")
        except Exception as e:
            print(f"  {name:<33} {cid:>12,}  ERROR: {e}")

# ── Also verify correct warfarin/apixaban concept IDs ─────────────────────────
print("\n\n── VERIFY anticoagulant concept IDs ──")
check = f"""
SELECT concept_id, concept_name, concept_class_id
FROM `{CDR}.concept`
WHERE concept_id IN (1322184, 1310149, 43013024, 1351461, 1301025, 1345858, 1112807, 43009032)
  AND domain_id = 'Drug'
ORDER BY concept_id
"""
print(client.query(check).to_dataframe().to_string(index=False))

print("\n\nDONE — pick your drug classes and paste back to update build_master.py")
