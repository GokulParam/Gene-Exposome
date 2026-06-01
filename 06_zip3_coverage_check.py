"""
CELL 6 — Check which zip3 codes are unmatched in exposome files
"""
import os, pandas as pd

WORKSPACE    = '/home/dataproc/workspaces/geneexposome'
EXPOSOME_DIR = f'{WORKSPACE}/Exposome'

master = pd.read_csv(f'{WORKSPACE}/master_dataset.csv',
                     usecols=['person_id', 'zip3'], dtype=str)

print(f"Cohort size: {len(master):,}")
print(f"zip3 coverage: {master['zip3'].notna().sum():,} ({100*master['zip3'].notna().mean():.1f}%)\n")

# Cohort zip3 counts
cohort_zip3 = (master['zip3']
               .dropna()
               .str.strip()
               .str.extract(r'(\d+)', expand=False)
               .str.zfill(3))

cohort_counts = cohort_zip3.value_counts().rename('n_people')

# Check each exposome file
for fname in ['GEE_Final_3Zip.csv', 'Zip3_Social_Exposome_Final.csv',
              'Noise_3Zip.csv', 'SMART_3Zip.csv',
              'Toxins_3Zip.csv', 'Wildfire_3Zip.csv']:
    fpath = os.path.join(EXPOSOME_DIR, fname)
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    zip_col = next((c for c in df.columns if 'zip' in c.lower()), df.columns[0])
    exp_zip3 = (df[zip_col].astype(str).str.strip()
                .str.extract(r'(\d+)', expand=False).str.zfill(3))
    unmatched = cohort_counts[~cohort_counts.index.isin(exp_zip3)]
    print(f"{fname}:")
    print(f"  Exposome zip3 count : {exp_zip3.nunique()}")
    print(f"  Unmatched zip3 codes: {len(unmatched)}  "
          f"({unmatched.sum():,} people, {100*unmatched.sum()/len(master):.1f}%)")
    if len(unmatched):
        print(f"  Unmatched zip3 list : {sorted(unmatched.index.tolist())}")
    print()
