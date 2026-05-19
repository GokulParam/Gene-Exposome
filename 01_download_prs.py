"""
CELL 1 — Download PRS scores from workspace bucket
====================================================
Run this once at the start of your session to pull all PRS files from GCS
into local Jupyter storage. Safe to re-run; gsutil -m cp overwrites silently.
"""

import subprocess
import os

# ── Workspace paths ────────────────────────────────────────────────────────
# WORKSPACE_BUCKET is set automatically by All of Us for your researcher workspace.
# It points to the GCS bucket that persists across sessions.
BUCKET = os.environ['WORKSPACE_BUCKET']

# Local base directory — this lives on the Jupyter persistent disk.
# Change this path if your notebook lives elsewhere.
WORKSPACE = '/home/jupyter/workspaces/geneexposome'

# Traits to download. Each maps to a subfolder in your GCS bucket and locally.
PRS_TRAITS = ['CAD', 'LDLC', 'OBESITY', 'SBP', 'T2D']

# ── Create local directories ───────────────────────────────────────────────
# exist_ok=True means this won't error if the folder already exists.
print("Creating local directories...")
for trait in PRS_TRAITS:
    local_path = f'{WORKSPACE}/PRS_Scores/{trait}'
    os.makedirs(local_path, exist_ok=True)
    print(f"  OK  {local_path}")

# ── Download all PRS scores from GCS ─────────────────────────────────────
# gsutil -m enables parallel (multi-threaded) transfer — faster for many files.
# cp -r copies recursively so it grabs all files inside the folder.
print("\nDownloading PRS scores from GCS bucket...")
print(f"Source bucket: {BUCKET}/PRS_Scores/\n")

download_errors = []

for trait in PRS_TRAITS:
    gcs_source = f'{BUCKET}/PRS_Scores/{trait}/*'
    local_dest = f'{WORKSPACE}/PRS_Scores/{trait}/'

    print(f"  [{trait}] Downloading {gcs_source} → {local_dest}")
    result = subprocess.run(
        ['gsutil', '-m', 'cp', '-r', gcs_source, local_dest],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"  [{trait}] ✓ Done")
    else:
        print(f"  [{trait}] ✗ FAILED")
        print(f"          stderr: {result.stderr.strip()}")
        download_errors.append(trait)

# ── Verify files landed ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICATION — Files present after download")
print("=" * 60)

for trait in PRS_TRAITS:
    local_path = f'{WORKSPACE}/PRS_Scores/{trait}'
    if os.path.exists(local_path):
        files = os.listdir(local_path)
        if files:
            # Show file names and sizes for quick sanity check
            for f in sorted(files):
                full = os.path.join(local_path, f)
                size_mb = os.path.getsize(full) / 1e6
                print(f"  {trait}/{f}  ({size_mb:.2f} MB)")
        else:
            print(f"  {trait}: directory exists but is EMPTY — check GCS path")
    else:
        print(f"  {trait}: directory NOT FOUND")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if not download_errors:
    print("All PRS traits downloaded successfully.")
else:
    print(f"WARNING: {len(download_errors)} trait(s) failed to download: {download_errors}")
    print("Check the GCS paths above and re-run for failed traits.")
print("=" * 60)
