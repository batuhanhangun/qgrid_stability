#!/usr/bin/env python
"""Download the UCI Electrical Grid Stability Simulated Data (dataset 471).

Uses only stdlib + pandas (no ucimlrepo):

    python scripts/download_data.py

Writes data/grid_stability.csv with columns tau1-4, p1-4, g1-4, stab, stabf.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = ("https://archive.ics.uci.edu/static/public/471/"
       "electrical+grid+stability+simulated+data.zip")

out = Path(__file__).resolve().parents[1] / "data" / "grid_stability.csv"
out.parent.mkdir(exist_ok=True)

print(f"Downloading {URL} ...")
with urllib.request.urlopen(URL, timeout=120) as r:
    blob = r.read()

with zipfile.ZipFile(io.BytesIO(blob)) as zf:
    csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    with zf.open(csv_name) as f:
        df = pd.read_csv(f)

df.columns = [c.strip() for c in df.columns]
expected = ([f"tau{i}" for i in range(1, 5)] + [f"p{i}" for i in range(1, 5)]
            + [f"g{i}" for i in range(1, 5)] + ["stab", "stabf"])
missing = set(expected) - set(df.columns)
if missing:
    raise SystemExit(f"Unexpected schema, missing: {sorted(missing)}")

df = df[expected]
df.to_csv(out, index=False)
frac = (df["stabf"].str.strip() == "unstable").mean()
print(f"Saved {len(df)} rows to {out} | unstable fraction: {frac:.3f}")
