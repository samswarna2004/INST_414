import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

csv_path = DATA / "STAGE_3_SPRINT_2.csv"

print(">>> Working dir:", ROOT)
print(">>> Expecting CSV at:", csv_path)

# ---- load CSV ----
if not csv_path.exists():
    print("!! CSV not found. Make sure the file is here:", csv_path)
    sys.exit(1)

df = pd.read_csv(csv_path)
if df.shape[1] == 1:  # fallback for semicolon delimiter
    df = pd.read_csv(csv_path, sep=";")

print(">>> Loaded shape:", df.shape)
print(">>> First few rows:")
print(df.head())

# ---- clean numerics ----
for c in ['strokes', 'sg_ott', 'sg_total']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- Histogram of strokes ----
if "strokes" in df.columns:
    plt.figure()
    df["strokes"].dropna().hist(bins=30, color="skyblue", edgecolor="black")
    plt.title("Distribution of Strokes")
    plt.xlabel("Strokes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGS / "hist_strokes.png", dpi=300)
    print(">>> Saved histogram:", FIGS / "hist_strokes.png")

# ---- Scatterplot example ----
if "sg_ott" in df.columns and "sg_total" in df.columns:
    plt.figure()
    plt.scatter(df["sg_ott"], df["sg_total"], alpha=0.5, color="darkgreen")
    plt.title("Strokes Gained Off the Tee vs Total")
    plt.xlabel("Strokes Gained: Off the Tee")
    plt.ylabel("Strokes Gained: Total")
    plt.tight_layout()
    plt.savefig(FIGS / "scatter_sg.png", dpi=300)
    print(">>> Saved scatterplot:", FIGS / "scatter_sg.png")
