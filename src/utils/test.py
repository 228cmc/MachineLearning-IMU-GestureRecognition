# test.py
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# --- bootstrap: ensure project root is importable (for config.py, Results/, data/, etc.)
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # ...\Gesture
SRC  = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- end bootstrap
from config import MODELS_DIR

BASE_DIR = os.path.join(ROOT, "Results", "tables")
print("DEBUG __file__ =", __file__)
print("DEBUG ROOT     =", ROOT)
print("DEBUG BASE_DIR =", BASE_DIR)
print("DEBUG EXISTS?  =", os.path.isdir(BASE_DIR))
if os.path.isdir(BASE_DIR):
    print("DEBUG CSVs    =", [f for f in os.listdir(BASE_DIR) if f.endswith(".csv")])

# Comparisons use base names (without seeds); resolver will find the *_seeds*.csv files.
COMPARISONS = [
    # KNN
    ("knn_few_cv_handcrafted.csv",      "knn_few_aug_cv_handcrafted.csv",   "KNN | handcrafted | noaug vs aug"),
    ("knn_few_cv_signature.csv",        "knn_few_aug_cv_signature.csv",     "KNN | signature   | noaug vs aug"),
    ("knn_few_cv_handcrafted.csv",      "knn_few_cv_signature.csv",         "KNN | noaug | handcrafted vs signature"),
    ("knn_few_aug_cv_handcrafted.csv",  "knn_few_aug_cv_signature.csv",     "KNN | aug   | handcrafted vs signature"),
    # RF
    ("rf_few_cv_handcrafted.csv",       "rf_aug_few_cv_handcrafted.csv",    "RF | handcrafted | noaug vs aug"),
    ("rf_few_cv_signature.csv",         "rf_aug_few_cv_signature.csv",      "RF | signature   | noaug vs aug"),
    ("rf_few_cv_handcrafted.csv",       "rf_few_cv_signature.csv",          "RF | noaug | handcrafted vs signature"),
    ("rf_aug_few_cv_handcrafted.csv",   "rf_aug_few_cv_signature.csv",      "RF | aug   | handcrafted vs signature"),
]

COLUMN = "acc"


def _resolve_path(base_name: str) -> str:
    """
    If the exact file exists, return it.
    Otherwise, search for a *_seeds*.csv matching the prefix (before .csv).
    Returns the most recent one by modification time.
    """
    exact = os.path.join(BASE_DIR, base_name)
    if os.path.exists(exact):
        return exact
    stem = os.path.splitext(base_name)[0]
    pattern = os.path.join(BASE_DIR, f"{stem}_seeds*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"Neither {exact} nor {pattern} found")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_series(path: str, col: str) -> pd.DataFrame:
    """
    Load CSV and return DataFrame with ['seed', col].
    Requires 'seed' column to align paired samples.
    """
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {os.path.basename(path)}. Available: {df.columns.tolist()}")
    if "seed" not in df.columns:
        raise ValueError(f"'seed' column missing in {os.path.basename(path)} to align episodes.")
    out = df[["seed", col]].dropna().copy()
    out.rename(columns={col: "value"}, inplace=True)
    return out


def cohens_d(vals1, vals2):
    diff = vals1 - vals2
    return np.mean(diff) / np.std(diff, ddof=1)


def cliffs_delta(vals1, vals2):
    n1, n2 = len(vals1), len(vals2)
    bigger = sum(x > y for x in vals1 for y in vals2)
    smaller = sum(x < y for x in vals1 for y in vals2)
    return (bigger - smaller) / (n1 * n2)


def evaluate(file1_base, file2_base, col, desc):
    path1 = _resolve_path(file1_base)
    path2 = _resolve_path(file2_base)

    s1 = _load_series(path1, col)
    s2 = _load_series(path2, col)

    # Align by common seeds
    merged = pd.merge(s1, s2, on="seed", how="inner", suffixes=("_1", "_2"))
    if merged.empty:
        raise ValueError(f"No overlapping seeds between {os.path.basename(path1)} and {os.path.basename(path2)}")
    vals1 = merged["value_1"].to_numpy()
    vals2 = merged["value_2"].to_numpy()

    mean1, std1 = np.mean(vals1), np.std(vals1, ddof=1)
    mean2, std2 = np.mean(vals2), np.std(vals2, ddof=1)
    t_stat, p_val = ttest_rel(vals1, vals2)
    d = cohens_d(vals1, vals2)
    delta = cliffs_delta(vals1, vals2)

    print(f"\n=== {desc} ===")
    print(f"{os.path.basename(path1)}: {mean1:.4f} ± {std1:.4f} (n={len(vals1)})")
    print(f"{os.path.basename(path2)}: {mean2:.4f} ± {std2:.4f} (n={len(vals2)})")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_val:.4e}")
    print(f"Cohen's d: {d:.4f}")
    print(f"Cliff's delta: {delta:.4f}")

    out_dir = os.path.join(ROOT, "Results", "stats")
    os.makedirs(out_dir, exist_ok=True)
    safe_desc = desc.lower().replace(" | ", "_").replace(" ", "_")
    out_file = os.path.join(out_dir, f"stats_{safe_desc}_{col}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"=== {desc} ===\n")
        f.write(f"{os.path.basename(path1)}: {mean1:.4f} ± {std1:.4f} (n={len(vals1)})\n")
        f.write(f"{os.path.basename(path2)}: {mean2:.4f} ± {std2:.4f} (n={len(vals2)})\n")
        f.write(f"Paired t-test: t={t_stat:.4f}, p={p_val:.4e}\n")
        f.write(f"Cohen's d: {d:.4f}\n")
        f.write(f"Cliff's delta: {delta:.4f}\n")
    print(f"[OK] Saved to {out_file}")


if __name__ == "__main__":
    for f1, f2, desc in COMPARISONS:
        evaluate(f1, f2, COLUMN, desc)
