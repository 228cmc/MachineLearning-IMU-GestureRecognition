# utils.py
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib as mpl

from config import (
    NOISE_REPS, MIXUP_REPS, KNN_NEIGHBORS_FEWSHOT,
    RESULTS_DIR, PLOTS_DIR, TABLES_DIR, METRICS_DIR
)


LIGHT_BG   = "#f5f6f8"
GRID_COLOR = "#d0d5db"
FS_ANN  = 16
FS_TICK = 15
CMAP_GREEN = mpl.colormaps.get_cmap("Greens")
CMAP_LILAC = mpl.colormaps.get_cmap("Purples")

#logger 
RESULTS_LOG: list[dict] = []

def log_result(dataset: str, model: str, experiment: str, k: int | None,
               acc: float, f1_weighted: float, f1_macro: float):
    from datetime import datetime
    RESULTS_LOG.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_type": dataset,
        "model": model,
        "experiment": experiment,
        "k_per_class": k,
        "accuracy": round(acc, 6),
        "f1_weighted": round(f1_weighted, 6),
        "f1_macro": round(f1_macro, 6),
    })

def save_results_table(output_dir=TABLES_DIR,
                       filename="experiments_summary_selectindex.csv"):
    """
    Save RESULTS_LOG to a CSV file, appending rows when the file exists.

    Parameters:
    - output_dir (str): Target directory for the CSV.
    - filename (str): Output CSV filename.

    Returns:
    - None. Saves a CSV to {output_dir}/{filename}, appending new rows.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df = pd.DataFrame(RESULTS_LOG)
    if df.empty:
        print("[WARN] No hay filas en RESULTS_LOG; no se escribió tabla.")
        return
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    df.to_csv(out_path, mode="a", header=write_header,
              index=False, encoding="utf-8")

# Data 
def load_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def _pick_feature_columns(df: pd.DataFrame, feature_mode: str) -> list[str]:
    if feature_mode == "signature":
        cols = [c for c in df.columns if c.startswith("Sig_")]
    elif feature_mode == "handcrafted":
        cols = [c for c in df.columns if c.startswith("Feature_")]
    else:
        raise ValueError("feature_mode debe ser 'signature' o 'handcrafted'")
    if cols:
        return cols
    drop_cols = [c for c in ["Timestamp", "Exercise_Type", "Label"] if c in df.columns]
    cols = (df.select_dtypes(include=[np.number])
              .drop(columns=drop_cols, errors="ignore")
              .columns.tolist())
    if not cols:
        raise ValueError("No encontré columnas numéricas para usar como features.")
    return cols

def preprocess_multiclass(df: pd.DataFrame, feature_mode: str):
    if "Exercise_Type" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'Exercise_Type' para multiclase.")
    if "Label" in df.columns:
        before = len(df)
        df = df[df["Label"] == 1].copy()
    df = df[df["Exercise_Type"].notna()].copy()
    feature_cols = _pick_feature_columns(df, feature_mode)
    label_col = "Exercise_Type"
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    y = df[label_col].values
    return X, y, feature_cols, label_col

def load_Xy(csv_path: str, feature_mode: str):
    df = load_dataframe(csv_path)
    X, y, feat_cols, y_col = preprocess_multiclass(df, feature_mode)
    return df, X, y, feat_cols, y_col

#  Few-shot split 
def select_k_per_class_indices(y: np.ndarray, k: int = 5, seed: int = 42, strict: bool = True):
    rng = np.random.default_rng(seed)
    train_chunks, test_chunks = [], []
    classes = np.unique(y)
    for cls in classes:
        pool = np.where(y == cls)[0]
        rng.shuffle(pool)
        if strict and len(pool) < k:
            raise ValueError(f"Class {cls} has {len(pool)} < k={k}")
        k_take = min(k, len(pool))
        train_cls = pool[:k_take]
        test_cls  = pool[k_take:]
        train_chunks.append(train_cls)
        test_chunks.append(test_cls)
    train_idx = np.concatenate(train_chunks) if train_chunks else np.array([], dtype=int)
    test_idx  = np.concatenate(test_chunks)  if test_chunks  else np.array([], dtype=int)
    return train_idx, test_idx

# -Metrics
def print_classification_metrics(y_true, y_pred, title="Metrics", save_name=None):
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    print(f"\n{title}")
    print(f"Accuracy: {acc:.4f} | F1-weighted: {f1w:.4f} | F1-macro: {f1m:.4f}\n")
    print("Classification Report:\n" + report)
    if save_name:
        metrics_root = os.path.join("Results", "metrics")
        os.makedirs(metrics_root, exist_ok=True)
        out_path = os.path.join(metrics_root, f"{save_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write(f"Accuracy: {acc:.4f} | F1-weighted: {f1w:.4f} | F1-macro: {f1m:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"[OK] Metrics saved at {out_path}")
    return acc, f1w, f1m

def save_confusion_matrix(y_true, y_pred, title: str, filename: str,
                          output_dir: str = os.path.join(PLOTS_DIR, "confusion"),
                          normalize: bool = True):
    """
    Draw and save a confusion matrix image for given labels.

    Parameters:
    - y_true (ndarray): Ground truth labels.
    - y_pred (ndarray): Predicted labels.
    - title (str): Plot title.
    - filename (str): Output image base name without extension.
    - output_dir (str): Directory to save the image.
    - normalize (bool): Normalize by row if True.

    Returns:
    - None. Saves a PNG to {output_dir}/{filename}.png.
    """
    labels = np.unique(y_true)
    norm = 'true' if normalize else None
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm)
    cmap = CMAP_LILAC if "handcrafted" in filename.lower() else CMAP_GREEN
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.set_facecolor(LIGHT_BG)
    im = ax.imshow(cm, cmap=cmap, vmin=0.0, vmax=1.0 if normalize else None)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                    ha="center", va="center",
                    fontsize=FS_ANN, fontweight="bold",
                    color=("white" if v >= 0.55 else "#1a1a1a"))
    pretty = [str(l).replace("_", " ") for l in labels]
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticklabels(pretty, rotation=90, ha="center", va="top", fontsize=FS_TICK)
    ax.set_yticklabels(pretty, fontsize=FS_TICK)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linestyle=":", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, pad=10, fontsize=20)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{filename}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"[OK] Confusion matrix saved at {out_path}")

# Few-shot episodic evaluation (N episodios) 
def episodic_cv(run_once_fn, seeds, model, **kwargs):
    """
    Run a few-shot experiment across multiple seeds (episodes).

    Parameters
    ----------
    run_once_fn : callable
        Function that executes a single episode. Must accept the kwargs from
        `episodic_cv` plus `seed`, and return either:
        - A tuple (acc, f1w, f1m), or
        - A scalar acc (in which case f1w and f1m will be NaN).
    seeds : iterable
        List or set of integer seeds for independent episodes.
    model : str
        Label for the model/condition. Used to name the output CSV.
    **kwargs :
        Additional arguments passed through to `run_once_fn`.

    Returns
    -------
    rows : list[dict]
        A list of dictionaries with keys {"seed","acc","f1w","f1m"} for each episode.
    summary : dict
        Dictionary of summary statistics:
        {
            "acc_mean","acc_std",
            "f1w_mean","f1w_std",
            "f1m_mean","f1m_std"
        }

    Side Effects
    ------------
    Writes a **single** CSV file with all per-episode results at:
        Results/tables/{model}_seeds<min>_<max>.csv

    Notes
    -----
    - If `run_once_fn` returns only accuracy, f1w and f1m are recorded as NaN.
    - The CSV contains one row per episode with columns: seed, acc, f1w, f1m.
    - <min> and <max> are the minimum and maximum seed values provided.
    """
    import random 

    rows = []
    for s in seeds:
        # -reseed for episode
        np.random.seed(int(s))
        random.seed(int(s))

        kargs = dict(kwargs)
        kargs["seed"] = s
        out = run_once_fn(**kargs)
        if isinstance(out, tuple) and len(out) == 3:
            acc, f1w, f1m = out
        else:
            acc, f1w, f1m = out, np.nan, np.nan
        rows.append({"seed": s, "acc": float(acc), "f1w": float(f1w), "f1m": float(f1m)})

    df = pd.DataFrame(rows)
    accs = df["acc"].values
    f1ws = df["f1w"].values[~np.isnan(df["f1w"].values)]
    f1ms = df["f1m"].values[~np.isnan(df["f1m"].values)]

    summary = {
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
        "f1w_mean": float(f1ws.mean()) if f1ws.size else np.nan,
        "f1w_std":  float(f1ws.std(ddof=1)) if f1ws.size else np.nan,
        "f1m_mean": float(f1ms.mean()) if f1ms.size else np.nan,
        "f1m_std":  float(f1ms.std(ddof=1)) if f1ms.size else np.nan,
    }

    out_dir = os.path.join("Results", "tables")
    os.makedirs(out_dir, exist_ok=True)


    try:
        seed_min, seed_max = int(np.min(seeds)), int(np.max(seeds))
        seed_tag = f"seeds{seed_min}_{seed_max}"
    except Exception:
        seed_tag = "seeds_custom"
    archived_path = os.path.join(out_dir, f"{model}_{seed_tag}.csv")
    df.to_csv(archived_path, index=False, encoding="utf-8")

    return rows, summary


# Aggregation of predictions by episodes
def episodic_collect_predictions(run_once_pred_fn, seeds, **kwargs):
    import numpy as np, random
    seeds_list = sorted({int(s) for s in seeds})  # orden y únicos
    all_true, all_pred, per_seed = [], [], {}
    for s in seeds_list:
        # reseed before each episode
        np.random.seed(s)
        random.seed(s)
        kargs = dict(kwargs); kargs["seed"] = s
        y_true, y_pred = run_once_pred_fn(**kargs)
        all_true.extend(y_true); all_pred.extend(y_pred)
        per_seed[s] = (np.asarray(y_true), np.asarray(y_pred))
    return np.asarray(all_true), np.asarray(all_pred), per_seed

# confusion matrix  
def save_confusion_matrix_counts(
    y_true,
    y_pred,
    labels,
    model_name, 
    filename,
    output_dir=os.path.join(PLOTS_DIR, "confusion"),
    normalize=True,
):

    """
    Draw and save a confusion matrix with optional normalization and weighted F1 in the title.

    Parameters:
    - y_true (ndarray): Ground truth labels.
    - y_pred (ndarray): Predicted labels.
    - labels (ndarray or list): Label set and order for the axes.
    - model_name (str): Title prefix for the plot.
    - filename (str): Output image base name without extension.
    - output_dir (str): Directory to save the figure.
    - normalize (bool): Normalize by row if True.

    Returns:
    - None. Saves a PNG to {output_dir}/{filename}.png.
    """
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm

    cmap = CMAP_LILAC if "handcrafted" in filename.lower() else CMAP_GREEN

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    im = ax.imshow(
        cm_plot,
        vmin=0.0 if normalize else None,
        vmax=1.0 if normalize else None,
        cmap=cmap,
    )

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            v = cm_plot[i, j]
            ax.text(
                j, i,
                f"{v:.2f}" if normalize else f"{int(cm[i,j])}",
                ha="center", va="center",
                fontsize=FS_ANN, fontweight="bold",
                color=("white" if (normalize and v >= 0.55) else "#1a1a1a"),
            )

    pretty = [str(l).replace("_", " ") for l in labels]
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(pretty, rotation=90, fontsize=FS_TICK)
    ax.set_yticklabels(pretty, fontsize=FS_TICK)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linestyle=":", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ax.set_title(f"Confusion Matrix – F1 score: {f1w*100:.2f}%", pad=15, fontsize=20)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(out_path, dpi=300, facecolor=LIGHT_BG, bbox_inches="tight")
    plt.close(fig)
