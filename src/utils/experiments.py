# experiments.py
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


# bootstrap: agrega la raíz del repo al sys.path (este archivo está en src/utils/)
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # ...\Gesture
SRC  = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import MODELS_DIR


from utils.utils import load_Xy, select_k_per_class_indices
from utils.augment import DataAugmenter


# Few-shot single run (support/query split)
def run_fewshot_once(X, y, k=5, seed=42, n_neighbors=5,
                     use_augment=False, feat_cols=None, y_col=None,
                     noise_reps=2, mixup_reps=3):
    """
    Run one few-shot episode with optional augmentation using KNN classifier.

    Parameters:
    - X (ndarray): Feature matrix.
    - y (ndarray): Labels.
    - k (int): Shots per class.
    - seed (int): Random seed.
    - n_neighbors (int): Number of KNN neighbors.
    - use_augment (bool): Apply augmentation or not.
    - feat_cols (list or None): Feature column names if augmenting.
    - y_col (str or None): Label column name if augmenting.
    - noise_reps (int): Number of noisy copies.
    - mixup_reps (int): Number of Mixup repetitions.

    Returns:
    - dict: Accuracy and F1 scores.
    """
    import random, numpy as np
    # seed to make the episode independent from execution order
    np.random.seed(seed)
    random.seed(seed)

    tr_idx, te_idx = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    # 2) augment with episode seed
    if use_augment and feat_cols is not None and y_col is not None:
        base_df = pd.DataFrame(Xtr, columns=feat_cols).assign(**{y_col: ytr})
        aug_df = DataAugmenter(seed=seed).augment_dataframe(
            base_df, feat_cols, y_col, noise_reps=noise_reps, mixup_reps=mixup_reps
        )
        Xtr, ytr = aug_df[feat_cols].values, aug_df[y_col].values

    nnb = min(n_neighbors, len(Xtr))
    model = KNeighborsClassifier(n_neighbors=nnb)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)

    acc = accuracy_score(yte, ypred)
    f1w = f1_score(yte, ypred, average="weighted", zero_division=0)
    f1m = f1_score(yte, ypred, average="macro", zero_division=0)
    return {"acc": acc, "f1w": f1w, "f1m": f1m}


#  Few-shot “5-fold CV” (episodic with different seeds) 
def run_fewshot_cv(X, y, k=5, seeds=(42, 43, 44, 45, 46), **kwargs):
    """
    Run few-shot evaluation with multiple seeds and compute mean and std metrics.

    Parameters:
    - X (ndarray): Feature matrix.
    - y (ndarray): Labels.
    - k (int): Shots per class.
    - seeds (tuple): Random seeds.
    - kwargs: Extra arguments for run_fewshot_once.

    Returns:
    - tuple: DataFrame with per-seed results and dict with summary statistics.
    """
    import numpy as np
    # Normalize seed list and run few-shot episodes for each
    seeds = tuple(sorted({int(s) for s in seeds}))
    rows = []
    for s in seeds:
        res = run_fewshot_once(X, y, k=k, seed=s, **kwargs)
        res["seed"] = s
        rows.append(res)
    df = pd.DataFrame(rows)

    # Compute summary statistics (mean and std of metrics)
    summary = {
        "acc_mean": df["acc"].mean(), "acc_std": df["acc"].std(ddof=1),
        "f1w_mean": df["f1w"].mean(), "f1w_std": df["f1w"].std(ddof=1),
        "f1m_mean": df["f1m"].mean(), "f1m_std": df["f1m"].std(ddof=1),
    }
    return df, summary




def append_results(path, rowdict):
    """
    Append one row of results to a CSV with fixed schema.

    Parameters:
    - path (str): Path to CSV file.
    - rowdict (dict): Row data.

    Returns:
    - None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Define a fixed schema for all rows
    cols = [
        "mode", "shots", "features", "augment",
        "acc", "acc_std",
        "f1w", "f1w_std",
        "f1m", "f1m_std",
        "seed"  
    ]
    # Convert numpy types to Python types and map values into the fixed schema
    def _to_py(x):
        if isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        return x

    row = {c: "" for c in cols}
    for k, v in rowdict.items():
        if k in row:
            row[k] = _to_py(v)

    write_header = not os.path.exists(path)
    pd.DataFrame([row], columns=cols).to_csv(
        path, mode="a", header=write_header, index=False, encoding="utf-8"
    )


# Plot comparison single vs episodic CV varying k
def plot_fewshot_comparison(csv_path, output_path="Results/plots/fewshot_compare_line.png",
                            features_filter=None, augment_filter=None):
    """
    Plot accuracy comparison between single run and episodic CV with error bars.

    Parameters:
    - csv_path (str): Input CSV file.
    - output_path (str): Path to save plot.
    - features_filter (str or None): Filter by feature type.
    - augment_filter (int or None): Filter by augmentation flag.

    Returns:
    - None
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    # Times New Roman style
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.edgecolor"] = "#333"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    df = pd.read_csv(csv_path)

    if features_filter is not None:
        df = df[df["features"] == features_filter]
    if augment_filter is not None:
        df = df[df["augment"] == int(augment_filter)]

    single = df[df["mode"] == "single_support_query"].sort_values("shots")
    cv     = df[df["mode"] == "episodic_cv"].sort_values("shots")

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    # Single line  blue (tab:blue)
    if not single.empty:
        ax.plot(single["shots"], single["acc"],
                marker="o", linewidth=2.0, markersize=5,
                color="tab:blue",
                label="Single Run")

    # CV line with error bars , orange (tab:orange)
    if not cv.empty:
        ax.errorbar(cv["shots"], cv["acc"], yerr=cv["acc_std"],
                    marker="s", markersize=5, linewidth=2.0, capsize=4,
                    color="tab:orange",
                    label="Episodic CV (5 seeds)")

    title = "Comparison of Few-Shot Evaluation Strategies"


    ax.set_title(title, fontsize=16, pad=8)
    ax.set_xlabel("Shots per class (k)")
    ax.set_ylabel("Accuracy")

    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.grid(axis="x", visible=False)

    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features", choices=["handcrafted", "signature"], required=True)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--out_csv", default="Results/tables/fewshot_compare.csv")
    parser.add_argument("--plot", action="store_true")  # genera plot tras acumular varios shots
    args = parser.parse_args()

    # load dataset
    df, X, y, feat_cols, y_col = load_Xy(args.csv, args.features)

    # 1) single episodio
    res_single = run_fewshot_once(
        X, y, k=args.shots, n_neighbors=args.n_neighbors,
        use_augment=args.augment, feat_cols=feat_cols, y_col=y_col
    )
    res_single.update({
        "mode": "single_support_query",
        "shots": args.shots,
        "features": args.features,
        "augment": int(args.augment),
        "acc_std": "",
        "f1w_std": "",
        "f1m_std": "",
        "seed": ""  
    })
    append_results(args.out_csv, res_single)

    # 2) 5 episodios (episodic CV)
    df_runs, summary = run_fewshot_cv(
        X, y, k=args.shots, n_neighbors=args.n_neighbors,
        use_augment=args.augment, feat_cols=feat_cols, y_col=y_col
    )
    summary_row = {
        "mode": "episodic_cv",
        "shots": args.shots,
        "features": args.features,
        "augment": int(args.augment),
        "acc": float(summary["acc_mean"]),
        "acc_std": float(summary["acc_std"]),
        "f1w": float(summary["f1w_mean"]),
        "f1w_std": float(summary["f1w_std"]),
        "f1m": float(summary["f1m_mean"]),
        "f1m_std": float(summary["f1m_std"]),
        "seed": "" 
    }
    append_results(args.out_csv, summary_row)

    # optional plot (requires several rows accumulated with different --shots)
    if args.plot:
        plot_name = f"Results/plots/fewshot_compare_{args.features}_{'aug' if args.augment else 'noaug'}.png"
        plot_fewshot_comparison(args.out_csv, output_path=plot_name,
                                features_filter=args.features, augment_filter=int(args.augment))
