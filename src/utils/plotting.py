# plotting.py (completo)

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# plotting.py
from config import FIG_FONT, TABLES_DIR, PLOTS_DIR
plt.rcParams["font.family"] = FIG_FONT

from utils.utils import (
    load_dataframe, load_Xy,
    episodic_collect_predictions, save_confusion_matrix_counts,
    NOISE_REPS, MIXUP_REPS
)
from models import RandomForestTrainer
from utils.augment import DataAugmenter
from pipelines import (
    rf_few_pred_once, rf_aug_few_pred_once,
    knn_few_pred_once, knn_few_aug_pred_once,
)



def plot_fewshot_results(results: dict, output_dir: str = "Results/plots"):
    """
    Plot few-shot accuracies for RandomForest and KNN with handcrafted and signature features.

    Parameters:
    - results (dict): Nested metrics dictionary produced by the sweep.
    - output_dir (str): Directory to save the figure.

    Returns:
    - None. Saves a PNG to {output_dir}/fewshot_rf_knn_handcrafted_signature.png.
    """
    shots = sorted([int(k.split("_")[0]) for k in results.keys()])

    hf_rf = {"fewshot": [], "aug": []}
    hf_knn = {"fewshot": [], "aug": []}
    sig_rf = {"fewshot": [], "aug": []}
    sig_knn = {"fewshot": [], "aug": []}

    for n in shots:
        res_local = results[f"{n}_shots"]
        hf_rf["fewshot"].append(res_local["HF_fewshot"]["RandomForest"]["accuracy"])
        hf_rf["aug"].append(res_local["HF_fewshot+Aug"]["RandomForest"]["accuracy"])
        hf_knn["fewshot"].append(res_local["HF_fewshot"]["KNN"]["accuracy"])
        hf_knn["aug"].append(res_local["HF_fewshot+Aug"]["KNN"]["accuracy"])
        sig_rf["fewshot"].append(res_local["Sig_fewshot"]["RandomForest"]["accuracy"])
        sig_rf["aug"].append(res_local["Sig_fewshot+Aug"]["RandomForest"]["accuracy"])
        sig_knn["fewshot"].append(res_local["Sig_fewshot"]["KNN"]["accuracy"])
        sig_knn["aug"].append(res_local["Sig_fewshot+Aug"]["KNN"]["accuracy"])

    plt.figure(figsize=(12, 10))
    color_few = "#e6b800"  
    color_aug = "#d62728"  

    # Subplot 1
    plt.subplot(2, 2, 1)
    plt.plot(shots, hf_rf["fewshot"], marker="o", color=color_few, linewidth=2, label="Few-shot")
    plt.plot(shots, hf_rf["aug"], marker="s", color=color_aug, linewidth=2, label="Few-shot + Aug")
    plt.title("Handcrafted - RandomForest", fontsize=22)
    plt.xlabel("Shots per class", fontsize=18); plt.ylabel("Accuracy", fontsize=18)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16); plt.legend(fontsize=16)

    # Subplot 2
    plt.subplot(2, 2, 2)
    plt.plot(shots, hf_knn["fewshot"], marker="o", color=color_few, linewidth=2, label="Few-shot")
    plt.plot(shots, hf_knn["aug"], marker="s", color=color_aug, linewidth=2, label="Few-shot + Aug")
    plt.title("Handcrafted - KNN", fontsize=22)
    plt.xlabel("Shots per class", fontsize=18); plt.ylabel("Accuracy", fontsize=18)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16); plt.legend(fontsize=16)

    # Subplot 3
    plt.subplot(2, 2, 3)
    plt.plot(shots, sig_rf["fewshot"], marker="o", color=color_few, linewidth=2, label="Few-shot")
    plt.plot(shots, sig_rf["aug"], marker="s", color=color_aug, linewidth=2, label="Few-shot + Aug")
    plt.title("Signature - RandomForest", fontsize=22)
    plt.xlabel("Shots per class", fontsize=18); plt.ylabel("Accuracy", fontsize=18)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16); plt.legend(fontsize=16)

    # Subplot 4
    plt.subplot(2, 2, 4)
    plt.plot(shots, sig_knn["fewshot"], marker="o", color=color_few, linewidth=2, label="Few-shot")
    plt.plot(shots, sig_knn["aug"], marker="s", color=color_aug, linewidth=2, label="Few-shot + Aug")
    plt.title("Signature - KNN", fontsize=22)
    plt.xlabel("Shots per class", fontsize=18); plt.ylabel("Accuracy", fontsize=18)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16); plt.legend(fontsize=16)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fewshot_rf_knn_handcrafted_signature.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_pca_splits(X_hf_train, y_hf_train, X_hf_test, y_hf_test,
                    X_sig_train, y_sig_train, X_sig_test, y_sig_test,
                    output_dir="Results/plots"):
    """
    Plot 2D PCA projections for train and test splits of handcrafted and signature features.

    Parameters:
    - X_hf_train, y_hf_train, X_hf_test, y_hf_test: Handcrafted train/test arrays.
    - X_sig_train, y_sig_train, X_sig_test, y_sig_test: Signature train/test arrays.
    - output_dir (str): Directory to save the figure.

    Returns:
    - None
    """
    import matplotlib.cm as cm

    FS_TITLE = 28
    FS_LABEL = 26
    FS_TICK  = 22
    FS_LEG   = 20

    classes_all = sorted(set(y_hf_train) | set(y_hf_test) | set(y_sig_train) | set(y_sig_test))
    cmap = cm.get_cmap("tab10", len(classes_all))
    color_map = {cls: cmap(i) for i, cls in enumerate(classes_all)}

    pca_hf = PCA(n_components=2)
    X_hf_train_pca = pca_hf.fit_transform(X_hf_train)
    X_hf_test_pca  = pca_hf.transform(X_hf_test)

    pca_sig = PCA(n_components=2)
    X_sig_train_pca = pca_sig.fit_transform(X_sig_train)
    X_sig_test_pca  = pca_sig.transform(X_sig_test)

    fig = plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(X_hf_train_pca[:, 0], X_hf_train_pca[:, 1],
                c=[color_map[c] for c in y_hf_train], alpha=0.6)
    ax1.set_title('Handcrafted Features Train', fontsize=FS_TITLE)
    ax1.set_xlabel('PC1', fontsize=FS_LABEL); ax1.set_ylabel('PC2', fontsize=FS_LABEL)
    ax1.tick_params(axis='both', labelsize=FS_TICK)

    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(X_hf_test_pca[:, 0], X_hf_test_pca[:, 1],
                c=[color_map[c] for c in y_hf_test], alpha=0.6)
    ax2.set_title('Handcrafted Features Test', fontsize=FS_TITLE)
    ax2.set_xlabel('PC1', fontsize=FS_LABEL); ax2.set_ylabel('PC2', fontsize=FS_LABEL)
    ax2.tick_params(axis='both', labelsize=FS_TICK)

    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(X_sig_train_pca[:, 0], X_sig_train_pca[:, 1],
                c=[color_map[c] for c in y_sig_train], alpha=0.6)
    ax3.set_title('Signature Features Train', fontsize=FS_TITLE)
    ax3.set_xlabel('PC1', fontsize=FS_LABEL); ax3.set_ylabel('PC2', fontsize=FS_LABEL)
    ax3.tick_params(axis='both', labelsize=FS_TICK)

    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(X_sig_test_pca[:, 0], X_sig_test_pca[:, 1],
                c=[color_map[c] for c in y_sig_test], alpha=0.6)
    ax4.set_title('Signature Features Test', fontsize=FS_TITLE)
    ax4.set_xlabel('PC1', fontsize=FS_LABEL); ax4.set_ylabel('PC2', fontsize=FS_LABEL)
    ax4.tick_params(axis='both', labelsize=FS_TICK)

    handles = [Line2D([0], [0], marker='o', linestyle='', color='w',
                      markerfacecolor=color_map[cls], markersize=14,
                      label=str(cls).replace("_", " "))
               for cls in classes_all]
    fig.legend(handles=handles, title="Classes",
               fontsize=FS_LEG, title_fontsize=FS_LEG,
               loc="lower center", ncol=len(classes_all),
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pca_features.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_class_distribution(counts_handcrafted: dict, counts_signature: dict,
                            output_dir="Results/plots"):
    """
    Plot side-by-side bar chart of class counts for handcrafted and signature features.

    Parameters:
    - counts_handcrafted (dict): Class counts for handcrafted features.
    - counts_signature (dict): Class counts for signature features.
    - output_dir (str): Directory to save the figure.

    Returns:
    - None. Saves a PNG to {output_dir}/class_distribution_hf_vs_sig.png.
    """
    classes = list(counts_handcrafted.keys())
    classes_clean = [c.replace("_", " ") for c in classes]
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, list(counts_handcrafted.values()), width,
           label="Handcrafted", color="#5E2B97", alpha=0.9)
    ax.bar(x + width/2, list(counts_signature.values()), width,
           label="Signature", color="#4CAF50", alpha=0.9)

    ax.set_title("Class distribution — Handcrafted vs Signature", fontsize=18)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(classes_clean, rotation=90, fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(0, max(max(counts_handcrafted.values()), max(counts_signature.values())) + 5)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.7)
    ax.set_ylim(0, 30)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "class_distribution_hf_vs_sig.png")
    plt.savefig(out_path, dpi=300)
    plt.close()



def counts_from_csv(csv_path: str) -> dict:
    """
    Count examples per class from a CSV, optionally filtering Label == 1.

    Parameters:
    - csv_path (str): Path to input CSV.

    Returns:
    - dict: Mapping class name to count.
    """
    df = load_dataframe(csv_path)
    if "Label" in df.columns:
        df = df[df["Label"] == 1]
    if "Exercise_Type" not in df.columns:
        raise ValueError(f"{csv_path} no tiene columna 'Exercise_Type'")
    return df["Exercise_Type"].value_counts().to_dict()





def run_pca_plot(seed: int, hf_csv: str, sig_csv: str) -> None:
    """
    Load datasets, split 80/20 with seed, and call plot_pca_splits.

    Parameters:
    - seed (int): Random seed for the splits.
    - hf_csv (str): Handcrafted CSV path.
    - sig_csv (str): Signature CSV path.

    Returns:
    - None
    """
    _, X_hf, y_hf, _, _ = load_Xy(hf_csv, "handcrafted")
    _, X_sig, y_sig, _, _ = load_Xy(sig_csv, "signature")
    X_hf_tr, X_hf_te, y_hf_tr, y_hf_te = train_test_split(X_hf, y_hf, test_size=0.2, random_state=seed)
    X_sig_tr, X_sig_te, y_sig_tr, y_sig_te = train_test_split(X_sig, y_sig, test_size=0.2, random_state=seed)
    plot_pca_splits(X_hf_tr, y_hf_tr, X_hf_te, y_hf_te, X_sig_tr, y_sig_tr, X_sig_te, y_sig_te)

from config import DEFAULT_SEED

def run_exploratory_plots(seeds: list[int], hf_csv: str, sig_csv: str) -> None:
    """
    Generate class distribution and PCA plots using a fixed split seed.

    Parameters:
    - seeds (list[int]): Seed list; the first one is used for the split.
    - hf_csv (str): Handcrafted CSV path.
    - sig_csv (str): Signature CSV path.

    Returns:
    - None
    """
    rs = seeds[0] if seeds else DEFAULT_SEED  # una sola semilla para el split

    # Distribution
    counts_hf = counts_from_csv(hf_csv)
    counts_sig = counts_from_csv(sig_csv)
    clases = sorted(set(counts_hf) | set(counts_sig))
    plot_class_distribution(
        {c: counts_hf.get(c, 0) for c in clases},
        {c: counts_sig.get(c, 0) for c in clases},
    )

    # 2) PCA (reproducible split with rs)
    _, X_hf, y_hf, _, _ = load_Xy(hf_csv, "handcrafted")
    _, X_sig, y_sig, _, _ = load_Xy(sig_csv, "signature")
    X_hf_tr, X_hf_te, y_hf_tr, y_hf_te = train_test_split(X_hf, y_hf, test_size=0.2, random_state=rs)
    X_sig_tr, X_sig_te, y_sig_tr, y_sig_te = train_test_split(X_sig, y_sig, test_size=0.2, random_state=rs)
    plot_pca_splits(X_hf_tr, y_hf_tr, X_hf_te, y_hf_te, X_sig_tr, y_sig_tr, X_sig_te, y_sig_te)


def run_confusion_matrix_agg(k: int, n_neighbors: int, seeds: list[int], hf_csv: str, sig_csv: str) -> None:
    """
    Aggregate episodic predictions and save confusion matrices for RF and KNN variants.

    Parameters Handcrafted CSV path.
    - sig_csv (str): Signature CSV path.

    Returns:
    - None:
    - k (int): Shots per class.
    - n_neighbors (int): KNN neighbors.
    - seeds (list[int]): Episode seeds.
    - hf_csv (str):
    """
    for mode, csv in [("handcrafted", hf_csv), ("signature", sig_csv)]:
        y_all, y_pred_all, _ = episodic_collect_predictions(
            rf_few_pred_once, seeds, csv_path=csv, feature_mode=mode, k=k
        )
        labels = np.unique(y_all)
        save_confusion_matrix_counts(y_all, y_pred_all, labels,
            model_name=f"RF few-shot (k={k})",
            filename=f"rf_few_k{k}_{mode}_agg")

        y_all, y_pred_all, _ = episodic_collect_predictions(
            rf_aug_few_pred_once, seeds, csv_path=csv, feature_mode=mode, k=k
        )
        save_confusion_matrix_counts(y_all, y_pred_all, labels,
            model_name=f"RF+Aug few-shot (k={k})",
            filename=f"rf_aug_few_k{k}_{mode}_agg")

        y_all, y_pred_all, _ = episodic_collect_predictions(
            knn_few_pred_once, seeds, csv_path=csv, feature_mode=mode, k=k, n_neighbors=n_neighbors
        )
        save_confusion_matrix_counts(y_all, y_pred_all, labels,
            model_name=f"KNN few-shot (k={k}, n={n_neighbors})",
            filename=f"knn_few_k{k}_{mode}_agg")

        y_all, y_pred_all, _ = episodic_collect_predictions(
            knn_few_aug_pred_once, seeds, csv_path=csv, feature_mode=mode, k=k, n_neighbors=n_neighbors
        )
        save_confusion_matrix_counts(y_all, y_pred_all, labels,
            model_name=f"KNN+Aug few-shot (k={k}, n={n_neighbors})",
            filename=f"knn_few_aug_k{k}_{mode}_agg")





def plot_knn_few_boxplot(
   tables_dir: str = TABLES_DIR,
    output_dir: str = "Results/plots",
    out_name: str = "knn_few_boxplot_seeds0_99.png"
) -> None:
    """
    Plot boxplots comparing KNN few-shot and few-shot+Aug for handcrafted vs signature across seeds.

    Parameters:
    - tables_dir (str): Directory containing *_seeds0_99.csv result tables.
    - output_dir (str): Directory to save the figure.
    - out_name (str): Output filename.

    Returns:
    - None. Saves a PNG to {output_dir}/{out_name}.
    """
    ...
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    files = {
        "hf_few":      "knn_few_cv_handcrafted_seeds0_99.csv",
        "sig_few":     "knn_few_cv_signature_seeds0_99.csv",
        "hf_few_aug":  "knn_few_aug_cv_handcrafted_seeds0_99.csv",
        "sig_few_aug": "knn_few_aug_cv_signature_seeds0_99.csv",
    }
    
    for key, fname in files.items():
        path = os.path.join(tables_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Falta {fname} en {tables_dir}")

    def _load_acc(path):
        df = pd.read_csv(path)
        if "acc" not in df.columns:
            raise ValueError(f"{os.path.basename(path)} no tiene columna 'acc'.")
        return df["acc"].dropna().values

    acc_hf_few      = _load_acc(os.path.join(tables_dir, files["hf_few"]))
    acc_sig_few     = _load_acc(os.path.join(tables_dir, files["sig_few"]))
    acc_hf_few_aug  = _load_acc(os.path.join(tables_dir, files["hf_few_aug"]))
    acc_sig_few_aug = _load_acc(os.path.join(tables_dir, files["sig_few_aug"]))

    # Colors
    color_hf  = "#5E2B97"  
    color_sig = "#4CAF50"  

    plt.figure(figsize=(10, 7))
    positions = [1, 2, 4, 5]  # par 1: few, par 2: few+aug
    data   = [acc_hf_few, acc_sig_few, acc_hf_few_aug, acc_sig_few_aug]
    colors = [color_hf,   color_sig,   color_hf,       color_sig]

    # Boxplots
    for d, pos, col in zip(data, positions, colors):
        bp = plt.boxplot(
            d, positions=[pos], widths=0.8, patch_artist=True,
            showfliers=True, whis=1.5
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(col); patch.set_alpha(0.65); patch.set_edgecolor("#333333")
        for med in bp["medians"]:
            med.set_color("#111111"); med.set_linewidth(2)

    # axis and titles
    plt.xticks([1.5, 4.5], ["KNN Few", "KNN Few+Aug"], fontsize=14)
    plt.ylabel("Accuracy", fontsize=16)
    plt.yticks(fontsize=12)
    plt.title("Comparison of KNN Few-shot Accuracies (seeds 0..99)", fontsize=18)
    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(facecolor=color_hf, edgecolor="#333333", label="Handcrafted", alpha=0.65),
        Patch(facecolor=color_sig, edgecolor="#333333", label="Signature",   alpha=0.65),
    ], title="Feature Set", loc="upper left", fontsize=12, title_fontsize=12)

    plt.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
