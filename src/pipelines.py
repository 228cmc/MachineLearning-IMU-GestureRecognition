# pipelines.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from utils.augment import DataAugmenter
from models import RandomForestTrainer, FewShotKNN, save_model
from utils.utils import (
    load_Xy, select_k_per_class_indices,
    print_classification_metrics, save_confusion_matrix,
    log_result, episodic_cv,
)

from config import NOISE_REPS, MIXUP_REPS, KNN_NEIGHBORS_FEWSHOT, MODELS_DIR

# Basic splits (80/20 and few-shot)

def run_rf(csv_path: str, feature_mode: str, out_dir: str):
    _, X, y, _, _ = load_Xy(csv_path, feature_mode)

    """
    Train Random Forest on 80/20 split and report metrics.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type, 'handcrafted' or 'signature'.
    - out_dir (str): Output directory for artifacts.

    Returns:
    - None. Saves a model file in MODELS_DIR and writes metrics to disk.
    """
    rf = RandomForestTrainer()
    # standard 80/20 split with fixed seed
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.model.fit(Xtr, ytr)
    ypred = rf.model.predict(Xte)
    base = f"rf_baseline_{feature_mode}"
    acc, f1w, f1m = print_classification_metrics(yte, ypred, "RF baseline", base)
    save_confusion_matrix(yte, ypred, title=f"Confusion Matrix – F1 score: {f1w*100:.2f}%", filename=base)

    save_model(rf.model, f"RandomForest_{feature_mode}.pkl")

def run_rf_few(csv_path: str, feature_mode: str, out_dir: str, k: int = 5, seed: int = 42):

    """
    Train Random Forest in few-shot mode and report metrics.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - out_dir (str): Output directory.
    - k (int): Shots per class.
    - seed (int): Random seed for episode split.

    Returns:
    - tuple: (rows_df, summary_dict). Also saves CSVs to Results/tables/rf_few_cv_{feature_mode}.csv
      and Results/tables/rf_few_cv_{feature_mode}_seeds<min>_<max}.csv.
    """
    _, X, y, _, _ = load_Xy(csv_path, feature_mode)
    # build support/query split
    train_idx, test_idx = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    rf = RandomForestTrainer()
    rf.model.fit(Xtr, ytr)
    ypred = rf.model.predict(Xte)
    base = f"rf_few_k{k}_{feature_mode}"
    acc, f1w, f1m = print_classification_metrics(yte, ypred, f"RF few-shot (k={k})", base)
    save_confusion_matrix(yte, ypred, title=f"Confusion Matrix – F1 score: {f1w*100:.2f}%", filename=base)
  
    save_model(rf.model, f"RandomForest_few_k{k}_{feature_mode}.pkl")

def run_rf_aug(csv_path: str, feature_mode: str, out_dir: str,
               noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS,
               seed: int = 42):
    """
    Train Random Forest with augmentation on 80/20 split.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - out_dir (str): Output directory.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.
    - seed (int): RNG seed for augmentation.

    Returns:
    - None
    """    
    # Train Random Forest with data augmentation (noise + mixup)
    _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    # build augmented training block
    base_df = pd.DataFrame(Xtr, columns=feat_cols).assign(**{y_col: ytr})
    aug_df = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col, noise_reps, mixup_reps)
    X_aug, y_aug = aug_df[feat_cols].values, aug_df[y_col].values
    rf = RandomForestTrainer()
    rf.model.fit(X_aug, y_aug)
    ypred = rf.model.predict(Xte)

    base = f"rf_aug_{feature_mode}_noise{noise_reps}_mix{mixup_reps}"
    acc, f1w, f1m = print_classification_metrics(yte, ypred,
                                                 f"RF+Aug (noise={noise_reps}, mix={mixup_reps})", base)
    save_confusion_matrix(yte, ypred, title=f"Confusion Matrix – F1 score: {f1w*100:.2f}%", filename=base)
    save_model(rf.model, f"RandomForest_Augmented_{feature_mode}.pkl")

def run_rf_aug_few(csv_path: str, feature_mode: str, out_dir: str,
                   k: int, seed: int,
                   noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS):
    """
    Train Random Forest with augmentation in few-shot mode.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - out_dir (str): Output directory.
    - k (int): Shots per class.
    - seed (int): RNG seed for split and augmentation.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.

    Returns:
    - None
    """
    _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
    train_idx, test_idx = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    # augment only the training support set
    base_df = pd.DataFrame(Xtr, columns=feat_cols).assign(**{y_col: ytr})
    aug_df = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col, noise_reps, mixup_reps)
    X_aug, y_aug = aug_df[feat_cols].values, aug_df[y_col].values
    rf = RandomForestTrainer()
    rf.model.fit(X_aug, y_aug)
    ypred = rf.model.predict(Xte)
    base = f"rf_aug_few_k{k}_{feature_mode}_noise{noise_reps}_mix{mixup_reps}"
    acc, f1w, f1m = print_classification_metrics(yte, ypred,
                                                 f"RF+Aug few-shot (k={k}, noise={noise_reps}, mix={mixup_reps})", base)
    save_confusion_matrix(yte, ypred, title=f"Confusion Matrix – F1 score: {f1w*100:.2f}%", filename=base)
    save_model(rf.model, f"RandomForest_Augmented_few_k{k}_{feature_mode}.pkl")

def run_knn_few(csv_path: str, feature_mode: str, out_dir: str,
                k: int = 5, seed: int = 42, n_neighbors: int = KNN_NEIGHBORS_FEWSHOT):
    _, X, y, _, _ = load_Xy(csv_path, feature_mode)
    """
    Train KNN in few-shot mode and report metrics.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - out_dir (str): Output directory.
    - k (int): Shots per class.
    - seed (int): Random seed for split.
    - n_neighbors (int): KNN neighbors.

    Returns:
    - None
    """
    knn = FewShotKNN(n_neighbors=n_neighbors)
    Xtr, Xte, ytr, yte, ypred = knn.train_k_shot_and_predict_rest(X, y, shots_per_class=k, seed=seed)

    base = f"knn_few_k{k}_{feature_mode}"
    acc, f1w, f1m = print_classification_metrics(yte, ypred,
                                                 f"KNN few-shot (k={k}, n={n_neighbors})", base)
    save_confusion_matrix(yte, ypred, title=f"Confusion Matrix – F1 score: {f1w*100:.2f}%", filename=base)
    save_model(knn.model, f"KNN_few_k{k}_{feature_mode}.pkl")

def run_knn_full(csv_path: str, feature_mode: str, out_dir: str, n_neighbors: int = 5):
    _, X, y, _, _ = load_Xy(csv_path, feature_mode)
    """
    Train KNN on 80/20 split and report metrics.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - out_dir (str): Output directory.
    - n_neighbors (int): KNN neighbors.

    Returns:
    - None
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)
    base = f"knn_full_n{n_neighbors}_{feature_mode}"
    acc, f1w, f1m = print_classification_metrics(yte, ypred,
                                                 f"KNN full (n={n_neighbors})", base)
    save_confusion_matrix(yte, ypred, title=f"Confusion Matrix – F1 score: {f1w*100:.2f}%", filename=base)

    save_model(model, f"KNN_full_n{n_neighbors}_{feature_mode}.pkl")

def run_knn_augment(csv_path: str, feature_mode: str, out_dir: str,
                    noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS,
                    n_neighbors: int = 5, seed: int = 42):
    """
    Train KNN with augmentation on 80/20 split.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - out_dir (str): Output directory.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.
    - n_neighbors (int): KNN neighbors.
    - seed (int): RNG seed for augmentation.

    Returns:
    - None
    """
    _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    base_df = pd.DataFrame(Xtr, columns=feat_cols).assign(**{y_col: ytr})
    aug_df = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col, noise_reps, mixup_reps)
    X_aug, y_aug = aug_df[feat_cols].values, aug_df[y_col].values
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_aug, y_aug)
    ypred = model.predict(Xte)
    base = f"knn_aug_n{n_neighbors}_{feature_mode}_noise{noise_reps}_mix{mixup_reps}"
    acc, f1w, f1m = print_classification_metrics(yte, ypred,
                                                 f"KNN+Aug (n={n_neighbors}, noise={noise_reps}, mix={mixup_reps})", base)
    save_confusion_matrix(yte, ypred, title=f"Confusion Matrix – F1 score: {f1w*100:.2f}%", filename=base)

    save_model(model, f"KNN_Augmented_n{n_neighbors}_{feature_mode}.pkl")

def run_knn_few_aug(csv_path: str, feature_mode: str, k: int, seed: int,
                    n_neighbors: int = KNN_NEIGHBORS_FEWSHOT):
    """
    Train KNN with augmentation in few-shot mode.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seed (int): RNG seed for split and augmentation.
    - n_neighbors (int): KNN neighbors.

    Returns:
    - float: Accuracy obtained on the query set.
    """
    _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
    train_idx, test_idx = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    base_df = pd.DataFrame(Xtr, columns=feat_cols).assign(**{y_col: ytr})
    aug_df = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col, NOISE_REPS, MIXUP_REPS)
    X_aug, y_aug = aug_df[feat_cols].values, aug_df[y_col].values
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_aug, y_aug)
    ypred = model.predict(Xte)
    base = f"knn_few_aug_k{k}_{feature_mode}_noise{NOISE_REPS}_mix{MIXUP_REPS}"
    acc, f1w, f1m = print_classification_metrics(yte, ypred,
                                                 f"KNN few-shot+Aug (k={k}, n={n_neighbors}, noise={NOISE_REPS}, mix={MIXUP_REPS})", base)
    save_confusion_matrix(yte, ypred, title=f"Confusion Matrix – F1 score: {f1w*100:.2f}%", filename=base)

 
    log_result("HF" if feature_mode == "handcrafted" else "Sig", "KNN", "few+aug", k, acc, f1w, f1m)
    save_model(model, f"KNN_few_aug_k{k}_{feature_mode}.pkl")
    return acc

#  Few-shot sweep (for curves)

def run_fewshot_experiment(n_shots: int, seed: int, results: dict, hf_csv: str, sig_csv: str) -> dict:
    """
    Few-shot sweep without persisting models. Computes metrics for:
      - Handcrafted: RF few, RF few+Aug, KNN few, KNN few+Aug
      - Signature:   RF few, RF few+Aug, KNN few, KNN few+Aug
    Updates `results` and logs metrics; no model files are saved.
    """
    key = f"{n_shots}_shots"
    results[key] = {"HF_fewshot": {}, "HF_fewshot+Aug": {}, "Sig_fewshot": {}, "Sig_fewshot+Aug": {}}

    # Handcrafted 
    _, X_hf, y_hf, feat_cols_hf, y_col_hf = load_Xy(hf_csv, "handcrafted")
    tr_hf, te_hf = select_k_per_class_indices(y_hf, k=n_shots, seed=seed, strict=True)

    # RF few-shot (HF)
    rf_hf = RandomForestTrainer()
    rf_hf.model.fit(X_hf[tr_hf], y_hf[tr_hf])
    ypred = rf_hf.model.predict(X_hf[te_hf])
    acc, f1w, f1m = print_classification_metrics(y_hf[te_hf], ypred, f"HF RF few-shot (k={n_shots})", None)
    results[key]["HF_fewshot"]["RandomForest"] = {"accuracy": float(acc)}
    log_result("HF", "RandomForest", "few", n_shots, acc, f1w, f1m)

    # RF few-shot + Aug (HF)
    base_df_hf = pd.DataFrame(X_hf[tr_hf], columns=feat_cols_hf).assign(**{y_col_hf: y_hf[tr_hf]})
    aug_df_hf  = DataAugmenter(seed=seed).augment_dataframe(
        base_df_hf, feat_cols_hf, y_col_hf, NOISE_REPS, MIXUP_REPS
    )
    rf_hf_aug = RandomForestTrainer()
    rf_hf_aug.model.fit(aug_df_hf[feat_cols_hf].values, aug_df_hf[y_col_hf].values)
    ypred = rf_hf_aug.model.predict(X_hf[te_hf])
    acc, f1w, f1m = print_classification_metrics(y_hf[te_hf], ypred, f"HF RF few-shot+Aug (k={n_shots})", None)
    results[key]["HF_fewshot+Aug"]["RandomForest"] = {"accuracy": float(acc)}
    log_result("HF", "RandomForest", "few+aug", n_shots, acc, f1w, f1m)

    # KNN few-shot (HF)
    knn_hf = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS_FEWSHOT)
    knn_hf.fit(X_hf[tr_hf], y_hf[tr_hf])
    ypred = knn_hf.predict(X_hf[te_hf])
    acc, f1w, f1m = print_classification_metrics(y_hf[te_hf], ypred, f"HF KNN few-shot (k={n_shots})", None)
    results[key]["HF_fewshot"]["KNN"] = {"accuracy": float(acc)}
    log_result("HF", "KNN", "few", n_shots, acc, f1w, f1m)

    # KNN few-shot + Aug (HF) — inline to avoid saving models
    knn_hf_aug = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS_FEWSHOT)
    knn_hf_aug.fit(aug_df_hf[feat_cols_hf].values, aug_df_hf[y_col_hf].values)
    ypred = knn_hf_aug.predict(X_hf[te_hf])
    acc, f1w, f1m = print_classification_metrics(y_hf[te_hf], ypred, f"HF KNN few-shot+Aug (k={n_shots})", None)
    results[key]["HF_fewshot+Aug"]["KNN"] = {"accuracy": float(acc)}
    log_result("HF", "KNN", "few+aug", n_shots, acc, f1w, f1m)

    #  Signature
    _, X_sig, y_sig, feat_cols_sig, y_col_sig = load_Xy(sig_csv, "signature")
    tr_sig, te_sig = select_k_per_class_indices(y_sig, k=n_shots, seed=seed, strict=True)

    # RF few-shot (Sig)
    rf_sig = RandomForestTrainer()
    rf_sig.model.fit(X_sig[tr_sig], y_sig[tr_sig])
    ypred = rf_sig.model.predict(X_sig[te_sig])
    acc, f1w, f1m = print_classification_metrics(y_sig[te_sig], ypred, f"Sig RF few-shot (k={n_shots})", None)
    results[key]["Sig_fewshot"]["RandomForest"] = {"accuracy": float(acc)}
    log_result("Sig", "RandomForest", "few", n_shots, acc, f1w, f1m)

    # RF few-shot + Aug (Sig)
    base_df_sig = pd.DataFrame(X_sig[tr_sig], columns=feat_cols_sig).assign(**{y_col_sig: y_sig[tr_sig]})
    aug_df_sig  = DataAugmenter(seed=seed).augment_dataframe(
        base_df_sig, feat_cols_sig, y_col_sig, NOISE_REPS, MIXUP_REPS
    )
    rf_sig_aug = RandomForestTrainer()
    rf_sig_aug.model.fit(aug_df_sig[feat_cols_sig].values, aug_df_sig[y_col_sig].values)
    ypred = rf_sig_aug.model.predict(X_sig[te_sig])
    acc, f1w, f1m = print_classification_metrics(y_sig[te_sig], ypred, f"Sig RF few-shot+Aug (k={n_shots})", None)
    results[key]["Sig_fewshot+Aug"]["RandomForest"] = {"accuracy": float(acc)}
    log_result("Sig", "RandomForest", "few+aug", n_shots, acc, f1w, f1m)

    # KNN few-shot (Sig)
    knn_sig = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS_FEWSHOT)
    knn_sig.fit(X_sig[tr_sig], y_sig[tr_sig])
    ypred = knn_sig.predict(X_sig[te_sig])
    acc, f1w, f1m = print_classification_metrics(y_sig[te_sig], ypred, f"Sig KNN few-shot (k={n_shots})", None)
    results[key]["Sig_fewshot"]["KNN"] = {"accuracy": float(acc)}
    log_result("Sig", "KNN", "few", n_shots, acc, f1w, f1m)

    # KNN few-shot + Aug (Sig) — inline to avoid saving models
    knn_sig_aug = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS_FEWSHOT)
    knn_sig_aug.fit(aug_df_sig[feat_cols_sig].values, aug_df_sig[y_col_sig].values)
    ypred = knn_sig_aug.predict(X_sig[te_sig])
    acc, f1w, f1m = print_classification_metrics(y_sig[te_sig], ypred, f"Sig KNN few-shot+Aug (k={n_shots})", None)
    results[key]["Sig_fewshot+Aug"]["KNN"] = {"accuracy": float(acc)}
    log_result("Sig", "KNN", "few+aug", n_shots, acc, f1w, f1m)

    return results




#  Episodic CV (internal helpers + wrappers)

def _knn_few_once(csv_path: str, feature_mode: str, k: int, seed: int, n_neighbors: int):
    """
    Single KNN few-shot episode used by CV wrappers.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seed (int): RNG seed.
    - n_neighbors (int): KNN neighbors.

    Returns:
    - tuple: (acc, f1w, f1m) from print_classification_metrics.
    """
    _, X, y, _, _ = load_Xy(csv_path, feature_mode)
    tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X[tr], y[tr])
    ypred = knn.predict(X[te])
    return print_classification_metrics(y[te], ypred, f"KNN few-shot (k={k}, seed={seed}, {feature_mode})", None)

def run_knn_few_cv(csv_path: str, feature_mode: str, k: int, seeds, n_neighbors: int):
    """
    Run episodic CV for KNN few-shot.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seeds (iterable): Seeds for episodes.
    - n_neighbors (int): KNN neighbors.

    Returns:
    - tuple: (rows_df, summary_dict). Also saves CSVs to Results/tables/knn_few_cv_{feature_mode}.csv
      and Results/tables/knn_few_cv_{feature_mode}_seeds<min>_<max>.csv.
    """
    return episodic_cv(_knn_few_once, seeds, csv_path=csv_path, feature_mode=feature_mode, k=k, n_neighbors=n_neighbors)

def _rf_few_once(csv_path: str, feature_mode: str, k: int, seed: int):
    """
    Single RF few-shot episode used by CV wrappers.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seed (int): RNG seed.

    Returns:
    - tuple: (acc, f1w, f1m) from print_classification_metrics.
    """
    _, X, y, _, _ = load_Xy(csv_path, feature_mode)
    tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    ypred = rf.predict(X[te])
    return print_classification_metrics(y[te], ypred, f"RF few-shot (k={k}, seed={seed}, {feature_mode})", None)

def run_rf_few_cv(csv_path: str, feature_mode: str, k: int, seeds):
    """
    Run episodic CV for RF few-shot.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seeds (iterable): Seeds for episodes.

    Returns:
    - tuple: (rows_df, summary_dict). Also saves CSVs to Results/tables/knn_few_aug_cv_{feature_mode}.csv
      and Results/tables/knn_few_aug_cv_{feature_mode}_seeds<min>_<max>.csv.
        """
    return episodic_cv(_rf_few_once, seeds, csv_path=csv_path, feature_mode=feature_mode, k=k)

def _rf_aug_few_once(csv_path: str, feature_mode: str, k: int, seed: int,
                     noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS):
    _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
    """
    Single RF few-shot with augmentation episode (for CV wrappers).

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seed (int): RNG seed.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.

    Returns:
    - tuple: (rows_df, summary_dict). Also saves CSVs to Results/tables/rf_few_cv_{feature_mode}.csv
    and Results/tables/rf_few_cv_{feature_mode}_seeds<min>_<max>.csv.
    """
    tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    base_df = pd.DataFrame(X[tr], columns=feat_cols).assign(**{y_col: y[tr]})
    aug_df  = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col,
                                                         noise_reps=noise_reps, mixup_reps=mixup_reps)
    rf = RandomForestTrainer()
    rf.model.fit(aug_df[feat_cols].values, aug_df[y_col].values)
    ypred = rf.model.predict(X[te])
    return print_classification_metrics(y[te], ypred, f"RF+Aug few-shot (k={k}, seed={seed}, {feature_mode})", None)

def run_rf_aug_few_cv(csv_path: str, feature_mode: str, k: int, seeds,
                      noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS):
    """
    Run episodic CV for RF few-shot with augmentation.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seeds (iterable): Seeds for episodes.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.

    Returns:
    - tuple: (rows_df, summary_dict). Also saves CSVs to Results/tables/rf_aug_few_cv_{feature_mode}.csv
      and Results/tables/rf_aug_few_cv_{feature_mode}_seeds<min>_<max>.csv.
    """
    return episodic_cv(_rf_aug_few_once, seeds,
                       csv_path=csv_path, feature_mode=feature_mode, k=k,
                       noise_reps=noise_reps, mixup_reps=mixup_reps)

def _knn_few_aug_once(csv_path: str, feature_mode: str, k: int, seed: int,
                      n_neighbors: int = KNN_NEIGHBORS_FEWSHOT,
                      noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS):
    """
    Single KNN few-shot with augmentation episode (for CV wrappers).

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seed (int): RNG seed.
    - n_neighbors (int): KNN neighbors.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.

    Returns:
    - tuple: (acc, f1w, f1m) from print_classification_metrics.
    """
    _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
    tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    base_df = pd.DataFrame(X[tr], columns=feat_cols).assign(**{y_col: y[tr]})
    aug_df  = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col,
                                                         noise_reps=noise_reps, mixup_reps=mixup_reps)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(aug_df[feat_cols].values, aug_df[y_col].values)
    ypred = knn.predict(X[te])
    return print_classification_metrics(y[te], ypred,
        f"KNN+Aug few-shot (k={k}, n={n_neighbors}, seed={seed}, {feature_mode})", None)

def run_knn_few_aug_cv(csv_path: str, feature_mode: str, k: int, seeds,
                       n_neighbors: int = KNN_NEIGHBORS_FEWSHOT,
                       noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS):
    """
    Run episodic CV for KNN few-shot with augmentation.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seeds (iterable): Seeds for episodes.
    - n_neighbors (int): KNN neighbors.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.

    Returns:
    - tuple: (rows_df, summary_dict). Also saves CSVs to Results/tables/knn_few_aug_cv_{feature_mode}.csv
      and Results/tables/knn_few_aug_cv_{feature_mode}_seeds<min>_<max>.csv.
    """
    return episodic_cv(_knn_few_aug_once, seeds,
                       csv_path=csv_path, feature_mode=feature_mode, k=k,
                       n_neighbors=n_neighbors, noise_reps=noise_reps, mixup_reps=mixup_reps)



def rf_few_pred_once(csv_path: str, feature_mode: str, k: int, seed: int):
    """
    Produce predictions for one RF few-shot episode.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seed (int): RNG seed.

    Returns:
    - tuple: (y_true, y_pred) for the query set.
    """
    _, X, y, _, _ = load_Xy(csv_path, feature_mode)
    tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    rf = RandomForestTrainer(); rf.model.fit(X[tr], y[tr])
    ypred = rf.model.predict(X[te])
    return y[te], ypred

def rf_aug_few_pred_once(csv_path: str, feature_mode: str, k: int, seed: int,
                         noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS):
    """
    Produce predictions for one RF few-shot+Aug episode.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seed (int): RNG seed.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.

    Returns:
    - tuple: (y_true, y_pred) for the query set.
    """
    _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
    tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    base_df = pd.DataFrame(X[tr], columns=feat_cols).assign(**{y_col: y[tr]})
    aug_df = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col, noise_reps, mixup_reps)
    rf = RandomForestTrainer(); rf.model.fit(aug_df[feat_cols].values, aug_df[y_col].values)
    ypred = rf.model.predict(X[te])
    return y[te], ypred

def knn_few_pred_once(csv_path: str, feature_mode: str, k: int, seed: int, n_neighbors: int):
    """
    Produce predictions for one KNN few-shot episode.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seed (int): RNG seed.
    - n_neighbors (int): KNN neighbors.

    Returns:
    - tuple: (y_true, y_pred) for the query set.
    """    
    _, X, y, _, _ = load_Xy(csv_path, feature_mode)
    tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    nnb = max(1, min(n_neighbors, len(tr)))
    model = KNeighborsClassifier(n_neighbors=nnb).fit(X[tr], y[tr])
    ypred = model.predict(X[te])
    return y[te], ypred

def knn_few_aug_pred_once(csv_path: str, feature_mode: str, k: int, seed: int, n_neighbors: int,
                          noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS):
    """
    Produce predictions for one KNN few-shot+Aug episode.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seed (int): RNG seed.
    - n_neighbors (int): KNN neighbors.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.

    Returns:
    - tuple: (y_true, y_pred) for the query set.
    """
    _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
    tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
    base_df = pd.DataFrame(X[tr], columns=feat_cols).assign(**{y_col: y[tr]})
    aug_df = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col, noise_reps, mixup_reps)
    nnb = max(1, min(n_neighbors, len(aug_df)))
    model = KNeighborsClassifier(n_neighbors=nnb).fit(aug_df[feat_cols].values, aug_df[y_col].values)
    ypred = model.predict(X[te])
    return y[te], ypred



# Episodic CV( external wrappers)

def run_rf_few_cv_episodes(csv_path: str, feature_mode: str, k: int, seeds: list[int]):
    """
    Wrapper to run RF few-shot episodic CV across seeds.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seeds (list[int]): Seeds for episodes.

    Returns:
    - tuple: (rows_df, summary_dict).
    """
    def _once(csv_path, feature_mode, k, seed):
        _, X, y, _, _ = load_Xy(csv_path, feature_mode)
        tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
        rf = RandomForestTrainer()
        rf.model.fit(X[tr], y[tr])
        ypred = rf.model.predict(X[te])
        return print_classification_metrics(y[te], ypred, f"RF few-shot (k={k}, {feature_mode}, seed={seed})", None)

    rows, summary = episodic_cv(
        _once, seeds,
        csv_path=csv_path, feature_mode=feature_mode,
        model=f"rf_few_cv_{feature_mode}", k=k,
    )
    return rows, summary

def run_rf_aug_few_cv_episodes(csv_path: str, feature_mode: str, k: int, seeds: list[int],
                               noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS):
    """
    Wrapper to run RF few-shot+Aug episodic CV across seeds.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - seeds (list[int]): Seeds for episodes.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.

    Returns:
    - tuple: (rows_df, summary_dict).
    """
    def _once(csv_path, feature_mode, k, seed, noise_reps, mixup_reps):
        _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
        tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
        base_df = pd.DataFrame(X[tr], columns=feat_cols).assign(**{y_col: y[tr]})
        aug_df = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col,
                                                            noise_reps=noise_reps, mixup_reps=mixup_reps)
        rf = RandomForestTrainer()
        rf.model.fit(aug_df[feat_cols].values, aug_df[y_col].values)
        ypred = rf.model.predict(X[te])
        return print_classification_metrics(y[te], ypred, f"RF+Aug few-shot (k={k}, {feature_mode}, seed={seed})", None)

    rows, summary = episodic_cv(
        _once, seeds,
        csv_path=csv_path, feature_mode=feature_mode,
        model=f"rf_aug_few_cv_{feature_mode}",
        k=k, noise_reps=noise_reps, mixup_reps=mixup_reps,
    )
    return rows, summary

def run_knn_few_cv_episodes(csv_path: str, feature_mode: str, k: int,
                            n_neighbors: int, seeds: list[int]):
    """
    Wrapper to run KNN few-shot episodic CV across seeds.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - n_neighbors (int): KNN neighbors.
    - seeds (list[int]): Seeds for episodes.

    Returns:
    - tuple: (rows_df, summary_dict).
    """
    def _once(csv_path, feature_mode, k, seed, n_neighbors):
        _, X, y, _, _ = load_Xy(csv_path, feature_mode)
        tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
        nnb = max(1, min(n_neighbors, len(tr)))
        model = KNeighborsClassifier(n_neighbors=nnb)
        model.fit(X[tr], y[tr])
        ypred = model.predict(X[te])
        return print_classification_metrics(y[te], ypred,
                    f"KNN few-shot (k={k}, n={nnb}, {feature_mode}, seed={seed})", None)

    rows, summary = episodic_cv(
        _once, seeds,
        csv_path=csv_path, feature_mode=feature_mode,
        model=f"knn_few_cv_{feature_mode}",
        k=k, n_neighbors=n_neighbors,
    )
    return rows, summary

def run_knn_few_aug_cv_episodes(csv_path: str, feature_mode: str, k: int,
                                n_neighbors: int, seeds: list[int],
                                noise_reps: int = NOISE_REPS, mixup_reps: int = MIXUP_REPS):
    
    """
    Wrapper to run KNN few-shot+Aug episodic CV across seeds.

    Parameters:
    - csv_path (str): Input CSV path.
    - feature_mode (str): Feature type.
    - k (int): Shots per class.
    - n_neighbors (int): KNN neighbors.
    - seeds (list[int]): Seeds for episodes.
    - noise_reps (int): Gaussian-noise replicas.
    - mixup_reps (int): Mixup repetitions.

    Returns:
    - tuple: (rows_df, summary_dict).
    """
    def _once(csv_path, feature_mode, k, seed, n_neighbors, noise_reps, mixup_reps):
        _, X, y, feat_cols, y_col = load_Xy(csv_path, feature_mode)
        tr, te = select_k_per_class_indices(y, k=k, seed=seed, strict=True)
        base_df = pd.DataFrame(X[tr], columns=feat_cols).assign(**{y_col: y[tr]})
        aug_df = DataAugmenter(seed=seed).augment_dataframe(base_df, feat_cols, y_col,
                                                            noise_reps=noise_reps, mixup_reps=mixup_reps)
        nnb = max(1, min(n_neighbors, len(aug_df)))
        model = KNeighborsClassifier(n_neighbors=nnb)
        model.fit(aug_df[feat_cols].values, aug_df[y_col].values)
        ypred = model.predict(X[te])
        return print_classification_metrics(y[te], ypred,
                    f"KNN+Aug few-shot (k={k}, n={nnb}, {feature_mode}, seed={seed})", None)

    rows, summary = episodic_cv(
        _once, seeds,
        csv_path=csv_path, feature_mode=feature_mode,
        model=f"knn_few_aug_cv_{feature_mode}",
        k=k, n_neighbors=n_neighbors,
        noise_reps=noise_reps, mixup_reps=mixup_reps,
    )
    return rows, summary
