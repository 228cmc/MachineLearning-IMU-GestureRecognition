# main.py
import os
import argparse
import random
import numpy as np
from typing import Dict, Any

import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ...\Gesture
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from pipelines import (
    run_rf, run_rf_few, run_rf_aug, run_rf_aug_few,
    run_knn_few, run_knn_full, run_knn_augment, run_knn_few_aug,
    run_fewshot_experiment,
    run_rf_few_cv_episodes, run_rf_aug_few_cv_episodes,
    run_knn_few_cv_episodes, run_knn_few_aug_cv_episodes,
)
from utils.utils import save_results_table
from utils.plotting import (
    plot_fewshot_results,
    run_exploratory_plots,
    run_confusion_matrix_agg,
    plot_knn_few_boxplot,
)

from config import (
    DATA_PATHS,
    RESULTS_DIR as OUTPUT_DIR,
    DEFAULT_SHOTS, DEFAULT_N_NEIGHBORS, DEFAULT_SEED,
)

def _parse_seeds_arg(seeds_str: str) -> list[int]:
    """
    Parse a seeds string into a list of integers.
    """
    s = (seeds_str or "").strip()
    if s in {"0..99", "0:100", "100"}:
        return list(range(100))
    return [int(x) for x in s.split(",") if x.strip()]

def parse_args():
    """
    Parse command-line arguments for running different training scenarios.

    Parameters (command-line flags):
    - train (bool): Run all training pipelines in batch mode.
    - train_cv (bool): Run all CV training pipelines in batch mode.
    - scenario (str): Select one of the different scenarios to run:
    
        * rf: Train Random Forest on full dataset (80/20 split).
        * rf_few: Random Forest in few-shot mode.
        * rf_aug: Random Forest with data augmentation.
        * rf_aug_few: Random Forest with augmentation in few-shot mode.
        * knn_few: KNN in few-shot mode.
        * knn_full: KNN on full dataset (80/20 split).
        * knn_aug: KNN with data augmentation.
        * knn_few_aug: KNN with augmentation in few-shot mode.
        * fewshot_sweep: Run few-shot experiments varying k.
        * exploratory_plots: Generate exploratory data plots.
        * confusion_matrix: Compute and plot confusion matrix.
        * boxplot_knn: Plot KNN few-shot boxplot.
        * rf_few_cv: Random Forest few-shot with episodic CV.
        * rf_aug_few_cv: Random Forest few-shot with augmentation and episodic CV.
        * knn_few_cv: KNN few-shot with episodic CV.
        * knn_few_aug_cv: KNN few-shot with augmentation and episodic CV.

            - csv (str): Input CSV file path.
    - features (str): Feature type, signature or handcrafted.
    - output_dir (str): Directory to store results.
    - shots_per_class (int): Number of shots per class for few-shot training.
    - seed (int): Random seed.
    - seeds (str): List of seeds or shorthand like "0..99".
    - hf_csv (str): Path to handcrafted CSV.
    - sig_csv (str): Path to signature CSV.
    - k_min (int): Minimum k value for few-shot sweep.
    - k_max (int): Maximum k value for few-shot sweep.
    - n_neighbors (int): Number of neighbors for KNN.
    - csv_box (str): Path to CSV for KNN boxplot.

    Returns:
    - argparse.Namespace: Parsed arguments object.
    """
    p = argparse.ArgumentParser(description="IMU training pipelines.")

    # batch modes
    p.add_argument("--train", action="store_true")
    p.add_argument("--train_cv", action="store_true")

    # scenario mode
    p.add_argument("--scenario", choices=[
        "rf", "rf_few", "rf_aug", "rf_aug_few",
        "knn_few", "knn_full", "knn_aug", "knn_few_aug",
        "fewshot_sweep",
        "exploratory_plots",
        "confusion_matrix",
        "boxplot_knn",
        "rf_few_cv", "rf_aug_few_cv", "knn_few_cv", "knn_few_aug_cv",
    ])

    # common options
    p.add_argument("--csv")
    p.add_argument("--features", choices=["signature", "handcrafted"])
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--shots_per_class", type=int, default=DEFAULT_SHOTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--hf_csv", default=DATA_PATHS["handcrafted"])
    p.add_argument("--sig_csv", default=DATA_PATHS["signature"])
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=17)
    p.add_argument("--n_neighbors", type=int, default=DEFAULT_N_NEIGHBORS)
    p.add_argument("--csv_box")

    return p.parse_args()


def batch_train(args):
    """
    Run batch training for RF and KNN with different settings.

    Parameters:
    - args (argparse.Namespace): Arguments object with fields:
      * n_neighbors (int): Number of neighbors for KNN.
      * seed (int): Random seed.
      * shots_per_class (int): Number of shots per class for few-shot runs.

    Returns:
    - None. Results are saved to output tables on disk.
    """
    # 80/20
    for mode in ("handcrafted", "signature"):
        run_rf(DATA_PATHS[mode], mode, OUTPUT_DIR)
        run_knn_full(DATA_PATHS[mode], mode, OUTPUT_DIR, n_neighbors=args.n_neighbors)
    # 80/20 + augment
    for mode in ("handcrafted", "signature"):
        run_rf_aug(DATA_PATHS[mode], mode, OUTPUT_DIR, seed=args.seed)
        run_knn_augment(DATA_PATHS[mode], mode, OUTPUT_DIR, n_neighbors=args.n_neighbors, seed=args.seed)
    # few-shot
    for mode in ("handcrafted", "signature"):
        run_rf_few(DATA_PATHS[mode], mode, OUTPUT_DIR, k=args.shots_per_class, seed=args.seed)
        run_knn_few(DATA_PATHS[mode], mode, OUTPUT_DIR, k=args.shots_per_class, seed=args.seed,
                    n_neighbors=args.n_neighbors)
    # few-shot + augment
    for mode in ("handcrafted", "signature"):
        run_rf_aug_few(DATA_PATHS[mode], mode, OUTPUT_DIR, k=args.shots_per_class, seed=args.seed)
        run_knn_few_aug(DATA_PATHS[mode], mode, k=args.shots_per_class, seed=args.seed,
                        n_neighbors=args.n_neighbors)
    save_results_table()

def batch_train_cv(args):
    """
    Run episodic CV training for RF and KNN (with and without augmentation).
    """
    seeds = _parse_seeds_arg(args.seeds)
    k = args.shots_per_class

    for mode in ("handcrafted", "signature"):
        run_rf_few_cv_episodes(DATA_PATHS[mode], mode, k=k, seeds=seeds)

    for mode in ("handcrafted", "signature"):
        run_rf_aug_few_cv_episodes(DATA_PATHS[mode], mode, k=k, seeds=seeds)

    for mode in ("handcrafted", "signature"):
        run_knn_few_cv_episodes(DATA_PATHS[mode], mode, k=k,
                                n_neighbors=args.n_neighbors, seeds=seeds)

    for mode in ("handcrafted", "signature"):
        run_knn_few_aug_cv_episodes(DATA_PATHS[mode], mode, k=k,
                                    n_neighbors=args.n_neighbors, seeds=seeds)


def main():
    args = parse_args()

    # Set global seeds for reproducibility
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(args.seed)
    np.random.seed(args.seed)

   # # requires at least one
    if not any([args.scenario, args.train, args.train_cv]):
        raise SystemExit("Debe indicar --scenario o --train o --train_cv")

    # batch
    if args.train:
        batch_train(args); return
    if args.train_cv:
        batch_train_cv(args); return

    # scenarios of running different models   
    if args.scenario == "rf":
        run_rf(args.csv, args.features, OUTPUT_DIR); save_results_table()

    elif args.scenario == "rf_few":
        run_rf_few(args.csv, args.features, OUTPUT_DIR, k=args.shots_per_class, seed=args.seed); save_results_table()

    elif args.scenario == "rf_aug":
        run_rf_aug(args.csv, args.features, OUTPUT_DIR, seed=args.seed); save_results_table()

    elif args.scenario == "rf_aug_few":
        run_rf_aug_few(args.csv, args.features, OUTPUT_DIR, k=args.shots_per_class, seed=args.seed); save_results_table()

    elif args.scenario == "knn_few":
        run_knn_few(args.csv, args.features, OUTPUT_DIR, k=args.shots_per_class, seed=args.seed,
                    n_neighbors=args.n_neighbors); save_results_table()

    elif args.scenario == "knn_full":
        run_knn_full(args.csv, args.features, OUTPUT_DIR, n_neighbors=args.n_neighbors); save_results_table()

    elif args.scenario == "knn_aug":
        run_knn_augment(args.csv, args.features, OUTPUT_DIR, n_neighbors=args.n_neighbors, seed=args.seed); save_results_table()

    elif args.scenario == "knn_few_aug":
        run_knn_few_aug(args.csv, args.features, k=args.shots_per_class, seed=args.seed,
                        n_neighbors=args.n_neighbors); save_results_table()


    # Scenario of running few-shot sweep across a range of k values
    elif args.scenario == "fewshot_sweep":
        res: Dict[str, Any] = {}
        for n in range(args.k_min, args.k_max):
            res = run_fewshot_experiment(n, args.seed, res,
                                         hf_csv=DATA_PATHS["handcrafted"],
                                         sig_csv=DATA_PATHS["signature"])
        plot_fewshot_results(res); save_results_table()



    # Scenarios of running episodic CV for KNN and RF (with or without augmentation)
    elif args.scenario == "knn_few_cv":
        if not args.csv or not args.features:
            raise ValueError("--csv y --features son obligatorios para 'knn_few_cv'")
        seeds = _parse_seeds_arg(args.seeds)
        _, summary = run_knn_few_cv_episodes(args.csv, args.features, k=args.shots_per_class,
                                             n_neighbors=args.n_neighbors, seeds=seeds)

    elif args.scenario == "knn_few_aug_cv":
        if not args.csv or not args.features:
            raise ValueError("--csv y --features son obligatorios para 'knn_few_aug_cv'")
        seeds = _parse_seeds_arg(args.seeds)
        _, summary = run_knn_few_aug_cv_episodes(args.csv, args.features, k=args.shots_per_class,
                                                 n_neighbors=args.n_neighbors, seeds=seeds)

    elif args.scenario == "rf_few_cv":
        if not args.csv or not args.features:
            raise ValueError("--csv y --features son obligatorios para 'rf_few_cv'")
        seeds = _parse_seeds_arg(args.seeds)
        _, summary = run_rf_few_cv_episodes(args.csv, args.features, k=args.shots_per_class, seeds=seeds)

    elif args.scenario == "rf_aug_few_cv":
        if not args.csv or not args.features:
            raise ValueError("--csv y --features son obligatorios para 'rf_aug_few_cv'")
        seeds = _parse_seeds_arg(args.seeds)
        _, summary = run_rf_aug_few_cv_episodes(args.csv, args.features, k=args.shots_per_class, seeds=seeds)



    # Scenarios of running different plotting settings (exploratory, confusion matrix, boxplot)
    elif args.scenario == "exploratory_plots":
        seeds = _parse_seeds_arg(args.seeds) if args.seeds else [DEFAULT_SEED]
        run_exploratory_plots(
            seeds=seeds,
            hf_csv=DATA_PATHS["handcrafted"],
            sig_csv=DATA_PATHS["signature"],
        )

    elif args.scenario == "confusion_matrix":
        seeds = _parse_seeds_arg(args.seeds) if args.seeds else [DEFAULT_SEED]
        run_confusion_matrix_agg(
            k=args.shots_per_class,
            n_neighbors=args.n_neighbors,
            seeds=seeds,
            hf_csv=DATA_PATHS["handcrafted"],
            sig_csv=DATA_PATHS["signature"],
        )

    elif args.scenario == "boxplot_knn":
        plot_knn_few_boxplot()



if __name__ == "__main__":
    main()
