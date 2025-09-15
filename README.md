# IMU-Based Gesture Recognition under Limited Data

## Introduction

This project explores how to recognize human gestures using small wearable motion sensors (IMUs). The challenge is working with very little labeled data, since in practice we cannot always collect hundreds of examples. To address this, we compare two classic classifiers (Random Forest and KNN), two feature types (handcrafted vs. signatures), and strategies such as data augmentation and few-shot learning. The goal is to build recognition pipelines that are accurate, stable, and practical for real-world scenarios like wearables, rehabilitation, and accessibility.

---

## Project Structure

```text
.
├── .gitignore
├── config.py
├── README.md
├── requirements.txt
├── results.txt
├── .vscode/
│   └── launch.json
├── data/
│   ├── handcrafted_training_data.csv
│   └── signature_training_data.csv
├── docs/
│   ├── DiagramsCode.png
│   └── Techniques.md
├── Results/
│   ├── metrics/
│   ├── models/
│   │   ├── KNN_Augmented_n5.pkl
│   │   ├── KNN_few_k5.pkl
│   │   ├── RandomForest.pkl
│   │   └── ...
│   ├── plots/
│   │   ├── class_distribution_hf_vs_sig.png
│   │   ├── fewshot_compare_signature_noaug.png
│   │   ├── knn_few_boxplot_seeds0_99.png
│   │   ├── pca_features.png
│   │   └── confusion/
│   ├── stats/
│   └── tables/
├── src/
│   ├── main.py
│   ├── models.py
│   ├── pipelines.py
│   └── utils/
│       ├── augment.py
│       ├── experiments.py
│       ├── plotting.py
│       ├── test.py
│       └── utils.py
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/imu-gesture-fewshot.git
cd imu-gesture-fewshot
pip install -r requirements.txt
```

---

## General Usage

All scenarios are run from `main.py`:

```bash
python src/main.py --scenario <scenario> [options]
```

**Datasets:**

* Handcrafted features → `data/handcrafted_training_data.csv`
* Signature features → `data/signature_training_data.csv`
  **Outputs:** under `Results/` (models, metrics, plots, tables).

---

## Examples

### 1. Exploratory plots (class distribution + PCA)

```bash
python src/main.py --scenario exploratory_plots --seeds "42"
```

### 2. Train baseline and few-shot models

```bash
python src/main.py --train --shots_per_class 5 --n_neighbors 5 --seed 42
```

### 3. Few-shot sweep (varying k)

```bash
python src/main.py --scenario fewshot_sweep --k_min 2 --k_max 17 --seed 42
```

### 4. Multi-seed evaluation

```bash
python src/main.py --train_cv --shots_per_class 5 --n_neighbors 5 --seeds "42,43,44,45,46"
```

### 5. Boxplots with 100 seeds

```bash
python src/main.py --scenario boxplot_knn
```

---

## Key Features

* Compare **handcrafted vs. signature features**.
* Evaluate **Random Forest** and **KNN** classifiers.
* Implement **data augmentation** (Gaussian noise + mixup).
* Run **few-shot learning** with episodic cross-validation.
* Export results: confusion matrices, accuracy curves, statistical tests.

---

## Results (highlights)

* Signature features + KNN + augmentation achieve **near-perfect accuracy with only k=6 shots per class**.
* Augmentation improves stability, especially for handcrafted features.
* Random Forest is a strong baseline under full-data, but less flexible in few-shot settings.
* Episodic evaluation produces more reliable results than single runs.

---

## Limitations

* Single-subject, single-session dataset → limited generalization.
* Only pre-processed feature vectors available (no raw IMU signals).

---

## Future Work

* Extend to **multi-user datasets** and new sensor placements.
* Explore **temporal augmentations** and online adaptation.

---

## References

* Zhang et al. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR.
* Thulasidasan et al. (2019). *On Mixup Training: Improved Calibration and Predictive Uncertainty*. NeurIPS.
* Vinyals et al. (2016). *Matching Networks for One Shot Learning*. NeurIPS.
* Snell et al. (2017). *Prototypical Networks for Few-shot Learning*. NeurIPS.

=
