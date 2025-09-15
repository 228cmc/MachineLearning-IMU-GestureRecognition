
# tree
C:.
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
│   ├── metrics/ (...)
│   ├── models/
│   │   ├── KNN_Augmented_n5.pkl
│   │   ├── KNN_few_k5.pkl
│   │   ├── KNN_full_n5.pkl
│   │   ├── RandomForest.pkl
│   │   ├── RandomForest_Augmented.pkl
│   │   ├── RandomForest_Augmented_few_k5.pkl
│   │   ├── RandomForest_few_k5.pkl
│   │   └── fewshot_sweep/ (...)
│   ├── plots/
│   │   ├── class_distribution_hf_vs_sig.png
│   │   ├── fewshot_compare_signature_noaug.png
│   │   ├── fewshot_results_mine_randomforestParametrization_equal_exercise.png
│   │   ├── knn_few_boxplot_seeds0_99.png
│   │   ├── pca_features.png
│   │   └── confusion/ (...)
│   ├── stats/ (...)
│   └── tables/ (...)
├── src/
│   ├── main.py
│   ├── models.py
│   ├── pipelines.py
│   ├── __init__.py
│   └── utils/
│       ├── augment.py
│       ├── experiments.py
│       ├── plotting.py
│       ├── test.py
│       ├── utils.py
│       └── __init__.py



# General usage pattern

```powershell
python src\main.py --scenario <scenario> [options]
```

**Notes:**

* Input datasets:

  * Handcrafted → `data\handcrafted_training_data.csv`
  * Signature   → `data\signature_training_data.csv`
* All outputs are written under `Results\` (models, metrics, tables, plots).
* Default arguments:
  `--shots_per_class 5`, `--n_neighbors 5`, `--seed 42`.

---

## 1) Exploratory plots (distribution + PCA)

```powershell
python src\main.py --scenario exploratory_plots --seeds "42"
```

**Generates:**

* Class distribution → `Results\plots\class_distribution_hf_vs_sig.png`
* PCA projections   → `Results\plots\pca_features.png`

---

## 2) Simple training/validation (80/20 and few-shot)

Run all base models (RF, KNN) with and without augmentation.

```powershell
python src\main.py --train --shots_per_class 5 --n_neighbors 5 --seed 42
```

**Outputs:**

* **Models:** `Results\models\*.pkl`
* **Metrics (TXT):** `Results\metrics\*.txt`
* **Confusion matrices (if k=5):** `Results\plots\confusion\*.png`

Covers in one go:

* Random Forest (handcrafted + signature) – 80/20, 80/20+aug, few-shot, few-shot+aug
* KNN (handcrafted + signature) – 80/20, 80/20+aug, few-shot, few-shot+aug

---


for watching the results  matrices
python src/main.py --scenario confusion_matrix --shots_per_class 5 --n_neighbors 5 --seeds "42"


## 3) Few-shot sweep

Evaluate varying number of shots (*k*).

```powershell
python src\main.py --scenario fewshot_sweep --k_min 2 --k_max 17 --seed 42
```

**Outputs:**

* Results table → `Results\tables\experiments_summary_selectindex.csv`
* Line plots   → `Results\plots\fewshot_results_mine_randomforestParametrization_equal_exercise.png`

---

## 4) Comparison between single and episodic

This script allows manual control of experiments.

```powershell
# Vary k (Signature features)
foreach ($k in 2..6) {
  python src\utils\experiments.py --csv data\signature_training_data.csv --features signature --shots $k --n_neighbors 5 --out_csv Results\tables\fewshot_compare.csv
}

# Append another run and plot
python src\utils\experiments.py --csv data\signature_training_data.csv --features signature --shots 6 --n_neighbors 5 --out_csv Results\tables\fewshot_compare.csv --plot

# With augmentation (Signature, k=5)
python src\utils\experiments.py --csv data\signature_training_data.csv --features signature --shots 5 --n_neighbors 5 --augment
```

**Outputs:**

* `Results\tables\fewshot_compare.csv`
* `Results\plots\fewshot_compare_signature_<aug|noaug>.png`

---

## 5) Multi-seed evaluation (e.g. 42–46)

```powershell
python src\main.py --train_cv --shots_per_class 5 --n_neighbors 5 --seeds "42,43,44,45,46"
```

**Outputs (CSV):**

* `Results\tables\rf_few_cv_handcrafted.csv`
* `Results\tables\rf_few_cv_signature.csv`
* `Results\tables\rf_aug_few_cv_handcrafted.csv`
* `Results\tables\rf_aug_few_cv_signature.csv`
* `Results\tables\knn_few_cv_handcrafted.csv`
* `Results\tables\knn_few_cv_signature.csv`
* `Results\tables\knn_few_aug_cv_handcrafted.csv`
* `Results\tables\knn_few_aug_cv_signature.csv`

---

## 6) Confusion matrices (aggregated over seeds)

```powershell
python src\main.py --scenario confusion_matrix --shots_per_class 5 --n_neighbors 5 --seeds "42,43,44,45,46"
```

**Outputs:**
Confusion matrices aggregated over the 5 seeds in `Results\plots\confusion\`.


## Running test results for few-shot

```powershell
python src\utils\test.py


## 7) Episodic CV with 100 seeds (statistical analysis)

```powershell
python src\main.py --train_cv --shots_per_class 5 --n_neighbors 5 --seeds "0..99"
```

**Outputs:**
CSV tables only under `Results\tables\`.

---

## 8) Boxplot KNN (reads the 100-seed CSVs)

```powershell
python src\main.py --scenario boxplot_knn
```

**Outputs:**
Boxplots comparing handcrafted vs signature, with and without augmentation →
`Results\plots\knn_few_boxplot_seeds0_99.png`

---


```

---

## Alternative experiments (`experiments.py`)

Manual experimentation with `src\utils\experiments.py` as shown in section 4.

---

## Run a single scenario directly

Example:

```powershell
python src\main.py --scenario rf_aug_few_cv --csv data\handcrafted_training_data.csv --features handcrafted --shots_per_class 5
```

---

## Where everything lands

* **Models:** `Results\models\*.pkl`
* **Metrics:** `Results\metrics\*.txt`
* **Tables:** `Results\tables\*.csv`
* **Plots:** `Results\plots\*`
  (confusion matrices under `Results\plots\confusion\`)

---

## References

* Zhang, H., Cisse, M., Dauphin, Y., & Lopez-Paz, D. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR.
* Thulasidasan, S., Chennupati, G., Bilmes, J., Bhattacharya, T., & Michalak, S. (2019). *On Mixup Training: Improved Calibration and Predictive Uncertainty*. NeurIPS.
* Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., & Wierstra, D. (2016). *Matching Networks for One Shot Learning*. NeurIPS.
* Snell, J., Swersky, K., & Zemel, R. (2017). *Prototypical Networks for Few-shot Learning*. NeurIPS.
* Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P., & Hospedales, T. (2018). *Learning to Compare: Relation Network for Few-Shot Learning*. CVPR.

