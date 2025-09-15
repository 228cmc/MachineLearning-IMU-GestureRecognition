# IMU-Based Gesture Recognition under Limited Data

## Introduction
This project investigates how to recognize human gestures using wearable motion sensors (IMUs).  
The main challenge is working with very limited labeled data, since in practice it is not always possible to collect hundreds of examples.  

We address this by comparing two classic classifiers (Random Forest and KNN), two feature types (handcrafted vs. signatures), and strategies such as data augmentation and few-shot learning.  

The goal is to design recognition pipelines that are accurate, stable, and practical for real-world applications such as wearables, rehabilitation, and accessibility.

---

## Project Structure
```

.
├── .gitignore
├── config.py
├── README.md
├── requirements.txt
├── results.txt
├── .vscode/
│   └── launch.json
├── data/
│   ├── handcrafted\_training\_data.csv
│   └── signature\_training\_data.csv
├── docs/
│   ├── DiagramsCode.png
│   └── Techniques.md
├── Results/
│   ├── metrics/
│   ├── models/
│   │   ├── KNN\_Augmented\_n5.pkl
│   │   ├── KNN\_few\_k5.pkl
│   │   ├── KNN\_full\_n5.pkl
│   │   ├── RandomForest.pkl
│   │   ├── RandomForest\_Augmented.pkl
│   │   ├── RandomForest\_Augmented\_few\_k5.pkl
│   │   ├── RandomForest\_few\_k5.pkl
│   │   └── fewshot\_sweep/ (...)
│   ├── plots/
│   │   ├── class\_distribution\_hf\_vs\_sig.png
│   │   ├── fewshot\_compare\_signature\_noaug.png
│   │   ├── fewshot\_results\_mine\_randomforestParametrization\_equal\_exercise.png
│   │   ├── knn\_few\_boxplot\_seeds0\_99.png
│   │   ├── pca\_features.png
│   │   └── confusion/ (...)
│   ├── stats/ (...)
│   └── tables/ (...)
├── src/
│   ├── main.py
│   ├── models.py
│   ├── pipelines.py
│   ├── **init**.py
│   └── utils/
│       ├── augment.py
│       ├── experiments.py
│       ├── plotting.py
│       ├── test.py
│       ├── utils.py
│       └── **init**.py

````

---

## Installation
```bash
git clone https://github.com/<your-username>/imu-gesture-fewshot.git
cd imu-gesture-fewshot
pip install -r requirements.txt
````

---

## General Usage

```powershell
python src\main.py --scenario <scenario> [options]
```

**Datasets:**

* Handcrafted → `data\handcrafted_training_data.csv`
* Signature   → `data\signature_training_data.csv`

**Outputs:** stored in `Results\` (models, metrics, tables, plots).
**Default arguments:** `--shots_per_class 5`, `--n_neighbors 5`, `--seed 42`.

---

## 1) Exploratory plots

```powershell
python src\main.py --scenario exploratory_plots --seeds "42"
```

Generates:

* `Results\plots\class_distribution_hf_vs_sig.png`
* `Results\plots\pca_features.png`

---

## 2) Training/validation (80/20 and few-shot)

```powershell
python src\main.py --train --shots_per_class 5 --n_neighbors 5 --seed 42
```

Outputs:

* Models → `Results\models\*.pkl`
* Metrics TXT → `Results\metrics\*.txt`
* Confusion matrices (if k=5) → `Results\plots\confusion\*.png`

Covers:

* Random Forest (handcrafted + signature) – 80/20, 80/20+aug, few-shot, few-shot+aug
* KNN (handcrafted + signature) – 80/20, 80/20+aug, few-shot, few-shot+aug

---

## 3) Few-shot sweep

```powershell
python src\main.py --scenario fewshot_sweep --k_min 2 --k_max 17 --seed 42
```

Outputs:

* Results table → `Results\tables\experiments_summary_selectindex.csv`
* Line plots → `Results\plots\fewshot_results_mine_randomforestParametrization_equal_exercise.png`

---

## 4) Single vs episodic comparison (manual experiments)

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

Outputs:

* `Results\tables\fewshot_compare.csv`
* `Results\plots\fewshot_compare_signature_<aug|noaug>.png`

---

## 5) Multi-seed evaluation

```powershell
python src\main.py --train_cv --shots_per_class 5 --n_neighbors 5 --seeds "42,43,44,45,46"
```

Outputs CSVs under `Results\tables\`.

---

## 6) Confusion matrices

```powershell
python src\main.py --scenario confusion_matrix --shots_per_class 5 --n_neighbors 5 --seeds "42,43,44,45,46"
```

Outputs:

* Confusion matrices aggregated → `Results\plots\confusion\`

---

## 7) Episodic CV with 100 seeds

```powershell
python src\main.py --train_cv --shots_per_class 5 --n_neighbors 5 --seeds "0..99"
```

Outputs:

* CSV tables only in `Results\tables\`.

---

## 8) Boxplot KNN

```powershell
python src\main.py --scenario boxplot_knn
```

Outputs:

* Boxplots → `Results\plots\knn_few_boxplot_seeds0_99.png`

---

## 9) Manual test

```powershell
python src\utils\test.py
```

---

## 10) Run a single scenario directly

```powershell
python src\main.py --scenario rf_aug_few_cv --csv data\handcrafted_training_data.csv --features handcrafted --shots_per_class 5
```

---

## Results (highlights)

* Signature + KNN + augmentation achieves high accuracy with as few as 6 shots per class.
* Augmentation improves stability, especially for handcrafted features.
* Random Forest is strong in full-data, but less flexible in few-shot.
* Episodic evaluation provides more reliable estimates than single runs.

---

## Limitations

* Single-subject, single-session dataset → limited generalization.
* Only pre-processed feature vectors (no raw IMU signals).

---

## Future Work

* Extend to multi-user datasets and different sensor placements.
* Explore temporal augmentations and online adaptation.

---

## References

* Zhang, H., Cisse, M., Dauphin, Y., & Lopez-Paz, D. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR.
* Thulasidasan, S., Chennupati, G., Bilmes, J., Bhattacharya, T., & Michalak, S. (2019). *On Mixup Training: Improved Calibration and Predictive Uncertainty*. NeurIPS.
* Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., & Wierstra, D. (2016). *Matching Networks for One Shot Learning*. NeurIPS.
* Snell, J., Swersky, K., & Zemel, R. (2017). *Prototypical Networks for Few-shot Learning*. NeurIPS.
* Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P., & Hospedales, T. (2018). *Learning to Compare: Relation Network for Few-Shot Learning*. CVPR.

