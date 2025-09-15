## 1. Few-Shot Learning
Few-Shot Learning (FSL) is about training classifiers when only a few samples per class are available. Instead of large datasets, FSL leverages strategies such as:  
- **Metric-based**: classify by similarity in a feature space (e.g., KNN, Prototypical Networks).  
- **Optimization-based**: fast adaptation through meta-learning methods (e.g., MAML).  
- **Transfer-based**: adapt pre-trained models with small new datasets.  

In this project, a **KNN pipeline** was used. Each class is represented by *k* examples (1–5 shots), and new samples are classified based on Euclidean distance. This approach is conceptually related to **Prototypical Networks** (Snell et al., 2017), which define a prototype per class in the embedding space.  

Relevant prior work:  
- **Matching Networks** (Vinyals et al., 2016): attention-based nearest-neighbor classification.  
- **FS-HGR** (Rahimian et al., 2021): few-shot applied to EMG gesture recognition.  
- **Schlüsener & Bücker (2022)**: fast learning of dynamic gestures with few-shot models.  
- **Zhang et al. (2020)**: meta-learning for multimodal gestures.  

Here, handcrafted and signature features were directly used as the representation space, with KNN as a metric-based baseline to test separability under low-data conditions.

---

## 2. Data Augmentation
Data augmentation increases training diversity while preserving gesture meaning. Two simple methods were implemented:  

### 2.1 Gaussian Noise
Random noise simulates sensor variability:  
\[
\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
\]  
This helps robustness against measurement errors and drift.  

### 2.2 Mixup
New samples are interpolations of pairs of examples:  
\[
\tilde{x} = \lambda x_i + (1-\lambda)x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda)y_j
\]  
with $\lambda \sim \text{Beta}(\alpha, \alpha)$.  
Mixup produces intermediate gestures, enriching the dataset.  

Both methods were applied on handcrafted and signature features.

---

## 3. Random Forest
Random Forest (RF) is an ensemble of decision trees with predictions aggregated by majority vote.  

Reasons for use:  
- Good performance on structured IMU features.  
- Robust to noise and overfitting.  
- Serves as a **baseline** for comparison with augmentation and few-shot KNN.  

RF is widely used in IMU/EMG gesture recognition for its balance between accuracy and interpretability.

---

## 4. Deep Learning (Context)
Deep models (CNNs, CNN–RNN hybrids, Transformers) capture both local and long-term dependencies and often achieve high accuracy.  
However, they need **large labeled datasets** and **high compute**.  

This project instead focuses on **lightweight classifiers (RF, KNN)** combined with **data-efficient methods (augmentation, few-shot)** for low-data scenarios.

---

## 5. Project Goal
The goal is to **develop efficient IMU-based gesture recognition pipelines under low-data conditions**.  

Main objectives:  
1. Compare handcrafted features vs. algebraic signatures.  
2. Evaluate training strategies:  
   - Baseline RF  
   - RF + augmentation  
   - Few-shot KNN (with/without augmentation)  
3. Analyze trade-offs in accuracy, robustness, interpretability, and data efficiency.  

Key findings:  
- **RF** performs very well with full data.  
- **Augmentation** improves stability, especially for handcrafted features.  
- **Few-shot KNN** highlights the advantage of signature features when only 1–5 samples per class are available.
