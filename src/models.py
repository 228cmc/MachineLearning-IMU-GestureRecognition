import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class RandomForestTrainer:
    """
    Simple wrapper to initialize and expose a RandomForestClassifier.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

class FewShotKNN:
    """
    Few-shot KNN helper to train on k shots per class and predict the rest.
    """
    def __init__(self, n_neighbors: int = 5):
        """
        Store KNN neighbor count.

        Parameters:
        - n_neighbors (int): Number of neighbors for KNN.

        Returns:
        - None
        """
        self.n_neighbors = n_neighbors
        self.model = None

    def train_k_shot_and_predict_rest(self, X, y, shots_per_class: int = 5, seed: int = 42):
        """
        Train KNN on k shots per class and predict remaining samples.

        Parameters:
        - X (ndarray): Feature matrix.
        - y (ndarray): Labels.
        - shots_per_class (int): Shots per class for training.
        - seed (int): Random seed for episode split.

        Returns:
        - tuple: (Xtr, Xte, ytr, yte, ypred)
        """
        from utils.utils import select_k_per_class_indices

        # build support/query split
        train_idx, test_idx = select_k_per_class_indices(y, k=shots_per_class, seed=seed, strict=True)
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]

        # fit KNN with bounded neighbor count
        n_neighbors = min(self.n_neighbors, len(Xtr))
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(Xtr, ytr)
        ypred = self.model.predict(Xte)
        return Xtr, Xte, ytr, yte, ypred

def save_model(model, model_path: str, subdir: str = None):
    """
    Persist a model with joblib to the configured models directory.

    Parameters:
    - model: Trained model object to persist.
    - model_path (str): Relative or absolute path for the saved file.
    - subdir (str or None): Optional subdirectory under MODELS_DIR.

    Returns:
    None but in the Absolute path  the model is saved.
    """
    from config import MODELS_DIR
    root = MODELS_DIR
    if subdir:
        models_root = os.path.join(root, subdir)
    else:
        models_root = root

    full_path = model_path if os.path.isabs(model_path) else os.path.join(models_root, model_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    joblib.dump(model, full_path)

