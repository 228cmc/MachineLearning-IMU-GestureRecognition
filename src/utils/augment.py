import numpy as np
import pandas as pd

DEFAULT_NOISE_LEVEL = 0.05
DEFAULT_MIXUP_ALPHA = 0.4

class DataAugmenter:
    def __init__(self, noise_level=DEFAULT_NOISE_LEVEL, mixup_alpha=DEFAULT_MIXUP_ALPHA, seed: int | None = None):
        """
        A class  that generates synthetic samples.

        Attributes received at initialization:
        - noise_level (float): standard deviation of Gaussian noise to add to features.
        - mixup_alpha (float): parameter of the Beta distribution used for Mixup coefficients.
        - seed (int or None): random seed for reproducibility (passed to numpy Generator).
        """
        self.noise_level = noise_level
        self.mixup_alpha = mixup_alpha
        self.rng = np.random.default_rng(seed)



    def add_gaussian_noise(self, features_matrix):
        """
        Add Gaussian noise to the input features.

        Parameters:
        - features_matrix (ndarray): Input features.

        Returns:
        - ndarray: Features with added Gaussian noise.
        """
        noise = self.rng.normal(0.0, self.noise_level, features_matrix.shape)
        return features_matrix + noise




    def _mix_two_samples(self, x1, x2):
        """
        Mix two samples using a Beta-distributed coefficient.

        Parameters:
        - x1 (ndarray): First sample.
        - x2 (ndarray): Second sample.

        Returns:
        - ndarray: Mixed sample.
        """
        lam = self.rng.beta(self.mixup_alpha, self.mixup_alpha)
        return lam * x1 + (1.0 - lam) * x2




    def mixup(self, X, y, repetitions):
        """
        Create synthetic samples by intra-class Mixup.

        Parameters:
        - X (ndarray): Feature matrix.
        - y (ndarray): Labels.
        - repetitions (int): Number of Mixup iterations.

        Returns:
        - (ndarray, ndarray): Mixed features and labels.
        """
        # container for synthetic samples
        mixed_rows = []
        n = len(X)

        # skip if not enough samples or repetitions <= 0
        if n < 2 or repetitions <= 0:
            return X[:0], y[:0]
        
        # determine unique classes
        classes = np.unique(y)

        # repeat mixing process
        for _ in range(repetitions):
            for cls in classes:
                # indices of current class
                idx_cls = np.where(y == cls)[0]
                m = len(idx_cls)
                # need at least 2 samples in class
                if m < 2: 
                    continue
                for _ in range(m):
                    # pick two distinct samples
                    i, j = self.rng.choice(idx_cls, size=2, replace=False)
                    # mix them
                    x_mix = self._mix_two_samples(X[i], X[j])
                    # store mixed sample with label
                    mixed_rows.append((x_mix, cls))

        # build output arrays
        if mixed_rows:
            X_mix = np.vstack([row[0] for row in mixed_rows])
            y_mix = np.array([row[1] for row in mixed_rows], dtype=y.dtype)
        else:
            X_mix = X[:0]; y_mix = y[:0]
        return X_mix, y_mix





    def augment_dataframe(self, df, feat_cols, label_col, noise_reps, mixup_reps):
        """
        Generate an augmented dataset with noise and Mixup.

        Parameters:
        - df (DataFrame): Input dataset.
        - feat_cols (list): Feature column names.
        - label_col (str): Label column name.
        - noise_reps (int): Number of noisy copies.
        - mixup_reps (int): Number of Mixup repetitions.

        Returns:
        - DataFrame: Augmented dataset.
        """
        X = df[feat_cols].values
        y = df[label_col].values
        # list to store all data blocks
        blocks = []

        # add original data
        if len(X):
            df_block = pd.DataFrame(X, columns=feat_cols)
            df_block[label_col] = y
            df_block["Origin"] = "original"
            blocks.append(df_block)

        # add Gaussian noise replicas
        for r in range(max(0, noise_reps)):
            Xn = self.add_gaussian_noise(X)
            df_block = pd.DataFrame(Xn, columns=feat_cols)
            df_block[label_col] = y
            df_block["Origin"] = f"noise_{r+1}"
            blocks.append(df_block)

        # add Mixup samples
        X_mix, y_mix = self.mixup(X, y, repetitions=mixup_reps) if mixup_reps > 0 else (X[:0], y[:0])
        if len(X_mix):
            df_block = pd.DataFrame(X_mix, columns=feat_cols)
            df_block[label_col] = y_mix
            df_block["Origin"] = "mixup"
            blocks.append(df_block)

        # concatenate all blocks (or copy original if empty)
        final_df = pd.concat(blocks, ignore_index=True) if blocks else df.copy()
        return final_df
