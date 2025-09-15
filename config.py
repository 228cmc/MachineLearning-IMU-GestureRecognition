"""
File with default parameters and paths for the project.
"""

import os


# Root of the repository
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Directories (can be overridden by environment variables)
DATA_DIR    = os.getenv("DATA_DIR",    os.path.join(BASE_DIR, "data"))
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.join(BASE_DIR, "Results"))
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# Input CSV files
DATA_PATHS = {
    "handcrafted": os.path.join(DATA_DIR, "handcrafted_training_data.csv"),
    "signature":   os.path.join(DATA_DIR, "signature_training_data.csv"),
}

# Default experimental parameters (can be overridden by environment variables)
DEFAULT_SHOTS         = int(os.getenv("DEFAULT_SHOTS", "5"))
DEFAULT_N_NEIGHBORS   = int(os.getenv("DEFAULT_N_NEIGHBORS", "5"))
DEFAULT_SEED          = int(os.getenv("DEFAULT_SEED", "42"))
NOISE_REPS            = int(os.getenv("NOISE_REPS", "2"))
MIXUP_REPS            = int(os.getenv("MIXUP_REPS", "3"))
KNN_NEIGHBORS_FEWSHOT = int(os.getenv("KNN_NEIGHBORS_FEWSHOT", "5"))

# Style for figures
FIG_FONT = "Times New Roman"
