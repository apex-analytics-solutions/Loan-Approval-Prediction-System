"""
Centralized project configuration, logging, path management, and artifact loading.

Every other module imports from here — this is the single source of truth
for paths, constants, and the logging setup.
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR      = PROJECT_ROOT / "data"
MODELS_DIR    = PROJECT_ROOT / "models"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR      = PROJECT_ROOT / "logs"

DATASET_PATH          = DATA_DIR / "loan_dataset_2025.csv"
ARTIFACTS_PATH        = MODELS_DIR / "loan_project_artifacts.pkl"
MODEL_COMPARISON_CSV  = OUTPUTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_CSV = OUTPUTS_DIR / "feature_importance.csv"
DASHBOARD_DATA_PATH   = OUTPUTS_DIR / "dashboard_data.pkl"


# ─────────────────────────────────────────────
# Training Constants
# ─────────────────────────────────────────────

TARGET          = "loan_paid_back"
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
CV_FOLDS        = 5
DROP_COLS       = ["monthly_income"]

VERSION = "2.1.0"


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging(
    name: str = "loan_project",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with console + optional file output.

    Call once at the entry point of each executable (train.py, app.py).
    Subsequent calls with the same *name* return the already-configured logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-14s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Directory Setup
# ─────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all required project directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Artifact Loading
# ─────────────────────────────────────────────

REQUIRED_ARTIFACT_KEYS = {"model", "feature_names", "encoders", "best_model_name"}


def load_artifacts(
    path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Load and validate the model artifact bundle.

    Returns dict with keys: model, feature_names, encoders, best_model_name.
    Raises FileNotFoundError or ValueError on problems.
    """
    path = path or ARTIFACTS_PATH
    log = logger or logging.getLogger("loan_project")

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found at {path}. Run train.py first."
        )

    log.info("Loading artifacts from %s", path)

    with open(path, "rb") as f:
        artifacts = pickle.load(f)

    missing = REQUIRED_ARTIFACT_KEYS - set(artifacts.keys())
    if missing:
        raise ValueError(
            f"Artifact file is corrupt or outdated — missing keys: {missing}. "
            f"Re-run train.py to regenerate."
        )

    model = artifacts["model"]
    features: List[str] = artifacts["feature_names"]
    encoders: Dict = artifacts["encoders"]
    best_name: str = artifacts.get("best_model_name", "Unknown")

    if not features:
        raise ValueError("Artifact 'feature_names' is empty. Re-run train.py.")

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model type {type(model).__name__} does not support predict_proba. "
            f"Training must produce a probabilistic classifier."
        )

    log.info(
        "Artifacts loaded: model=%s, features=%d, encoders=%d, best=%s",
        type(model).__name__, len(features), len(encoders), best_name,
    )

    return artifacts


_cached_artifacts: Optional[Dict[str, Any]] = None


def load_artifacts_cached() -> Dict[str, Any]:
    """
    Process-level cached loader.
    First call deserializes from disk; subsequent calls return the same
    objects instantly.  Works both inside Streamlit and in plain Python
    (train.py, tests).
    """
    global _cached_artifacts
    if _cached_artifacts is None:
        _cached_artifacts = load_artifacts()
    return _cached_artifacts


def save_artifacts(
    artifacts: Dict[str, Any],
    path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Serialize artifact bundle to disk."""
    path = path or ARTIFACTS_PATH
    log = logger or logging.getLogger("loan_project")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(artifacts, f)

    log.info("Artifacts saved to %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path
