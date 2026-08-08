"""
Train and validate several candidate models on train_dataset.csv using
repeated K-Fold cross-validation (safe strategy for a 150-row dataset),
pick the best one, retrain it on the full training set, and save it.

Run:
    python src/train.py
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from feature_engineering import engineer_features, FEATURE_COLUMNS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "best_model.joblib"

RANDOM_STATE = 42


def load_data():
    train_path = DATA_DIR / "train_dataset.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Put train_dataset.csv in {DATA_DIR}/ before running this script."
        )
    df = pd.read_csv(train_path)
    df = engineer_features(df)
    X = df[FEATURE_COLUMNS]
    y = df["overall_yield"]
    return X, y


def build_candidates():
    """Return dict of name -> sklearn Pipeline (scaling + model)."""
    candidates = {}

    candidates["Ridge (physics-linear baseline)"] = Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])

    candidates["RandomForest"] = Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            random_state=RANDOM_STATE
        )),
    ])

    candidates["GradientBoosting"] = Pipeline([
        ("model", GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=3, random_state=RANDOM_STATE
        )),
    ])

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    candidates["GaussianProcess"] = Pipeline([
        ("scale", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=5, random_state=RANDOM_STATE
        )),
    ])

    candidates["SVR (RBF)"] = Pipeline([
        ("scale", StandardScaler()),
        ("model", SVR(kernel="rbf", C=10, epsilon=0.5)),
    ])

    return candidates


def evaluate_candidates(X, y, candidates, n_splits=5, n_repeats=10):
    """Repeated K-Fold CV -> mean/std RMSE per candidate. This is the safe
    way to compare models on only 150 rows: a single train/test split has
    too much variance to trust."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(
            pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        rmse_scores = -scores
        results[name] = (rmse_scores.mean(), rmse_scores.std())
        print(f"{name:35s}  RMSE = {rmse_scores.mean():.4f}  (+/- {rmse_scores.std():.4f})")
    return results


def tune_random_forest(X, y):
    """Light hyperparameter search for the RF, nested inside CV so we don't
    leak test information into the choice of hyperparameters."""
    param_grid = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [1, 3, 5],
    }
    pipe = Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE))])
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    grid.fit(X, y)
    print("\nBest RF params:", grid.best_params_)
    print("Best RF CV RMSE:", -grid.best_score_)
    return grid.best_estimator_, -grid.best_score_


def main():
    X, y = load_data()
    print(f"Loaded {len(X)} training rows, {X.shape[1]} engineered features.\n")

    candidates = build_candidates()
    print("=== Repeated 5-Fold CV comparison (10 repeats) ===")
    results = evaluate_candidates(X, y, candidates)

    print("\n=== Tuning best tree model (Random Forest) ===")
    best_rf, best_rf_rmse = tune_random_forest(X, y)

    # Pick overall winner among all candidates + tuned RF
    all_results = {name: r[0] for name, r in results.items()}
    all_results["RandomForest (tuned)"] = best_rf_rmse
    winner_name = min(all_results, key=all_results.get)
    print(f"\n>>> Winning model: {winner_name}  (CV RMSE = {all_results[winner_name]:.4f})")

    if winner_name == "RandomForest (tuned)":
        final_model = best_rf
    else:
        final_model = candidates[winner_name]

    # Refit on the FULL training set for final predictions
    final_model.fit(X, y)
    joblib.dump({"model": final_model, "features": FEATURE_COLUMNS}, MODEL_PATH)
    print(f"\nSaved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
