"""
Load the trained model, run it on test_dataset.csv, and write the
competition submission file: [TeamName].csv with a single column
'overall_yield', 50 rows, in the same order as test_dataset.csv.

Run:
    python src/predict.py --team_name MyTeam
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import engineer_features

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "best_model.joblib"
OUTPUT_DIR = ROOT / "submissions"


def main(team_name: str):
    test_path = DATA_DIR / "test_dataset.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Put test_dataset.csv in {DATA_DIR}/ first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run train.py first to produce best_model.joblib.")

    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]

    test_df = pd.read_csv(test_path)
    test_feat = engineer_features(test_df)[features]

    preds = model.predict(test_feat)

    out = pd.DataFrame({"overall_yield": preds.round(3)})

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{team_name}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} predictions to {out_path}")
    print(out.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_name", type=str, default="MyTeam")
    args = parser.parse_args()
    main(args.team_name)
