# Reactor Yield Prediction — Hackathon Kit

## 0. Setup (do this first)

```bash
cd reactor_project
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Place the two provided files here:
```
data/train_dataset.csv
data/test_dataset.csv
```
(Download them from the Google Drive links in the problem statement —
File → Download in Drive, not "open with Google Sheets".)

## 1. Train and select the best model

```bash
python src/train.py
```

This will:
1. Load `train_dataset.csv` and engineer physics-informed features
   (`src/feature_engineering.py`).
2. Compare 5 candidate models (Ridge baseline, Random Forest, Gradient
   Boosting, Gaussian Process, SVR) using **Repeated 5-Fold CV (10 repeats)**
   — the safe way to compare models on only 150 rows.
3. Run a light hyperparameter search on Random Forest, nested inside CV.
4. Pick whichever candidate has the lowest mean CV RMSE, refit it on the
   **full** training set, and save it to `best_model.joblib`.

You'll see console output like:
```
Ridge (physics-linear baseline)     RMSE = 3.1245  (+/- 0.6102)
RandomForest                        RMSE = 2.4013  (+/- 0.5544)
GradientBoosting                    RMSE = 2.3810  (+/- 0.5721)
GaussianProcess                     RMSE = 2.2905  (+/- 0.5980)
SVR (RBF)                           RMSE = 2.6650  (+/- 0.6203)

Best RF params: {...}
Best RF CV RMSE: 2.15...

>>> Winning model: RandomForest (tuned)  (CV RMSE = 2.15..)
```
(Actual numbers depend on the real data — this is illustrative.)

## 2. Generate the competition submission

```bash
python src/predict.py --team_name YourTeamName
```

Writes `submissions/YourTeamName.csv` — exactly 50 rows, one column
`overall_yield`, in test_dataset.csv row order. **This is the file you
upload to Unstop.** Rename `--team_name` to your actual registered team
name before the real submission (remember: only ONE submission allowed).

## 3. Explore, visualize, and build your pitch notebook

Open `notebook.ipynb` in Jupyter — it has EDA, the physics explanation,
the CV model comparison, SHAP feature-importance plots, and the
overfitting-prevention writeup, all in one place. This is what you'd polish
and hand in as the "fully documented Jupyter Notebook" if shortlisted.

```bash
jupyter notebook notebook.ipynb
```

## 4. Before you submit — checklist

- [ ] Did you look at `train.describe()` for weird values / outliers first?
- [ ] Does the CV RMSE spread (the `+/- std`) look reasonable, or is one
      fold wildly different (possible outlier row)?
- [ ] Did the non-linear models actually beat the Ridge baseline by a
      meaningful margin? If not, that's a sign to simplify, not add
      complexity — say this in your pitch, it shows engineering judgement.
- [ ] Try swapping in `xgboost` or `lightgbm` if installed — often a small
      RMSE improvement over sklearn's GradientBoostingRegressor, but same
      CV-comparison logic applies. Don't add it just because it's fashionable
      — only keep it if CV RMSE actually improves.
- [ ] Sanity-check predictions: no yield values below 0% or above 100%.
      If your model predicts outside that range, clip it — physically the
      yield of B can't exceed 100% of A, and you should say this out loud
      in your pitch (shows you understand physical constraints).
- [ ] Re-read the "Process Insight" bullet in the rubric — practice a
      60-90 second explanation of *why* residence time and temperature
      interact the way they do, using section 1 of the notebook.

## 5. Ideas to push RMSE further (if you have time before the deadline)

- **Stacking/ensembling** the top-2 CV models (simple average of predictions)
  often beats any single model on small, noisy datasets.
- **Leave-one-out CV** (150 folds) gives an even lower-variance RMSE
  estimate than 5-fold, at the cost of more compute — cheap here since the
  dataset is tiny.
- **Monotonic constraints** in XGBoost/LightGBM on features you're
  physically confident about (e.g. throughput should not increase yield
  indefinitely) — makes the model provably consistent with chemistry,
  which is a strong pitch point for "robustness."
- Try clipping/asserting `0 <= prediction <= 100` as a final post-processing
  step — free RMSE points if your model ever strays outside physical bounds.
