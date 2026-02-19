did we # Session 03: Phase 3 Complete - Model Training & Evaluation

**Date:** 2026-02-10
**Focus:** ML modeling pipeline — training, tuning, evaluation, and explainability

## What Was Built

### 1. Trainer (src/models/trainer.py)

| Function | Description |
|----------|-------------|
| **load_fold** | Reads fold parquets, separates features/labels/metadata, fills NaN with training median |
| **get_feature_columns** | Returns numeric columns excluding 16 protected metadata columns |
| **train_baseline** | Logistic regression with class_weight="balanced", StandardScaler |
| **train_lgbm** | LightGBM Booster with early stopping and scale_pos_weight |
| **train_xgb** | XGBoost Booster with early stopping and scale_pos_weight |

### 2. Evaluator (src/models/evaluator.py)

| Function | Description |
|----------|-------------|
| **precision_at_k** | Precision among top-K predicted players |
| **recall_at_k** | Recall among top-K predicted players |
| **compute_metrics** | ROC-AUC, average precision, Brier score, P@K, R@K |
| **ensemble_predictions** | Weighted average of LGBM + XGB probabilities |
| **calibrate_probabilities** | Isotonic regression fitted on val, applied to test |
| **evaluate_fold** | Full pipeline: ensemble → calibrate → metrics → predictions DataFrame |
| **cross_fold_summary** | Mean/std of all metrics across 3 folds |

### 3. Tuner (src/models/tuner.py)

| Function | Description |
|----------|-------------|
| **tune_lgbm** | Optuna study (maximize ROC-AUC on val) for LightGBM hyperparams |
| **tune_xgb** | Optuna study (maximize ROC-AUC on val) for XGBoost hyperparams |
| **_lgbm_objective** | Single trial: samples from config tuning_space, trains, scores |
| **_xgb_objective** | Single trial: same for XGBoost |

**Tuning space:** num_leaves, learning_rate, feature_fraction, bagging_fraction, min_child_samples, reg_alpha, reg_lambda (LGBM) / max_depth, subsample, colsample_bytree, min_child_weight (XGB)

### 4. Explainer (src/models/explainer.py)

| Function | Description |
|----------|-------------|
| **compute_shap_values** | TreeExplainer for LGBM/XGB, samples down for performance |
| **feature_importance** | Mean |SHAP| per feature, sorted descending |
| **explain_player** | Top-N features driving a specific player's score |
| **generate_explanations** | Full SHAP pipeline: both models, combined importance |

### 5. Pipeline & CLI Updates

| Component | Description |
|-----------|-------------|
| **DataPipeline.run_phase3()** | Full orchestration: tune → train → evaluate → explain → save |
| **CLI `train`** | `--n-trials`, `--skip-tuning`, `--input`, `--output` |
| **CLI `evaluate`** | Display saved evaluation_results.json |

## Test Coverage

| Test Class | Tests | What's Covered |
|------------|-------|----------------|
| TestLoadFold | 4 | Shapes, feature exclusion, NaN filling, metadata |
| TestTrainer | 4 | LR/LGBM/XGB training, class weight application |
| TestEvaluator | 8 | P@K, R@K, ROC-AUC, Brier, ensemble, calibration, cross-fold |
| TestTuner | 3 | Optuna returns valid params, ROC-AUC in [0,1] |
| TestExplainer | 4 | SHAP shapes, sorted importance, player explanation, full pipeline |
| TestEvaluateFold | 1 | End-to-end: train → evaluate → predictions DataFrame |
| **Total** | **24** | **All passing** |

**Full suite: 220 passed, 4 skipped (integration), 0 failures**

## Key Design Decisions

1. **Class weights over SMOTE** — `scale_pos_weight=10` in LGBM/XGB is simpler and avoids synthetic data artifacts with ~5% positive rate
2. **Tune on fold 1, apply params to all folds** — Avoids N*trials compute cost while still using walk-forward splits
3. **Ensemble then calibrate** — Weighted average (0.5/0.5) of LGBM+XGB, then isotonic calibration on val set
4. **Four focused modules** — Each independently testable, no monolith
5. **`l1_ratio=0` instead of `penalty="l2"`** — scikit-learn 1.8 deprecated `penalty` param
6. **Handle 2D and 3D SHAP values** — TreeExplainer sometimes returns (n_samples, n_features, n_outputs)

## Output Files (when run on real data)

```
outputs/models/
├── baseline_logistic.joblib       # LR + scaler
├── lgbm_fold{1,2,3}.joblib       # LightGBM per fold
├── xgb_fold{1,2,3}.joblib        # XGBoost per fold
├── calibrator_fold{1,2,3}.joblib # Isotonic calibrators
├── best_params_lgbm.json         # Tuned hyperparameters
├── best_params_xgb.json
├── evaluation_results.json       # All metrics per fold per model
├── predictions_test.csv          # player_id, prob, label, fold
├── shap_values_fold{1,2,3}.npz  # SHAP arrays
└── feature_importance.csv        # Aggregated SHAP importance
```

## Dependencies Added

- lightgbm 4.6.0
- xgboost 3.1.3
- optuna 4.7.0
- shap 0.50.0
- scikit-learn 1.8.0
- joblib 1.5.3
- pyarrow 23.0.0

## Issues Encountered

1. **Missing pyarrow** — Tests writing parquet in tmp_path failed until pyarrow was installed
2. **scikit-learn 1.8 deprecation** — `penalty` param in LogisticRegression deprecated; switched to `l1_ratio=0`
3. **SHAP 3D values** — Some tree models return shape (n, features, outputs) instead of (n, features); added handling for both

## Git Commits

- `0fba2d8` — Add Phase 3: model training, evaluation, tuning, and SHAP explanations

## CLI Usage Examples

```bash
# Train with full tuning (100 trials)
python -m src.data.cli train --input data/processed --output outputs/models

# Train with fewer trials
python -m src.data.cli train --n-trials 20

# Train without tuning (use base params)
python -m src.data.cli train --skip-tuning

# View evaluation results
python -m src.data.cli evaluate --output outputs/models
```

## Next Steps (Phase 4)

1. **Reporting & Dashboard**
   - Player scouting reports with SHAP explanations
   - Similar successful players for context
   - Confidence-ranked shortlists

2. **Production Pipeline**
   - Scheduled scraping + retraining
   - Model versioning and drift detection

3. **Validation on Real Data**
   - Run full pipeline on scraped data
   - Check ROC-AUC > 0.75, Precision@20 > 0.30 targets
   - Inspect top predictions for face validity

---

*Phase 3 complete. Phases 1-3 (scrapers → features → models) form a complete ML pipeline.*
