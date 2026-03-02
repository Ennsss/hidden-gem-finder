# Hidden Gem Finder

**An end-to-end ML system that scouts lower-tier European football leagues to predict which players will break into the top 5 leagues within 3 years.**

The system scrapes real player data from three sources, fuses them via fuzzy matching, engineers 48 features, and trains a calibrated LightGBM + XGBoost ensemble — all evaluated with walk-forward temporal cross-validation to prevent data leakage. It ships with a Streamlit scouting dashboard featuring SHAP-powered player profiles.

---

## How It Works

```
Scrape 3 sources          Fuzzy-match players        Engineer 48 features
 FBref (stats)     -->     across sources      -->    per-90, growth, league-
 Transfermarkt ($)          by name + team             adjusted, interactions
 Understat (xG)

        |                        |                          |
        v                        v                          v

 DuckDB storage           Label breakouts              Train ensemble
 80,000+ rows             (top-5 league,         -->   LightGBM + XGBoost
 10 seasons               900+ min in 3yr)             + isotonic calibration

        |                        |                          |
        v                        v                          v

 Cache raw HTML           Walk-forward folds           SHAP explanations
 for reproducibility      (no future leakage)          per-player profiles
```

---

## Results

The model achieves **73.3% precision in the top 10** predictions and **0.717 ROC-AUC** across three walk-forward folds — meaning nearly 3 out of 4 players flagged as top prospects actually broke into a top-5 league.

### Real Predictions the Model Got Right

| Player | Age | Club | Predicted | Broke Out To |
|--------|-----|------|-----------|-------------|
| Eberechi Eze | 21 | QPR | 64.6% | Crystal Palace (PL) |
| Bryan Mbeumo | 20 | Brentford | 62.0% | Brentford (PL) |
| Said Benrahma | 24 | Brentford | 60.0% | West Ham (PL) |
| Ollie Watkins | 24 | Brentford | 59.4% | Aston Villa (PL) |
| Matty Cash | 22 | Nott'm Forest | 58.9% | Aston Villa (PL) |
| Jarrod Bowen | 23 | Hull City | 58.2% | West Ham (PL) |
| Ben White | 22 | Leeds United | 56.9% | Arsenal (PL) |
| Brennan Johnson | 20 | Nott'm Forest | 55.6% | Tottenham (PL) |
| Francisco Trincao | 20 | Braga | 54.2% | Wolves (PL) |
| Charles De Ketelaere | 21 | Club Brugge | 52.8% | AC Milan (Serie A) |

44% of top false positives are "near-misses" — players who did reach a top-5 league but fell short of the 900-minute threshold.

### Cross-Fold Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.717 |
| Average Precision | 0.403 |
| Brier Score | 0.133 |
| Precision@10 | 0.733 |
| Precision@20 | 0.650 |
| Precision@50 | 0.587 |

### What Drives Predictions (Top SHAP Features)

| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| `minutes_share x age` | 0.365 | Young players who are regular starters |
| `shot_volume_above_avg` | 0.107 | More shots than league peers |
| `goals_above_avg` | 0.106 | Outscoring their league average |
| `career_avg_minutes` | 0.090 | Consistently trusted with playing time |
| `assists_above_avg` | 0.085 | Creative output above peers |
| `age` | 0.084 | Younger = higher breakout chance |

The dominant signal: **young players who play a lot of minutes are by far the strongest breakout predictor** — more than any individual skill metric.

---

## Data Sources

| Source | What It Provides | Rows | Coverage |
|--------|-----------------|------|----------|
| [FBref](https://fbref.com) | Goals, assists, shots, passes, defensive stats, possession | 44,851 | All leagues, 2015-2025 |
| [Transfermarkt](https://transfermarkt.com) | Market valuations, demographics, transfer history | 8,200 | All leagues |
| [Understat](https://understat.com) | xG, npxG, xA, key passes | 27,182 | Top-5 leagues only |

Players are linked across sources using **fuzzy matching** (rapidfuzz) on normalized names + teams within the same league and season.

---

## Feature Engineering

Starting from raw stats, the pipeline creates **48 features** through:

1. **Per-90 normalization** — All counting stats converted to per-90-minute rates
2. **Derived features** — Shot conversion, goal contribution, involvement score, age potential factor
3. **Career tracking** — Multi-season averages, trends (slope), max achievements via FBref's stable player IDs
4. **Growth metrics** — Year-over-year change in key stats (minutes, goals, progressive actions)
5. **League adjustments** — Multiply attacking stats by league difficulty coefficients (Championship 0.78, Eredivisie 0.85, etc.)
6. **Baseline comparisons** — `goals_above_avg`, `assists_above_avg` vs league-position-season means
7. **Interaction features** — `age x goal_contribution`, `shot_conversion x age`, `start_ratio x goals_above_avg`
8. **Proxy xG model** — A GBM trained on Understat data to estimate xG for leagues without Understat coverage
9. **Selection** — Variance filter + correlation filter (>0.90) reduces ~150 candidates to 48

---

## Model Architecture

```
                    +-----------+
Training Data ----->| LightGBM  |----+
(48 features,       +-----------+    |     Weighted Average     +-------------+
 class-weighted)                     +---->  (50/50)  --------> | Platt       |---> Calibrated
                    +-----------+    |                          | Calibration |     Probability
                --->| XGBoost   |----+                          +-------------+
                    +-----------+
```

- **Class imbalance**: ~12.8% positive rate, handled via auto-computed `scale_pos_weight`
- **Hyperparameter tuning**: Optuna Bayesian search (100 trials per model), optimizing ROC-AUC
- **Calibration**: Platt scaling fit on validation set ensures probabilities are well-calibrated
- **Validation**: 3-fold walk-forward (train through 2017-20, test 2019-22) — never trains on future data

---

## Project Structure

```
hidden-gem-finder/
├── config/
│   ├── leagues.yaml           # Source/target leagues, difficulty coefficients
│   ├── features.yaml          # Feature definitions and selection thresholds
│   └── model_params.yaml      # Training params, fold config, tuning spaces
├── dashboard/
│   ├── app.py                 # Streamlit entry point
│   └── pages/                 # Rankings, Player Profile, Model Insights, Scout
├── scripts/
│   ├── scrape_all.py          # Batch scraping across leagues/seasons
│   ├── predict_current.py     # Score current-season players
│   └── validate_predictions.py# Generate validation report
├── src/
│   ├── scrapers/
│   │   ├── base_scraper.py    # Rate-limiting, caching, retry, UA rotation
│   │   ├── fbref_scraper.py   # SeleniumBase UC mode (Cloudflare bypass)
│   │   ├── transfermarkt_scraper.py
│   │   └── understat_scraper.py
│   ├── storage/
│   │   └── database.py        # DuckDB schema + CRUD operations
│   ├── data/
│   │   ├── cli.py             # All CLI commands
│   │   ├── cleaning.py        # Position normalization, age/minutes filters
│   │   ├── matching.py        # Fuzzy cross-source player matching
│   │   ├── labeling.py        # Breakout labels + temporal splits
│   │   └── pipeline.py        # Orchestrates Phase 2 & 3
│   ├── features/
│   │   ├── engineering.py     # Per-90, derived, growth, interactions
│   │   ├── selection.py       # Variance + correlation filtering
│   │   └── proxy_xg.py       # GBM-based xG estimation
│   └── models/
│       ├── trainer.py         # LightGBM, XGBoost, logistic baseline
│       ├── evaluator.py       # Precision@K, ROC-AUC, ensemble, calibration
│       ├── tuner.py           # Optuna hyperparameter search
│       ├── explainer.py       # SHAP TreeExplainer
│       └── predictor.py       # Inference on new data
├── tests/                     # 223 tests (79 scraping, 117 features, 24 models)
├── outputs/
│   ├── models/                # Trained models, SHAP values, evaluation results
│   └── validation_report.md   # Human-readable model audit
└── data/
    ├── players.duckdb         # Main database
    ├── raw/                   # Cached HTML/JSON from scrapers
    └── processed/             # Parquet files (enriched, labeled, fold splits)
```

---

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Check the data
python -m src.data.cli stats

# Run the full pipeline
python -m src.data.cli process              # Feature engineering + labeling
python -m src.data.cli train --n-trials 100 # Train with Optuna tuning

# Score current players
python scripts/predict_current.py

# Launch the dashboard
streamlit run dashboard/app.py
```

### CLI Reference

```bash
# Scraping
python -m src.data.cli scrape --league championship --season 2024-2025
python -m src.data.cli scrape-all --season 2024-2025

# Data processing
python -m src.data.cli process              # Run Phase 2 pipeline
python -m src.data.cli label-stats          # View breakout label distribution

# Training
python -m src.data.cli train                # Train (skip tuning, use saved params)
python -m src.data.cli train --n-trials 100 # Train with Optuna tuning
python -m src.data.cli evaluate             # Print saved evaluation metrics

# Utilities
python -m src.data.cli stats               # Database row counts
python -m src.data.cli leagues              # List supported leagues
python -m src.data.cli validate             # Data quality checks
```

---

## Scouting Dashboard

Launch with `streamlit run dashboard/app.py` — 5 pages:

| Page | What It Shows |
|------|--------------|
| **Overview** | Summary metrics, dataset stats, project description |
| **Player Rankings** | Filterable table — filter by league, position, season, age, probability |
| **Player Profile** | SHAP waterfall chart, radar chart vs peers, similar player search |
| **Model Insights** | ROC curves, precision-recall, calibration plot, per-league AUC breakdown |
| **Scout Player** | Scout-focused workflow for evaluating specific prospects |

---

## Walk-Forward Validation

The model is evaluated using temporal walk-forward splits to guarantee no future data leaks into training:

| Fold | Train Through | Validation | Test |
|------|--------------|------------|------|
| 1 | 2017-18 | 2018-19 | 2019-20 |
| 2 | 2018-19 | 2019-20 | 2020-21 |
| 3 | 2019-20 | 2020-21 | 2021-22 |

---

## Scouted Leagues

**Source leagues** (where we look for hidden gems):

| League | Country | Difficulty Coefficient |
|--------|---------|----------------------|
| Championship | England | 0.78 |
| Eredivisie | Netherlands | 0.85 |
| Primeira Liga | Portugal | 0.82 |
| Belgian Pro League | Belgium | 0.75 |
| Serie B | Italy | 0.72 |
| Ligue 2 | France | 0.70 |
| Austrian Bundesliga | Austria | 0.68 |
| Scottish Premiership | Scotland | 0.65 |

**Target leagues** (breakout destinations): Premier League, La Liga, Bundesliga, Serie A, Ligue 1

A player is labeled as a **breakout** if they accumulate 900+ minutes in any target league within 3 years.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML** | LightGBM, XGBoost, scikit-learn, Optuna, SHAP |
| **Data** | pandas, DuckDB, pyarrow |
| **Scraping** | SeleniumBase (Cloudflare bypass), requests, BeautifulSoup, rapidfuzz |
| **Dashboard** | Streamlit, Plotly |
| **Testing** | pytest (223 tests), pytest-mock |
| **Config** | YAML (leagues, features, model params) |

---

## Known Limitations

- **FBref xG is paywalled** — xG/npxG/xA stats return NULL for all players. Mitigated with a proxy xG model trained on Understat data.
- **Understat covers top-5 leagues only** — Feeder-league training data has 0% Understat xG coverage, so xG features are dropped by the variance filter during training.
- **Label strictness** — Requiring 900+ minutes means short loan spells to top-5 leagues don't count as breakouts, creating "near-miss" false positives.
- **Scraping fragility** — FBref requires SeleniumBase UC mode for Cloudflare bypass; site structure changes may break parsers.

---

## License

Private project — not for distribution.
