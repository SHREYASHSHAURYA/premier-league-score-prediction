# Premier League Match Score Prediction (Machine Learning)

This project predicts English Premier League (EPL) football match outcomes using machine learning. It estimates expected goals for home and away teams, predicts a likely scoreline, and computes win, draw, and loss probabilities. The system is designed as a clean, reproducible ML project with both experimentation notebooks and a runnable command-line interface.

---

## Project Overview

Football match score prediction is challenging due to inherent randomness, limited data per match, and constantly changing team form. Instead of attempting exact score classification, this project models goals scored by each team using regression techniques combined with Poisson distributions.

The scope is intentionally limited to a single league (EPL) to avoid cross-league inconsistencies and maintain data quality.

---

## Key Features

- Uses more than 10 seasons of EPL historical data  
- Time-aware feature engineering to prevent data leakage  
- Rolling team form features based on the last five matches  
- Poisson regression for modeling goal counts  
- Win, draw, and loss probability estimation  
- Command-line interface for match predictions  

---

## Project Structure

```
premier-league-score-prediction/
├── data/
│   ├── raw/                  # merged raw match data
│   └── processed/            # cleaned and feature-engineered data
├── notebooks/
│   └── 01_data_exploration.ipynb
├── results/
│   ├── poisson_home_model.pkl
│   └── poisson_away_model.pkl
├── main.py                   # CLI prediction script
├── requirements.txt
└── README.md
```

---

## Data

The dataset consists of publicly available English Premier League match data covering multiple seasons. Each match includes:

- Full-time home goals (FTHG)
- Full-time away goals (FTAG)
- Home and away team names
- Match date

All seasons are merged into a single dataset prior to processing.

---

## Feature Engineering

All features are computed using only information available **before** each match to avoid data leakage.

The final feature set includes rolling averages over the previous five matches for each team, such as:

- Goals scored
- Goals conceded
- Points per match
- Goal difference

Matches without sufficient historical data are removed to ensure valid feature values.

---

## Models

A baseline linear regression model is used to establish reference performance.

The final system uses two Poisson regression models:
- One for predicting home team goals
- One for predicting away team goals

The predicted expected goals (xG) are converted into full probability distributions over possible scorelines.

---

## Evaluation

Model performance is evaluated using Mean Absolute Error (MAE) on predicted goals.

Typical results:
- Home goals MAE ≈ 1.0
- Away goals MAE ≈ 0.9

Exact score prediction is not the primary objective. Correct outcome classification (win, draw, loss) and well-calibrated probabilities are considered more important.

---

## Running the Project

### Installation

Install dependencies using:

```
pip install -r requirements.txt
```

### Prediction

Run the command-line predictor:

```
python main.py
```

---

## Example

**Input**
```
Home team: Fulham
Away team: Everton
```

**Output**
```
predicted_score : 2 - 1
home_xG         : 1.64
away_xG         : 1.17
home_win_prob   : 0.481
draw_prob       : 0.245
away_win_prob   : 0.273
```

---

## Interpretation

The predicted score is derived from expected goals, while the probabilities reflect uncertainty in match outcomes.

A correct prediction is judged primarily by the outcome class (win, draw, or loss), not by the exact scoreline. This mirrors how professional football analytics models are evaluated.

---

## Limitations

- No player-level data (injuries, lineups, transfers)
- No betting odds or external contextual features
- Predictions depend on the most recent match data available

---

## Future Work

- Incorporate player availability and squad strength
- Update data on a rolling basis
- Add odds-based calibration
- Deploy as a web API or lightweight frontend

---

## Disclaimer

This project is intended for educational and analytical purposes only. It is **not** designed or recommended for gambling or betting applications.
