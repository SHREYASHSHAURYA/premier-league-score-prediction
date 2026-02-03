# Premier League Match Score Prediction (Machine Learning)

This project predicts English Premier League (EPL) football match outcomes using machine learning. It estimates expected goals for home and away teams, predicts a likely scoreline, and computes win, draw, and loss probabilities. The project is built as a clean, reproducible ML system with both experimentation (notebooks) and a runnable command-line interface.

## Project Overview

Football match score prediction is challenging due to randomness, limited data per match, and constantly changing team form. Instead of attempting exact score classification, this project models goals scored by each team using regression techniques and Poisson distributions. The scope is intentionally limited to a single league (EPL) to avoid cross-league inconsistencies and maintain data quality.

## Key Features

- Uses more than 10 seasons of EPL historical data  
- Time-aware feature engineering to prevent data leakage  
- Rolling team form features based on the last five matches  
- Poisson regression for modeling goal counts  
- Win, draw, and loss probability estimation  
- Command-line interface for live match predictions  

## Project Structure

`premier-league-score-prediction/`  
`├── data/`  
`│   ├── raw/                  # merged raw match data`  
`│   └── processed/            # cleaned and feature-engineered data`  
`├── notebooks/`  
`│   └── 01_data_exploration.ipynb`  
`├── results/`  
`│   ├── poisson_home_model.pkl`  
`│   ├── poisson_away_model.pkl`  
`├── main.py                   # CLI prediction script`  
`├── requirements.txt`  
`└── README.md`  

## Data

The dataset consists of publicly available English Premier League match data covering multiple seasons. Each match includes full-time home goals (FTHG), full-time away goals (FTAG), team names, and match dates. All seasons are merged into a single dataset before processing.

## Feature Engineering

All features are computed using only information available before each match. The final feature set includes rolling averages over the previous five matches for each team, such as goals scored, goals conceded, points per match, and goal difference. Matches without sufficient historical data are removed to ensure valid feature values.

## Models

A baseline linear regression model is used to establish a reference performance. The final system uses Poisson regression models to predict expected goals for home and away teams separately. These expected goals are then converted into full probability distributions over possible scorelines.

## Evaluation

Model performance is evaluated using Mean Absolute Error (MAE) on predicted goals. Typical results achieve approximately 1.0 MAE for home goals and 0.9 MAE for away goals. Exact score prediction is not the primary objective; correct outcome classification and calibrated probabilities are more important.

## Running the Project

Install dependencies using `pip install -r requirements.txt`.

Run the predictor using `python main.py`.

Example input:  
Home team: Fulham  
Away team: Everton  

Example output:  
predicted_score: 2 - 1  
home_xG: 1.64  
away_xG: 1.17  
home_win_prob: 0.481  
draw_prob: 0.245  
away_win_prob: 0.273  

## Interpretation

The predicted score is derived from expected goals, while the probabilities represent uncertainty in match outcomes. A correct prediction is judged primarily by the outcome class (win, draw, or loss), not by the exact scoreline. This mirrors how real-world football analytics models are evaluated.

## Limitations

The model does not account for player-level information such as injuries, lineups, or transfers. Betting odds and external contextual factors are also excluded. Predictions depend on the most recent match data available in the dataset.

## Future Work

Potential improvements include incorporating player availability, updating data on a rolling basis, adding odds-based calibration, and deploying the model as a web API or simple frontend application.

## Disclaimer

This project is intended for educational and analytical purposes only. It is not designed or recommended for gambling or betting applications.
