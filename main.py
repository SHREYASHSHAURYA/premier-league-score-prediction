import joblib
import pandas as pd
import numpy as np
from scipy.stats import poisson

df_feat = pd.read_csv("data/processed/features.csv")

pois_home = joblib.load("results/poisson_home_model.pkl")
pois_away = joblib.load("results/poisson_away_model.pkl")


def predict_match(home_team, away_team, max_goals=6):
    home_matches = df_feat[
        (df_feat["HomeTeam"] == home_team) | (df_feat["AwayTeam"] == home_team)
    ]
    away_matches = df_feat[
        (df_feat["HomeTeam"] == away_team) | (df_feat["AwayTeam"] == away_team)
    ]

    if len(home_matches) == 0 or len(away_matches) == 0:
        raise ValueError("Team name not found. Use exact dataset names.")

    home_last = home_matches.iloc[-1]
    away_last = away_matches.iloc[-1]

    def get(row, col, default=0.0):
        return row[col] if col in row.index else default

    X = pd.DataFrame([{
        "home_goals_last5": get(home_last, "home_goals_last5"),
        "away_goals_last5": get(away_last, "away_goals_last5"),
        "home_conceded_last5": get(home_last, "home_conceded_last5"),
        "away_conceded_last5": get(away_last, "away_conceded_last5"),
        "home_points": get(home_last, "home_points"),
        "away_points": get(away_last, "away_points"),
        "home_gd_last5": get(home_last, "home_gd_last5"),
        "away_gd_last5": get(away_last, "away_gd_last5"),
    }])

    home_xg = float(pois_home.predict(X)[0])
    away_xg = float(pois_away.predict(X)[0])

    home_probs = poisson.pmf(range(max_goals + 1), home_xg)
    away_probs = poisson.pmf(range(max_goals + 1), away_xg)

    win = draw = loss = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = home_probs[i] * away_probs[j]
            if i > j:
                win += p
            elif i == j:
                draw += p
            else:
                loss += p

    return {
        "predicted_score": f"{round(home_xg)} - {round(away_xg)}",
        "home_xG": round(home_xg, 2),
        "away_xG": round(away_xg, 2),
        "home_win_prob": round(win, 3),
        "draw_prob": round(draw, 3),
        "away_win_prob": round(loss, 3),
    }


if __name__ == "__main__":
    home = input("Home team: ")
    away = input("Away team: ")

    result = predict_match(home, away)

    print("\nPrediction")
    for k, v in result.items():
        print(f"{k}: {v}")
