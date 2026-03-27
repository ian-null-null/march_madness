from typing import Dict, Tuple

import numpy as np
import optuna as opt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from .config import EloConfig


class EloSystem:
    """Handles Elo rating calculations and optimizations."""

    def __init__(self, config: EloConfig):
        self.config = config

    def update_rating(self, w_elo: float, l_elo: float, w_loc: str, margin: int) -> Tuple[float, float]:
        if w_loc == "H":
            w_adj = w_elo + self.config.home_court_advantage
        elif w_loc == "A":
            w_adj = w_elo - self.config.home_court_advantage
        else:
            w_adj = w_elo

        elo_diff = w_adj - l_elo
        exp_w = 1.0 / (1.0 + 10 ** (-elo_diff / self.config.scaler))

        mult = (np.log(abs(margin) + 1) * self.config.margin_multiplier) / (
            0.001 * abs(elo_diff) + self.config.margin_multiplier
        )

        change = self.config.k_factor * mult * (1.0 - exp_w)
        return w_elo + change, l_elo - change

    def compute_season_ratings(self, df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
        df = df.sort_values(["Season", "DayNum"])
        elo = {}
        season_elos = {}
        prev_season = None

        for row in df.itertuples(index=False):
            season = row.Season
            if season != prev_season:
                if prev_season is not None:
                    for tid, rating in elo.items():
                        season_elos[(prev_season, tid)] = rating

                    elo = {
                        tid: self.config.season_regression * r
                        + (1 - self.config.season_regression) * self.config.initial_rating
                        for tid, r in elo.items()
                    }
                prev_season = season

            w_id = row.WTeamID
            l_id = row.LTeamID
            margin = row.WScore - row.LScore
            w_elo = elo.get(w_id, self.config.initial_rating)
            l_elo = elo.get(l_id, self.config.initial_rating)
            elo[w_id], elo[l_id] = self.update_rating(w_elo, l_elo, row.WLoc, margin)

        if prev_season is not None:
            for tid, rating in elo.items():
                season_elos[(prev_season, tid)] = rating

        return season_elos

    def optimize_parameters(
        self,
        reg_df: pd.DataFrame,
        tourney_df: pd.DataFrame,
        seeds_df: pd.DataFrame,
        feature_engineer,
        year_start: int,
        year_end: int,
        n_trials: int = 30,
    ) -> EloConfig:
        def objective(trial):
            trial_config = EloConfig(
                k_factor=trial.suggest_float("k_factor", 10, 50),
                home_court_advantage=trial.suggest_float("home_court_advantage", 50, 150),
                season_regression=trial.suggest_float("season_regression", 0.5, 0.9),
                margin_multiplier=trial.suggest_float("margin_multiplier", 1.5, 3.0),
                scaler=trial.suggest_float("scaler", 200, 800),
            )

            trial_elo_system = EloSystem(trial_config)
            trial_elos = trial_elo_system.compute_season_ratings(reg_df)

            trial_reg_seasons = {}
            for year in range(year_start, year_end + 1):
                trial_reg_seasons[year] = feature_engineer.compute_season_stats(year, trial_elos, reg_df, seeds_df)

            val_scores = []
            for season in range(year_start + 1, year_end):
                val_games = tourney_df.query("Season == @season")
                if val_games.empty:
                    continue

                x_train_cv_list, y_train_cv_list = [], []
                train_df = tourney_df.query("Season < @season")

                for train_season in sorted(train_df.Season.unique()):
                    train_games = train_df.query("Season == @train_season")
                    reg_stats = trial_reg_seasons[train_season]
                    x_t, y_t = feature_engineer.create_decision_matrix(reg_stats, train_games)
                    x_train_cv_list.append(x_t)
                    y_train_cv_list.append(y_t)

                if not x_train_cv_list:
                    continue

                x_train_cv = np.vstack(x_train_cv_list)
                y_train_cv = np.hstack(y_train_cv_list)
                y_train_cv[y_train_cv == -1] = 0

                x_val, y_val = feature_engineer.create_decision_matrix(trial_reg_seasons[season], val_games)

                scaler = StandardScaler()
                x_train_cv = scaler.fit_transform(x_train_cv)
                x_val = scaler.transform(x_val)
                lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
                lr.fit(x_train_cv, y_train_cv)
                preds = lr.predict_proba(x_val)[:, 1]
                brier = brier_score_loss(y_val, preds)
                val_scores.append(brier)

            return np.mean(val_scores) if val_scores else 1.0

        print("Optimizing Elo parameters...")
        study = opt.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best Elo params: {best_params}")
        return EloConfig(**best_params)
