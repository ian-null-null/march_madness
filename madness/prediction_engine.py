from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .feature_engineer import FeatureEngineer


class PredictionEngine:
    """Handles generating predictions for tournament matchups."""

    def __init__(self, model: xgb.Booster, reg_seasons: Dict[int, pd.DataFrame]):
        self.model = model
        self.reg_seasons = reg_seasons

    def predict_matchups(self, sample_df: pd.DataFrame, team_ids: List[int], upset_factor: float = 0.0) -> List[Tuple[str, float]]:
        valid_rows = []
        ids = []

        for row in sample_df.itertuples():
            matchup_id = row.ID
            season, team_a, team_b = [int(x) for x in matchup_id.split("_")]
            if team_a in team_ids and team_b in team_ids:
                valid_rows.append((season, team_a, team_b, matchup_id))
                ids.append(matchup_id)

        if not valid_rows:
            return []

        first_season_stats = next(iter(self.reg_seasons.values()))
        n_features_per_team = len(first_season_stats.columns)
        seed_idx = first_season_stats.columns.get_loc("Seed") if "Seed" in first_season_stats.columns else None
        n_total_features = FeatureEngineer.get_pair_feature_count(n_features_per_team, seed_idx)
        x_test = np.zeros((len(valid_rows), n_total_features))

        for i, (season, team_a, team_b, _) in enumerate(valid_rows):
            team_a_features = self.reg_seasons[season].loc[team_a].values
            team_b_features = self.reg_seasons[season].loc[team_b].values
            x_test[i] = FeatureEngineer.build_pair_features(team_a_features, team_b_features, seed_idx=seed_idx)

        d_test = xgb.DMatrix(x_test)
        predictions = self.model.predict(d_test)
        predictions = self.calibrate_upset_bias(predictions, upset_factor=upset_factor)
        return list(zip(ids, predictions))

    @staticmethod
    def calibrate_upset_bias(predictions: np.ndarray, upset_factor: float = 0.0) -> np.ndarray:
        if upset_factor == 0:
            return predictions
        return predictions + (0.5 - predictions) * upset_factor
