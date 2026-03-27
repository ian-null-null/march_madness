from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.linear_model import Ridge

from .config import ModelConfig


class FeatureEngineer:
    """Handles feature engineering for team statistics."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    @staticmethod
    def get_pair_feature_count(n_features_per_team: int, seed_idx: Optional[int] = None) -> int:
        if seed_idx is not None and 0 <= seed_idx < n_features_per_team:
            return 3 * n_features_per_team - 2
        return 3 * n_features_per_team

    @staticmethod
    def build_pair_features(
        team_a_features: np.ndarray,
        team_b_features: np.ndarray,
        eps: float = 1e-6,
        seed_idx: Optional[int] = None,
    ) -> np.ndarray:
        diff = team_a_features - team_b_features

        if seed_idx is not None and 0 <= seed_idx < len(team_a_features):
            keep_mask = np.ones(len(team_a_features), dtype=bool)
            keep_mask[seed_idx] = False
            team_a_inter = team_a_features[keep_mask]
            team_b_inter = team_b_features[keep_mask]
        else:
            team_a_inter = team_a_features
            team_b_inter = team_b_features

        ratio = team_a_inter / (team_b_inter + eps)
        signed_log_diff = np.sign(team_a_inter) * np.log1p(np.abs(team_a_inter)) - np.sign(team_b_inter) * np.log1p(
            np.abs(team_b_inter)
        )
        return np.concatenate([diff, ratio, signed_log_diff])

    @staticmethod
    def get_pair_feature_names(columns: List[str], seed_col: str = "Seed") -> List[str]:
        inter_cols = [col for col in columns if col != seed_col]
        return [f"{col}_diff" for col in columns] + [f"{col}_ratio" for col in inter_cols] + [
            f"{col}_signed_log_diff" for col in inter_cols
        ]

    def compute_season_stats(
        self,
        year: int,
        elos: Dict[Tuple[int, int], float],
        df: pd.DataFrame,
        seeds: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = df.sort_values(["Season", "DayNum"]).query(f"Season == {year}")
        df = df.copy()

        df["WPoss"] = df.WFGA - df.WOR + df.WTO + 0.475 * df.WFTA
        df["LPoss"] = df.LFGA - df.LOR + df.LTO + 0.475 * df.LFTA
        df["W_off_eff"] = 100 * df.WScore / df.WPoss
        df["L_off_eff"] = 100 * df.LScore / df.LPoss
        df["W_def_eff"] = 100 * df.LScore / df.LPoss
        df["L_def_eff"] = 100 * df.WScore / df.WPoss
        df["W_true_shoot"] = df.WScore / (2 * (df.WFGA + 0.475 * df.WFTA))
        df["L_true_shoot"] = df.LScore / (2 * (df.LFGA + 0.475 * df.LFTA))
        df["W_off_rebound"] = df.WOR / (df.WOR + df.LDR)
        df["L_off_rebound"] = df.LOR / (df.LOR + df.WDR)
        df["W_def_rebound"] = df.WDR / (df.WDR + df.LDR)
        df["L_def_rebound"] = df.LDR / (df.LDR + df.WDR)
        df["W_turnover"] = df.WTO / df.WPoss
        df["L_turnover"] = df.LTO / df.LPoss
        df["W_free_throw"] = df.WFTM / df.WFTA
        df["L_free_throw"] = df.LFTM / df.LFTA
        df["W_3pt_pct"] = df.WFGM3 / (df.WFGA3 + 1e-6)
        df["L_3pt_pct"] = df.LFGM3 / (df.LFGA3 + 1e-6)

        teams = pd.concat([df.WTeamID, df.LTeamID]).unique()
        teams = np.sort(teams)
        team_to_idx = {int(team): i for i, team in enumerate(teams)}

        n_teams = len(teams)
        n_games = len(df)
        x = lil_matrix((2 * n_games, 2 * n_teams))
        y = np.zeros(2 * n_games)

        row = 0
        for game in df.itertuples():
            w_idx = team_to_idx[game.WTeamID]
            l_idx = team_to_idx[game.LTeamID]
            x[row, w_idx] = 1
            x[row, n_teams + l_idx] = -1
            y[row] = game.W_off_eff
            row += 1

            x[row, l_idx] = 1
            x[row, n_teams + w_idx] = -1
            y[row] = game.L_off_eff
            row += 1

        model = Ridge(alpha=self.model_config.ridge_alpha, fit_intercept=True)
        model.fit(x, y)

        coefs = model.coef_
        metrics = pd.DataFrame({"Off_rate": coefs[:n_teams], "Def_rate": coefs[n_teams:]}, index=teams)

        years = {s for s, _ in elos.keys()}
        elo_dict = {season: {} for season in years}
        for (season, team_id), rating in elos.items():
            elo_dict[season][team_id] = rating
        elo_df = pd.Series(elo_dict[year]).to_frame().rename(columns={0: "Elo"})

        w_shoot = df[["WTeamID", "W_true_shoot"]].copy()
        w_shoot.columns = ["TeamID", "true_shoot"]
        l_shoot = df[["LTeamID", "L_true_shoot"]].copy()
        l_shoot.columns = ["TeamID", "true_shoot"]
        shoot_stats = pd.concat([w_shoot, l_shoot]).groupby("TeamID")["true_shoot"].median().to_frame()

        wins_long = df[["WTeamID", "DayNum"]].rename(columns={"WTeamID": "TeamID"}).assign(win=1)
        losses_long = df[["LTeamID", "DayNum"]].rename(columns={"LTeamID": "TeamID"}).assign(win=0)
        all_games_long = pd.concat([wins_long, losses_long])
        last10_win_pct = (
            all_games_long.groupby("TeamID")[["DayNum", "win"]]
            .apply(lambda g: g.sort_values("DayNum").tail(10)["win"].mean())
            .rename("last10_win_pct")
        )

        w_pace = df[["WTeamID", "WPoss"]].rename(columns={"WTeamID": "TeamID", "WPoss": "pace"})
        l_pace = df[["LTeamID", "LPoss"]].rename(columns={"LTeamID": "TeamID", "LPoss": "pace"})
        avg_pace = pd.concat([w_pace, l_pace]).groupby("TeamID")["pace"].mean().rename("avg_pace").to_frame()

        w_3pt = df[["WTeamID", "W_3pt_pct"]].rename(columns={"WTeamID": "TeamID", "W_3pt_pct": "three_pt_pct"})
        l_3pt = df[["LTeamID", "L_3pt_pct"]].rename(columns={"LTeamID": "TeamID", "L_3pt_pct": "three_pt_pct"})
        three_pt_var = pd.concat([w_3pt, l_3pt]).groupby("TeamID")["three_pt_pct"].std().rename("three_pt_var").to_frame()

        w_to = df[["WTeamID", "W_turnover"]].rename(columns={"WTeamID": "TeamID", "W_turnover": "turnover_rate"})
        l_to = df[["LTeamID", "L_turnover"]].rename(columns={"LTeamID": "TeamID", "L_turnover": "turnover_rate"})
        turnover_chaos = pd.concat([w_to, l_to]).groupby("TeamID")["turnover_rate"].std().rename("turnover_chaos").to_frame()

        w_sv = df[["WTeamID", "W_true_shoot"]].rename(columns={"WTeamID": "TeamID", "W_true_shoot": "true_shoot_vol"})
        l_sv = df[["LTeamID", "L_true_shoot"]].rename(columns={"LTeamID": "TeamID", "L_true_shoot": "true_shoot_vol"})
        shoot_volatility = pd.concat([w_sv, l_sv]).groupby("TeamID")["true_shoot_vol"].std().rename("shoot_volatility").to_frame()

        result = metrics.join(
            [elo_df, shoot_stats, last10_win_pct, avg_pace, three_pt_var, turnover_chaos, shoot_volatility],
            how="left",
        )
        result = result.fillna(result.median(numeric_only=True))

        if seeds is not None:
            season_seeds = seeds[seeds["Season"] == year][["TeamID", "Seed"]].copy().set_index("TeamID")
            result = result.join(season_seeds, how="left")
        else:
            result["Seed"] = np.nan

        result["Seed"] = np.log1p(result["Seed"].fillna(17))
        return result

    def create_decision_matrix(self, reg_stats: pd.DataFrame, tour_games: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n_games = len(tour_games)
        n_features_per_team = len(reg_stats.columns)
        seed_idx = reg_stats.columns.get_loc("Seed") if "Seed" in reg_stats.columns else None
        n_total_features = self.get_pair_feature_count(n_features_per_team, seed_idx)

        x = np.zeros((2 * n_games, n_total_features))
        y = np.zeros(2 * n_games)

        winner_ids = tour_games["WTeamID"].values
        loser_ids = tour_games["LTeamID"].values

        for i, (winner_id, loser_id) in enumerate(zip(winner_ids, loser_ids)):
            w_features = reg_stats.loc[winner_id].values
            l_features = reg_stats.loc[loser_id].values
            x[2 * i] = self.build_pair_features(w_features, l_features, seed_idx=seed_idx)
            y[2 * i] = 1
            x[2 * i + 1] = self.build_pair_features(l_features, w_features, seed_idx=seed_idx)
            y[2 * i + 1] = 0

        return x, y
