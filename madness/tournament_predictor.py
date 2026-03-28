import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import DataPaths, EloConfig, ModelConfig, TrainingConfig
from .data_loader import DataLoader
from .elo import EloSystem
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .prediction_engine import PredictionEngine


class TournamentPredictor:
    """Main class for predicting March Madness tournament outcomes."""

    FROZEN_ELO_PARAMS = {
        "mens": {
            "k_factor": 49.252870951541276,
            "home_court_advantage": 50.28426142114505,
            "season_regression": 0.893261901459784,
            "margin_multiplier": 2.168182260261729,
            "scaler": 578.4503964798329,
        },
        "womens": {
            "k_factor": 40.876708626893155,
            "home_court_advantage": 135.06784812094847,
            "season_regression": 0.8952883368506546,
            "margin_multiplier": 2.6058070207391553,
            "scaler": 313.57223189638535,
        },
    }

    FROZEN_XGB_PARAMS = {
        "mens": {
            "eta": 0.01627512899991115,
            "max_depth": 4,
            "subsample": 0.8590270371967373,
            "colsample_bytree": 0.7247868667203111,
            "lambda": 0.43187812146857757,
            "alpha": 0.5914487717584811,
            "scale_pos_weight": 0.7595138861397499,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        "womens": {
            "eta": 0.021419679163005176,
            "max_depth": 3,
            "subsample": 0.644000454284833,
            "colsample_bytree": 0.633480883000547,
            "lambda": 1.7979825580114568,
            "alpha": 0.324633708022161,
            "scale_pos_weight": 0.7983007380367149,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
    }

    def __init__(
        self,
        elo_config: Optional[EloConfig] = None,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        data_paths: Optional[DataPaths] = None,
    ):
        self.elo_config = elo_config or EloConfig()
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.data_paths = data_paths or DataPaths()

        self.data_loader = DataLoader(self.data_paths)
        self.elo_system = EloSystem(self.elo_config)
        self.feature_engineer = FeatureEngineer(self.model_config)
        self.model_trainer = ModelTrainer(self.model_config, self.training_config)

    def train_gender(
        self,
        gender: str,
        optimize_elo: bool = False,
        optimize_xgb: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple:
        print(f"Processing {gender} tournament...")

        reg_df = self.data_loader.load_regular_season(gender)
        tourney_df = self.data_loader.load_tournament(gender)
        seeds_df = self.data_loader.load_seeds(gender)

        year_start = self.training_config.mens_start_year if gender == "mens" else self.training_config.womens_start_year
        year_end = self.training_config.end_year

        if optimize_elo:
            elo_system = self.elo_system
        else:
            frozen_elo = EloConfig(**self.FROZEN_ELO_PARAMS[gender])
            elo_system = EloSystem(frozen_elo)
            print(f"Skipping Optuna, using frozen {gender} Elo params")

        elos = elo_system.compute_season_ratings(reg_df)

        reg_seasons = {}
        for year in range(year_start, year_end + 1):
            reg_seasons[year] = self.feature_engineer.compute_season_stats(year, elos, reg_df, seeds_df)

        train_df = tourney_df.query("Season < @year_end")
        x_train_list, y_train_list = [], []

        for season in sorted(train_df.Season.unique()):
            games = train_df.query("Season == @season")
            reg_stats = reg_seasons[season]
            x_t, y_t = self.feature_engineer.create_decision_matrix(reg_stats, games)
            x_train_list.append(x_t)
            y_train_list.append(y_t)

        x_train = np.vstack(x_train_list)
        y_train = np.hstack(y_train_list)
        y_train[y_train == -1] = 0

        if optimize_elo:
            optimized_elo_config = elo_system.optimize_parameters(
                reg_df,
                tourney_df,
                seeds_df,
                self.feature_engineer,
                year_start,
                year_end,
                self.training_config.elo_trials,
            )
            elo_system = EloSystem(optimized_elo_config)
            elos = elo_system.compute_season_ratings(reg_df)
            reg_seasons = {}
            for year in range(year_start, year_end + 1):
                reg_seasons[year] = self.feature_engineer.compute_season_stats(year, elos, reg_df, seeds_df)

        if optimize_xgb:
            best_xgb_params = self.model_trainer.optimize_xgb_params(x_train, y_train)
        else:
            best_xgb_params = self.FROZEN_XGB_PARAMS[gender].copy()
            print(f"Skipping Optuna, using frozen {gender} XGBoost params")

        _, xgb_model = self.model_trainer.train_models(x_train, y_train, best_xgb_params)

        validation_results = self.model_trainer.validate_models(
            reg_seasons, tourney_df, self.feature_engineer, year_start, year_end
        )

        cols = reg_stats.columns.tolist()
        feature_names = self.feature_engineer.get_pair_feature_names(cols, seed_col="Seed")
        importance = xgb_model.get_score(importance_type="gain")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        impactful_features: Dict[str, float] = {}
        #print(f"\nMost impactful features for {gender}:")
        for f, score in sorted_importance:
            idx = int(f[1:])
            feature_name = feature_names[idx]
            impactful_features[feature_name] = float(score)
            #print(f"{feature_name}: {score:.4f}")

        # Keep latest diagnostics accessible even when not explicitly returned.
        if not hasattr(self, "last_validation_results"):
            self.last_validation_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
        if not hasattr(self, "last_impactful_features"):
            self.last_impactful_features: Dict[str, Dict[str, float]] = {}
        self.last_validation_results[gender] = validation_results
        self.last_impactful_features[gender] = impactful_features

        if return_diagnostics:
            return xgb_model, reg_seasons, validation_results, impactful_features

        return xgb_model, reg_seasons

    def predict_gender(
        self,
        gender: str,
        model: xgb.Booster,
        reg_seasons: Dict[int, pd.DataFrame],
        stage: int = 2,
    ) -> List[Tuple[str, float]]:
        team_ids = self.data_loader.load_teams(gender)
        sample_df = self.data_loader.load_sample_submission(stage)
        prediction_engine = PredictionEngine(model, reg_seasons)
        return prediction_engine.predict_matchups(sample_df, team_ids, upset_factor=self.training_config.upset_factor)

    def run_full_pipeline(self, stage: int = 2) -> None:
        start_time = time.time()
        mens_model, mens_reg_seasons = self.train_gender("mens", optimize_elo=False, optimize_xgb=False)
        womens_model, womens_reg_seasons = self.train_gender("womens", optimize_elo=False, optimize_xgb=False)

        print("Generating predictions...")
        mens_predictions = self.predict_gender("mens", mens_model, mens_reg_seasons, stage)
        womens_predictions = self.predict_gender("womens", womens_model, womens_reg_seasons, stage)

        print("Writing submission...")
        all_predictions = mens_predictions + womens_predictions
        submission_path = f"{self.data_paths.base_path}SubmissionStage{stage}.csv"

        with open(submission_path, "w") as f:
            f.write("ID,Pred\n")
            for matchup_id, pred in all_predictions:
                f.write(f"{matchup_id},{pred:.5f}\n")

        print(f"Submission written to {submission_path}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
