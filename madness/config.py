from dataclasses import dataclass
from typing import Dict


@dataclass
class EloConfig:
    """Configuration for Elo rating system parameters."""

    k_factor: float = 20.0
    initial_rating: float = 1500.0
    home_court_advantage: float = 100.0
    season_regression: float = 0.75
    margin_multiplier: float = 2.2
    scaler: float = 400.0


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""

    ridge_alpha: float = 10.0
    xgb_params: Dict = None
    xgb_rounds: int = 400
    xgb_early_stop: int = 50
    logit_c: float = 1.0
    logit_max_iter: int = 1000
    cv_folds: int = 5
    scale_pos_weight: float = 0.5
    use_balanced_logit: bool = True

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "eta": 0.03,
                "max_depth": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambda": 1.0,
                "alpha": 0.5,
                "scale_pos_weight": self.scale_pos_weight,
                "seed": 9001,
            }


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    mens_start_year: int = 2003
    womens_start_year: int = 2010
    end_year: int = 2026
    pred_start_year: int = 2022
    pred_end_year: int = 2026
    elo_trials: int = 30
    xgb_trials: int = 50
    upset_factor: float = 0.15


@dataclass
class DataPaths:
    """Configuration for data file paths."""

    base_path: str = "data/"
    mens_reg: str = "MRegularSeasonDetailedResults.csv"
    mens_tourney: str = "MNCAATourneyDetailedResults.csv"
    womens_reg: str = "WRegularSeasonDetailedResults.csv"
    womens_tourney: str = "WNCAATourneyDetailedResults.csv"
    mens_teams: str = "MTeams.csv"
    womens_teams: str = "WTeams.csv"
    mens_seeds: str = "MNCAATourneySeeds.csv"
    womens_seeds: str = "WNCAATourneySeeds.csv"
    sample_submission_stage1: str = "SampleSubmissionStage1.csv"
    sample_submission_stage2: str = "SampleSubmissionStage2.csv"

    @property
    def mens_reg_path(self) -> str:
        return f"{self.base_path}{self.mens_reg}"

    @property
    def mens_tourney_path(self) -> str:
        return f"{self.base_path}{self.mens_tourney}"

    @property
    def womens_reg_path(self) -> str:
        return f"{self.base_path}{self.womens_reg}"

    @property
    def womens_tourney_path(self) -> str:
        return f"{self.base_path}{self.womens_tourney}"

    @property
    def mens_teams_path(self) -> str:
        return f"{self.base_path}{self.mens_teams}"

    @property
    def womens_teams_path(self) -> str:
        return f"{self.base_path}{self.womens_teams}"

    @property
    def mens_seeds_path(self) -> str:
        return f"{self.base_path}{self.mens_seeds}"

    @property
    def womens_seeds_path(self) -> str:
        return f"{self.base_path}{self.womens_seeds}"
