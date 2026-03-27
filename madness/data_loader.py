import numpy as np
import pandas as pd

from .config import DataPaths


class DataLoader:
    """Handles loading and preprocessing of tournament data."""

    def __init__(self, paths: DataPaths):
        self.paths = paths

    def load_regular_season(self, gender: str) -> pd.DataFrame:
        if gender == "mens":
            return pd.read_csv(self.paths.mens_reg_path)
        if gender == "womens":
            return pd.read_csv(self.paths.womens_reg_path)
        raise ValueError(f"Invalid gender: {gender}")

    def load_tournament(self, gender: str) -> pd.DataFrame:
        if gender == "mens":
            return pd.read_csv(self.paths.mens_tourney_path)[["Season", "WTeamID", "LTeamID"]]
        if gender == "womens":
            return pd.read_csv(self.paths.womens_tourney_path)[["Season", "WTeamID", "LTeamID"]]
        raise ValueError(f"Invalid gender: {gender}")

    def load_teams(self, gender: str) -> np.ndarray:
        if gender == "mens":
            return pd.read_csv(self.paths.mens_teams_path).TeamID.unique()
        if gender == "womens":
            return pd.read_csv(self.paths.womens_teams_path).TeamID.unique()
        raise ValueError(f"Invalid gender: {gender}")

    def load_sample_submission(self, stage: int) -> pd.DataFrame:
        if stage == 1:
            return pd.read_csv(f"{self.paths.base_path}{self.paths.sample_submission_stage1}")
        if stage == 2:
            return pd.read_csv(f"{self.paths.base_path}{self.paths.sample_submission_stage2}")
        raise ValueError(f"Invalid stage: {stage}")

    def load_seeds(self, gender: str) -> pd.DataFrame:
        if gender == "mens":
            seeds = pd.read_csv(self.paths.mens_seeds_path)
        elif gender == "womens":
            seeds = pd.read_csv(self.paths.womens_seeds_path)
        else:
            raise ValueError(f"Invalid gender: {gender}")

        seeds["Seed"] = seeds.iloc[:, 1].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
        return seeds
