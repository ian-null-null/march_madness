import numpy as np

from .config import DataPaths, EloConfig, ModelConfig
from .data_loader import DataLoader
from .elo import EloSystem
from .feature_engineer import FeatureEngineer


def inspect_decision_matrix():
    """Debug helper to inspect the output of create_decision_matrix."""
    data_loader = DataLoader(DataPaths())
    reg_df = data_loader.load_regular_season("mens")
    tourney_df = data_loader.load_tournament("mens")
    seeds_df = data_loader.load_seeds("mens")

    elo_system = EloSystem(EloConfig())
    elos = elo_system.compute_season_ratings(reg_df)

    feature_engineer = FeatureEngineer(ModelConfig())
    year = 2023
    reg_stats = feature_engineer.compute_season_stats(year, elos, reg_df, seeds_df)
    year_games = tourney_df.query("Season == @year")

    if year_games.empty:
        print(f"No tournament games found for year {year}")
        return None, None

    x, y = feature_engineer.create_decision_matrix(reg_stats, year_games)
    print("=== Decision Matrix Inspection ===")
    print(f"Year: {year}")
    print(f"Number of games: {len(year_games)}")
    print(f"Feature matrix shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"NaN values: {np.isnan(x).sum()}")
    print(f"Infinite values: {np.isinf(x).sum()}")
    return x, y
