from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna as opt
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig, TrainingConfig
from .feature_engineer import FeatureEngineer


class ModelTrainer:
    """Handles model training and hyperparameter optimization."""

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config

    def optimize_xgb_params(self, x_train: np.ndarray, y_train: np.ndarray, n_trials: Optional[int] = None) -> Dict:
        if n_trials is None:
            n_trials = self.training_config.xgb_trials

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "lambda": trial.suggest_float("lambda", 0.1, 2.0),
                "alpha": trial.suggest_float("alpha", 0.0, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.3, 0.8),
            }
            dtrain = xgb.DMatrix(x_train, label=y_train)
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=self.model_config.xgb_rounds,
                nfold=self.model_config.cv_folds,
                early_stopping_rounds=self.model_config.xgb_early_stop,
                metrics="logloss",
                seed=42,
            )
            return cv_results["test-logloss-mean"].min()

        print("Optimizing XGBoost parameters...")
        study = opt.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        print(f"Best XGBoost params: {best_params}")
        return best_params

    def train_models(
        self, x_train: np.ndarray, y_train: np.ndarray, xgb_params: Optional[Dict] = None
    ) -> Tuple[LogisticRegression, xgb.Booster]:
        if xgb_params is None:
            xgb_params = self.model_config.xgb_params.copy()

        logit_model = LogisticRegression(
            C=self.model_config.logit_c,
            solver="lbfgs",
            max_iter=self.model_config.logit_max_iter,
            class_weight="balanced" if self.model_config.use_balanced_logit else None,
        )
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        logit_model.fit(x_train_scaled, y_train)

        dtrain = xgb.DMatrix(x_train, label=y_train)
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=self.model_config.xgb_rounds)
        return logit_model, xgb_model

    def validate_models(
        self,
        reg_seasons: Dict[int, pd.DataFrame],
        tourney_df: pd.DataFrame,
        feature_engineer: FeatureEngineer,
        year_start: int,
        year_end: int,
    ) -> Dict[int, Dict[str, Any]]:
        print("\nValidation results:")
        results: Dict[int, Dict[str, Any]] = {}

        for season in tqdm(range(year_start + 1, year_end)):
            val_games = tourney_df.query("Season == @season")
            if val_games.empty:
                #print(f"season {season}: skipped (no tournament games)")
                continue

            train_df = tourney_df.query("Season < @season")
            x_train_list, y_train_list = [], []

            for train_season in sorted(train_df.Season.unique()):
                train_games = train_df.query("Season == @train_season")
                x_t, y_t = feature_engineer.create_decision_matrix(reg_seasons[train_season], train_games)
                x_train_list.append(x_t)
                y_train_list.append(y_t)

            if not x_train_list:
                #print(f"season {season}: skipped (no prior seasons for training)")
                continue

            x_train = np.vstack(x_train_list)
            y_train = np.hstack(y_train_list)
            y_train[y_train == -1] = 0

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            logit_model = LogisticRegression(
                C=self.model_config.logit_c,
                solver="lbfgs",
                max_iter=self.model_config.logit_max_iter,
            )
            logit_model.fit(x_train, y_train)

            dtrain = xgb.DMatrix(x_train, label=y_train)
            xgb_model = xgb.train(
                self.model_config.xgb_params,
                dtrain,
                num_boost_round=self.model_config.xgb_rounds,
            )

            x_val, y_val = feature_engineer.create_decision_matrix(reg_seasons[season], val_games)
            y_val[y_val == -1] = 0

            dval = xgb.DMatrix(x_val)
            logit_preds = logit_model.predict_proba(x_val)[:, 1]
            xgb_preds = xgb_model.predict(dval)

            logit_brier = brier_score_loss(y_val, logit_preds)
            xgb_brier = brier_score_loss(y_val, xgb_preds)
            #print(f"season {season}: logit brier {logit_brier:.5f}, xgb brier {xgb_brier:.5f}", flush=True)
            results[season] = {
                "status": "ok",
                "logit_brier": float(logit_brier),
                "xgb_brier": float(xgb_brier),
            }

        return results
