"""
FPL XGBoost Modeling Script

Trains global and position-specific XGBoost models for one-week-ahead FPL point forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import fpl_data_pipeline as pipeline

# Configuration
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22"]
VAL_SEASONS = ["2022-23"]
TEST_SEASONS = ["2023-24"]

DEFAULT_XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
}


def get_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load and prepare datasets for modeling.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    X_train, y_train, X_val, y_val, X_test, y_test = pipeline.build_datasets_for_modeling(
        seasons=SEASONS,
        train_seasons=TRAIN_SEASONS,
        val_seasons=VAL_SEASONS,
        test_seasons=TEST_SEASONS,
        player_windows=[3, 5, 10],
        team_windows=[3, 5, 10],
    )
    
    print("\nDataset summary:")
    print(f"  Train: {len(X_train)} examples, {X_train.shape[1]} features")
    print(f"  Val:   {len(X_val)} examples, {X_val.shape[1]} features")
    print(f"  Test:  {len(X_test)} examples, {X_test.shape[1]} features")
    print("  Target: FPL total_points for GW t+1 (one-week-ahead forecast).")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict | None = None,
) -> Tuple[XGBRegressor, float, float]:
    """
    Train an XGBRegressor model with early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation target.
    params : Dict, optional
        XGBoost parameters to override defaults.

    Returns
    -------
    Tuple[XGBRegressor, float, float]
        model, val_mae, val_rmse
    """
    model_params = DEFAULT_XGB_PARAMS.copy()
    if params is not None:
        model_params.update(params)
    
    if "eval_metric" not in model_params:
        model_params["eval_metric"] = "mae"
    if "early_stopping_rounds" not in model_params:
        model_params["early_stopping_rounds"] = 50

    model = XGBRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = root_mean_squared_error(y_val, y_val_pred)

    return model, val_mae, val_rmse


def evaluate_on_test(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
    """
    Evaluate a trained model on the test set.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.

    Returns
    -------
    Tuple[float, float]
        test_mae, test_rmse
    """
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    return test_mae, test_rmse


def get_feature_importances(
    model: XGBRegressor,
    feature_names: list,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract top feature importances from a trained model.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    feature_names : list
        List of feature names.
    top_n : int
        Number of top features to return, default 20.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'feature' and 'importance', sorted by importance.
    """
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    return df.sort_values("importance", ascending=False).head(top_n)


def run_global_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, XGBRegressor]:
    """
    Train one global model on all positions.

    Parameters
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test
        Train/val/test datasets.

    Returns
    -------
    Tuple[pd.DataFrame, XGBRegressor]
        Results DataFrame with one row and the trained model.
    """
    print("\n" + "=" * 60)
    print("Global Model (All Positions)")
    print("=" * 60)
    
    print("Training model...")
    model, val_mae, val_rmse = train_xgb_model(X_train, y_train, X_val, y_val)
    
    test_mae, test_rmse = evaluate_on_test(model, X_test, y_test)
    
    print(f"Val MAE (points):  {val_mae:.3f}")
    print(f"Val RMSE (points): {val_rmse:.3f}")
    print(f"Test MAE (points):  {test_mae:.3f}")
    print(f"Test RMSE (points): {test_rmse:.3f}")
    
    print("\nTop 20 feature importances:")
    print("-" * 60)
    feature_imp_df = get_feature_importances(model, X_train.columns.tolist(), top_n=20)
    print(feature_imp_df.to_string(index=False))
    
    results_df = pd.DataFrame([{
        "model_type": "global",
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }])
    
    return results_df, model


def run_position_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Train separate models for each position (GK, DEF, MID, FWD).

    Parameters
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test
        Train/val/test datasets.

    Returns
    -------
    pd.DataFrame
        Results DataFrame with one row per position.
    """
    print("\n" + "=" * 60)
    print("Position-Specific Models")
    print("=" * 60)
    
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    results = []
    
    # Get position from META_position in training data
    # We need to merge position info back from original data
    # For simplicity, assume META_position is in X_train
    if 'META_position' not in X_train.columns:
        print("Warning: META_position not found in features, cannot run position models")
        return pd.DataFrame()
    
    for position_id, position_name in position_map.items():
        print(f"\n{position_name} (position {position_id}):")
        print("-" * 60)
        
        # Filter by position
        train_mask = X_train['META_position'] == position_id
        val_mask = X_val['META_position'] == position_id
        test_mask = X_test['META_position'] == position_id
        
        X_train_pos = X_train[train_mask].copy()
        y_train_pos = y_train.loc[train_mask]
        X_val_pos = X_val[val_mask].copy()
        y_val_pos = y_val.loc[val_mask]
        X_test_pos = X_test[test_mask].copy()
        y_test_pos = y_test.loc[test_mask]
        
        # Drop META_position from features
        X_train_pos = X_train_pos.drop(columns=['META_position'])
        X_val_pos = X_val_pos.drop(columns=['META_position'])
        X_test_pos = X_test_pos.drop(columns=['META_position'])
        
        if len(X_train_pos) == 0:
            print(f"  No training data for {position_name}, skipping...")
            continue
        
        print(f"  Train: {len(X_train_pos)} examples, {X_train_pos.shape[1]} features")
        print(f"  Val:   {len(X_val_pos)} examples")
        print(f"  Test:  {len(X_test_pos)} examples")
        
        print("  Training model...")
        model, val_mae, val_rmse = train_xgb_model(X_train_pos, y_train_pos, X_val_pos, y_val_pos)
        
        test_mae, test_rmse = evaluate_on_test(model, X_test_pos, y_test_pos)
        
        print(f"  Val MAE (points):  {val_mae:.3f}")
        print(f"  Val RMSE (points): {val_rmse:.3f}")
        print(f"  Test MAE (points):  {test_mae:.3f}")
        print(f"  Test RMSE (points): {test_rmse:.3f}")
        
        print(f"\n  Top 10 features for {position_name}:")
        feature_imp_df = get_feature_importances(model, X_train_pos.columns.tolist(), top_n=10)
        print(feature_imp_df.to_string(index=False))
        
        results.append({
            "position": position_name,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("FPL XGBoost Modeling - Predicting Next Gameweek Points")
    print("=" * 60)
    
    # Load datasets
    X_train, y_train, X_val, y_val, X_test, y_test = get_datasets()
    
    # Run global model
    global_results, global_model = run_global_model(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Run position-specific models
    position_results = run_position_models(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Final Summary - Position-Specific Results")
    print("=" * 60)
    
    if len(position_results) > 0:
        print("\nPer-Position Performance:")
        print("-" * 60)
        display_cols = ["position", "val_mae", "val_rmse", "test_mae", "test_rmse"]
        display_df = position_results[display_cols].copy()
        
        for col in ["val_mae", "val_rmse", "test_mae", "test_rmse"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        
        print(display_df.to_string(index=False))
    
    print("\nNote:")
    print("  - This is a one-week-ahead forecasting task (predicting GW t+1 from data up to GW t).")
    print("  - Lower MAE/RMSE (in FPL points) indicate better predictive performance.")
    print("  - Per-position models allow us to see how predictability differs by position")
    print("    (GK, DEF, MID, FWD), as different positions have different scoring patterns.")
    print()
