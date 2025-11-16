"""
FPL XGBoost Modeling Script

Trains global and position-specific XGBoost models for one-week-ahead
FPL point forecasting, using the canonical feature pipeline
(last 4 games + ATT/DEF team form + ELO).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# IMPORTANT: use the new feature pipeline (no CSV loading)
import feature_engineering as fe

# Configuration
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22"]
VAL_SEASONS = ["2022-23"]
TEST_SEASONS = ["2023-24"]

TARGET_COL = "target_points_next_gw"

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


# -------------------------------------------------------------------
# Dataset construction (using feature_engineering.build_fpl_feature_table)
# -------------------------------------------------------------------

def get_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build the full feature table via the canonical pipeline, then split
    into train / val / test and construct (X, y) for each split.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    print("\nBuilding feature table via feature_engineering.build_fpl_feature_table()...")
    # Do NOT use the CSV; call the pipeline directly
    df = fe.build_fpl_feature_table(
        seasons=SEASONS,
        window=4,
        save_path=None,   # don't write to disk; we use the in-memory DataFrame
    )

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in feature table")

    # ID / metadata columns (not used as features directly)
    id_cols = [
        "season",
        "gw",
        "player_id",
        "name",
        "team",
        "opponent_team",
    ]

    # Same-GW outcome / leakage columns (MUST NOT be used as features)
    leak_cols = [
        "total_points", "bonus", "bps", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "saves", "penalties_saved",
        "penalties_missed", "ict_index", "influence", "creativity", "threat",
        "red_cards", "yellow_cards", "own_goals",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "selected", "transfers_in", "transfers_out",
        "transfers_balance", "value", "xP", "team_a_score", "team_h_score",
    ]

    # Feature prefixes we want to include
    feature_prefixes = [
        "FORM_",
        "TEAM_ATT_",
        "TEAM_DEF_",
        "OPP_ATT_",
        "OPP_DEF_",
        "RATE_",
        "META_",
        "PRICE_",
    ]

    # Build feature column list from the feature table
    feature_cols = []
    for col in df.columns:
        if any(col.startswith(p) for p in feature_prefixes):
            feature_cols.append(col)
        elif col in ["did_not_play_last_game", "TEAM_elo", "OPP_elo"]:
            feature_cols.append(col)

    # Remove IDs + leakage columns from the feature set
    exclude = set(id_cols + leak_cols + [TARGET_COL])
    feature_cols = sorted(list({c for c in feature_cols if c not in exclude}))

    # Sanity check: ensure META_position is present (for position models)
    if "META_position" not in feature_cols and "META_position" in df.columns:
        feature_cols.append("META_position")

    # Train / val / test masks
    train_mask = df["season"].isin(TRAIN_SEASONS)
    val_mask = df["season"].isin(VAL_SEASONS)
    test_mask = df["season"].isin(TEST_SEASONS)

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, TARGET_COL].copy()

    X_val = df.loc[val_mask, feature_cols].copy()
    y_val = df.loc[val_mask, TARGET_COL].copy()

    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, TARGET_COL].copy()

    # Fill any remaining NaNs with 0
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    print("\nDataset summary:")
    print(f"  Train: {len(X_train)} examples, {X_train.shape[1]} features")
    print(f"  Val:   {len(X_val)} examples, {X_val.shape[1]} features")
    print(f"  Test:  {len(X_test)} examples, {X_test.shape[1]} features")
    print("  Target: one-week-ahead FPL points (target_points_next_gw).")

    return X_train, y_train, X_val, y_val, X_test, y_test


# -------------------------------------------------------------------
# Model training / evaluation
# -------------------------------------------------------------------

def train_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict | None = None,
) -> Tuple[XGBRegressor, float, float]:
    """
    Train an XGBRegressor model with early stopping.

    Returns
    -------
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
    """Evaluate a trained model on the test set."""
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    return test_mae, test_rmse


def get_feature_importances(
    model: XGBRegressor,
    feature_names: list,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return top-n feature importances."""
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    return df.sort_values("importance", ascending=False).head(top_n)


# -------------------------------------------------------------------
# Global model (all positions)
# -------------------------------------------------------------------

def run_global_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, XGBRegressor]:
    """
    Train a single global model on all positions.
    """
    print("\n" + "=" * 60)
    print("Global Model (All Positions)")
    print("=" * 60)

    print("Training model...")
    model, val_mae, val_rmse = train_xgb_model(X_train, y_train, X_val, y_val)
    test_mae, test_rmse = evaluate_on_test(model, X_test, y_test)

    print(f"Val MAE (points):  {val_mae:.3f}")
    print(f"Val RMSE (points): {val_rmse:.3f}")
    print(f"Test MAE (points): {test_mae:.3f}")
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


# -------------------------------------------------------------------
# Position-specific models
# -------------------------------------------------------------------

def run_position_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Train separate models for each position (GK, DEF, MID, FWD),
    using META_position in the feature matrix to subset.
    """
    print("\n" + "=" * 60)
    print("Position-Specific Models")
    print("=" * 60)

    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    results = []

    if "META_position" not in X_train.columns:
        print("Warning: META_position not found in features, cannot run position models")
        return pd.DataFrame()

    for position_id, position_name in position_map.items():
        print(f"\n{position_name} (position {position_id}):")
        print("-" * 60)

        train_mask = X_train["META_position"] == position_id
        val_mask = X_val["META_position"] == position_id
        test_mask = X_test["META_position"] == position_id

        X_train_pos = X_train[train_mask].copy()
        y_train_pos = y_train.loc[train_mask]
        X_val_pos = X_val[val_mask].copy()
        y_val_pos = y_val.loc[val_mask]
        X_test_pos = X_test[test_mask].copy()
        y_test_pos = y_test.loc[test_mask]

        # Drop META_position from features for the model itself
        X_train_pos = X_train_pos.drop(columns=["META_position"])
        X_val_pos = X_val_pos.drop(columns=["META_position"])
        X_test_pos = X_test_pos.drop(columns=["META_position"])

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
        print(f"  Test MAE (points): {test_mae:.3f}")
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


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("FPL XGBoost Modeling - Predicting Next Gameweek Points")
    print("=" * 60)

    # Build datasets from the pipeline
    X_train, y_train, X_val, y_val, X_test, y_test = get_datasets()

    # Global model
    global_results, global_model = run_global_model(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Position-specific models
    position_results = run_position_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Final summary
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
    print("  - Features use ONLY the last 4 games of player and team form, plus ELO.")
    print("  - Lower MAE/RMSE (in FPL points) indicate better predictive performance.")
    print("  - Per-position models show how predictability differs by position (GK/DEF/MID/FWD).")
    print()