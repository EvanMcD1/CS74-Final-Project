"""
FPL Data Pipeline

Clean, efficient module for preparing Fantasy Premier League data for XGBoost modeling.
Creates leakage-free features for one-week-ahead forecasting (predicting GW t+1 from data up to GW t).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def load_raw_data_from_urls(seasons: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load merged gameweek data for given seasons from the vaastav FPL GitHub repository.

    Parameters
    ----------
    seasons : List[str]
        List of season strings, e.g. ["2019-20", "2020-21", ...].

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Concatenated DataFrame and list of successfully loaded seasons.

    Raises
    ------
    ValueError
        If no seasons could be loaded.
    """
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}/gws/merged_gw.csv"
    
    dfs = []
    loaded_seasons = []
    
    for season in seasons:
        url = base_url.format(season)
        try:
            df_season = pd.read_csv(url, low_memory=False)
            df_season['season'] = season
            
            # Ensure numeric gw column
            if 'GW' in df_season.columns:
                df_season['gw'] = pd.to_numeric(df_season['GW'], errors='coerce').astype('Int64')
            elif 'gw' in df_season.columns:
                df_season['gw'] = pd.to_numeric(df_season['gw'], errors='coerce').astype('Int64')
            
            dfs.append(df_season)
            loaded_seasons.append(season)
        except Exception as e:
            print(f"Warning: Could not load season {season}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No data loaded from any season")
    
    df = pd.concat(dfs, ignore_index=True)
    return df, loaded_seasons


def create_player_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a stable player_id column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'player_id' column.
    """
    df = df.copy()
    
    if 'code' in df.columns:
        df['player_id'] = df['code'].astype(str)
    else:
        if 'season' in df.columns:
            df['player_id'] = df['name'].astype(str) + "_" + df['season'].astype(str)
        else:
            df['player_id'] = df['name'].astype(str)
    
    return df


def sort_for_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort DataFrame for time-series operations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame by player_id, season, gw.
    """
    df = df.copy()
    return df.sort_values(["player_id", "season", "gw"]).reset_index(drop=True)


def add_player_form_features(df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Add leakage-free rolling form features per player.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted DataFrame with player_id, season, gw.
    windows : List[int]
        Rolling window sizes, default [3, 5, 10].

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'FORM_' prefixed columns.
    """
    df = df.copy()
    
    base_stats = ['total_points', 'minutes', 'goals_scored', 'assists']
    group_cols = ["player_id", "season"]
    
    for stat in base_stats:
        if stat not in df.columns:
            continue
        
        # Create temporary shifted column
        temp_col = f'_shifted_{stat}'
        df[temp_col] = df.groupby(group_cols)[stat].shift(1)
        
        for window in windows:
            # Mean
            df[f'FORM_{stat}_mean_{window}'] = (
                df.groupby(group_cols)[temp_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=group_cols, drop=True)
            )
            # Sum
            df[f'FORM_{stat}_sum_{window}'] = (
                df.groupby(group_cols)[temp_col]
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(level=group_cols, drop=True)
            )
        
        # Clean up temporary column
        df = df.drop(columns=[temp_col])
    
    # Start rate (fraction of games with minutes >= 60)
    if 'minutes' in df.columns:
        temp_minutes_col = '_shifted_minutes'
        df[temp_minutes_col] = df.groupby(group_cols)['minutes'].shift(1)
        temp_started_col = '_started'
        df[temp_started_col] = (df[temp_minutes_col] >= 60).astype(int)
        
        for window in windows:
            df[f'FORM_start_rate_{window}'] = (
                df.groupby(group_cols)[temp_started_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=group_cols, drop=True)
            )
        
        # Clean up temporary columns
        df = df.drop(columns=[temp_minutes_col, temp_started_col])
    
    return df


def add_team_and_opponent_form_features(df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Add team-level and opponent-strength features.

    Parameters
    ----------
    df : pd.DataFrame
        Player-level DataFrame.
    windows : List[int]
        Rolling window sizes, default [3, 5, 10].

    Returns
    -------
    pd.DataFrame
        DataFrame with 'TEAM_' and 'OPP_' prefixed columns.
    """
    df = df.copy()
    
    # Build team-level table
    team_df = df.groupby(["team", "season", "gw"], as_index=False).agg({
        'goals_scored': 'sum',
        'goals_conceded': 'first'
    })
    team_df = team_df.sort_values(["team", "season", "gw"]).reset_index(drop=True)
    
    # Add rolling stats per team
    group_cols = ["team", "season"]
    for stat in ['goals_scored', 'goals_conceded']:
        # Create temporary shifted column
        temp_col = f'_shifted_{stat}'
        team_df[temp_col] = team_df.groupby(group_cols)[stat].shift(1)
        
        for window in windows:
            team_df[f'TEAM_{stat}_mean_{window}'] = (
                team_df.groupby(group_cols)[temp_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=group_cols, drop=True)
            )
        
        # Clean up temporary column
        team_df = team_df.drop(columns=[temp_col])
    
    # Merge TEAM_ features back to player df
    team_cols = ['team', 'season', 'gw'] + [col for col in team_df.columns if col.startswith('TEAM_')]
    df = df.merge(team_df[team_cols], on=["team", "season", "gw"], how='left')
    
    # Create opponent-strength features
    opp_df = team_df.copy()
    opp_df = opp_df.rename(columns={'team': 'opponent_team'})
    opp_df = opp_df.rename(columns={col: col.replace('TEAM_', 'OPP_') for col in opp_df.columns if col.startswith('TEAM_')})
    
    # Ensure compatible types for merge
    if 'opponent_team' in df.columns:
        df['opponent_team'] = df['opponent_team'].astype(str)
        opp_df['opponent_team'] = opp_df['opponent_team'].astype(str)
    
    opp_cols = ['opponent_team', 'season', 'gw'] + [col for col in opp_df.columns if col.startswith('OPP_')]
    df = df.merge(opp_df[opp_cols], on=["opponent_team", "season", "gw"], how='left')
    
    return df


def add_per_minute_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-minute rate features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'RATE_' prefixed columns.
    """
    df = df.copy()
    
    # Avoid division by zero
    minutes_safe = df['minutes'].replace(0, np.nan)
    
    if 'goals_scored' in df.columns:
        df['RATE_goals_per_min'] = df['goals_scored'] / minutes_safe
    if 'assists' in df.columns:
        df['RATE_assists_per_min'] = df['assists'] / minutes_safe
    
    if 'expected_goals' in df.columns:
        df['RATE_xG_per_min'] = df['expected_goals'] / minutes_safe
    if 'expected_assists' in df.columns:
        df['RATE_xA_per_min'] = df['expected_assists'] / minutes_safe
    
    # Fill NaNs with 0
    rate_cols = [col for col in df.columns if col.startswith('RATE_')]
    df[rate_cols] = df[rate_cols].fillna(0)
    
    return df


def add_did_not_play_last_game(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary feature indicating if player did not play last game.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'did_not_play_last_game' column.
    """
    df = df.copy()
    
    group_cols = ["player_id", "season"]
    prev_minutes = df.groupby(group_cols)["minutes"].shift(1)
    df['did_not_play_last_game'] = ((prev_minutes == 0) | prev_minutes.isna()).astype(int)
    
    return df


def add_meta_and_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add meta and price features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'META_' and 'PRICE_' prefixed columns.
    """
    df = df.copy()
    
    # Home/Away
    if 'was_home' in df.columns:
        df['META_is_home'] = df['was_home'].astype(int)
    
    # Position
    if 'element_type' in df.columns:
        df['META_position'] = df['element_type'].astype(int)
    elif 'position' in df.columns:
        # Map string positions to numeric (1=GK, 2=DEF, 3=MID, 4=FWD)
        position_map = {
            'GK': 1, 'GKP': 1, 'Goalkeeper': 1,
            'DEF': 2, 'DEFENDER': 2, 'Defender': 2,
            'MID': 3, 'MIDFIELDER': 3, 'Midfielder': 3,
            'FWD': 4, 'FORWARD': 4, 'Forward': 4, 'ATT': 4, 'ATTACKER': 4
        }
        df['META_position'] = df['position'].map(position_map).fillna(0).astype(int)
    
    # Gameweek
    if 'gw' in df.columns:
        df['META_gw'] = pd.to_numeric(df['gw'], errors='coerce').fillna(0).astype(int)
    
    # Price
    if 'now_cost' in df.columns:
        df['PRICE_now'] = pd.to_numeric(df['now_cost'], errors='coerce').fillna(0)
        df['PRICE_now_m'] = df['PRICE_now'] / 10.0
    
    return df


def create_next_week_target(df: pd.DataFrame, base_col: str = "total_points", target_col: str = "target_points_next_gw") -> pd.DataFrame:
    """
    Create next-gameweek target by shifting total_points forward.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    base_col : str
        Base column to shift, default "total_points".
    target_col : str
        Target column name, default "target_points_next_gw".

    Returns
    -------
    pd.DataFrame
        DataFrame with added target column, rows with NaN target dropped.
    """
    df = df.copy()
    
    group_cols = ["player_id", "season"]
    df[target_col] = df.groupby(group_cols)[base_col].shift(-1)
    
    initial_rows = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows (last gameweek of each player/season, no next GW to predict)")
    
    return df


def make_splits(
    df: pd.DataFrame,
    train_seasons: List[str],
    val_seasons: List[str],
    test_seasons: List[str],
    target_col: str = "target_points_next_gw",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split DataFrame into train/val/test sets based on seasons.

    Parameters
    ----------
    df : pd.DataFrame
        Fully-featured DataFrame with target column.
    train_seasons : List[str]
        Seasons for training.
    val_seasons : List[str]
        Seasons for validation.
    test_seasons : List[str]
        Seasons for testing.
    target_col : str
        Target column name, default "target_points_next_gw".

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Columns to exclude
    exclude_cols = ["season", "gw", "player_id", target_col]
    
    # Leakage columns (same-GW outcome features)
    leak_cols = [
        "total_points", "bonus", "bps", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "saves", "penalties_saved",
        "penalties_missed", "ict_index", "influence", "creativity", "threat",
        "red_cards", "yellow_cards", "own_goals",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "selected", "transfers_in", "transfers_out",
        "transfers_balance", "value", "xP", "team_a_score", "team_h_score",
    ]
    
    # Feature prefixes to include
    feature_prefixes = ["FORM_", "TEAM_", "OPP_", "RATE_", "META_", "PRICE_"]
    
    # Get all feature columns
    feature_cols = [
        col for col in df.columns
        if any(col.startswith(prefix) for prefix in feature_prefixes)
        or col == "did_not_play_last_game"
    ]
    
    # Remove excluded and leakage columns
    feature_cols = [col for col in feature_cols if col not in exclude_cols + leak_cols]
    feature_cols = sorted(list(set(feature_cols)))
    
    # Create splits
    train_mask = df['season'].isin(train_seasons)
    val_mask = df['season'].isin(val_seasons)
    test_mask = df['season'].isin(test_seasons)
    
    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, target_col].copy()
    
    X_val = df.loc[val_mask, feature_cols].copy()
    y_val = df.loc[val_mask, target_col].copy()
    
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, target_col].copy()
    
    # Fill NaNs with 0
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_datasets_for_modeling(
    seasons: List[str],
    train_seasons: List[str],
    val_seasons: List[str],
    test_seasons: List[str],
    player_windows: List[int] = [3, 5, 10],
    team_windows: List[int] = [3, 5, 10],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    High-level function that orchestrates the entire data pipeline.

    Parameters
    ----------
    seasons : List[str]
        All seasons to attempt loading.
    train_seasons : List[str]
        Seasons for training.
    val_seasons : List[str]
        Seasons for validation.
    test_seasons : List[str]
        Seasons for testing.
    player_windows : List[int]
        Rolling windows for player form features, default [3, 5, 10].
    team_windows : List[int]
        Rolling windows for team stats, default [3, 5, 10].

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    print("Loading raw data from GitHub...")
    df, loaded_seasons = load_raw_data_from_urls(seasons)
    print(f"Successfully loaded seasons: {sorted(loaded_seasons)}")
    
    missing = set(seasons) - set(loaded_seasons)
    if missing:
        print(f"Warning: the following seasons could not be loaded and will be ignored: {sorted(missing)}")
    
    print(f"Loaded {len(df)} rows from {len(loaded_seasons)} seasons")
    
    # Filter train/val/test seasons to only include successfully loaded ones
    train_seasons = [s for s in train_seasons if s in loaded_seasons]
    val_seasons = [s for s in val_seasons if s in loaded_seasons]
    test_seasons = [s for s in test_seasons if s in loaded_seasons]
    
    if not train_seasons:
        raise ValueError("No training seasons available after filtering to loaded seasons")
    if not val_seasons:
        print("Warning: No validation seasons available after filtering to loaded seasons")
    if not test_seasons:
        print("Warning: No test seasons available after filtering to loaded seasons")
    
    # Feature engineering pipeline
    print("Creating player_id column...")
    df = create_player_id_column(df)
    
    print("Sorting for time series...")
    df = sort_for_time_series(df)
    
    print("Adding player form features...")
    df = add_player_form_features(df, windows=player_windows)
    
    print("Adding team and opponent form features...")
    df = add_team_and_opponent_form_features(df, windows=team_windows)
    
    print("Adding per-minute rate features...")
    df = add_per_minute_rate_features(df)
    
    print("Adding did_not_play_last_game feature...")
    df = add_did_not_play_last_game(df)
    
    print("Adding meta and price features...")
    df = add_meta_and_price_features(df)
    
    print("Creating next-gameweek target (predicting GW t+1 from data up to GW t)...")
    df = create_next_week_target(df)
    
    print("Creating train/val/test splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = make_splits(
        df, train_seasons, val_seasons, test_seasons
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} rows, {len(X_train.columns)} features")
    print(f"  Val:   {len(X_val)} rows, {len(X_val.columns)} features")
    print(f"  Test:  {len(X_test)} rows, {len(X_test.columns)} features")
    
    return X_train, y_train, X_val, y_val, X_test, y_test
