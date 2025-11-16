"""
FPL Feature Pipeline (Last 4 Games + Team ELO)

Builds a canonical player-gameweek feature table for forecasting FPL points.

Key design:
  - Uses ONLY last 4 games for all "form" features (player, team, opponent).
  - Separates team context into ATTACK and DEFENSE:
      TEAM_ATT_*   (own team attack form)
      TEAM_DEF_*   (own team defense form)
      OPP_ATT_*    (opponent attack form)
      OPP_DEF_*    (opponent defense form)
  - Computes a simple team ELO per team per gameweek (TEAM_elo, OPP_elo).
  - Returns the full feature DataFrame and saves it to ./data/fpl_features_last4_elo.csv

This table is meant to be the shared input to:
  - Linear Regression
  - XGBoost
  - LLM-based models
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# 1. Raw data loading
# -------------------------------------------------------------------

def load_raw_data_from_urls(seasons: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load merged gameweek data for given seasons from the vaastav FPL GitHub repository.

    Parameters
    ----------
    seasons : list of str
        Season labels, e.g. ["2019-20", "2020-21", ...].

    Returns
    -------
    df : pd.DataFrame
        Concatenated DataFrame of all successfully loaded seasons.
    loaded_seasons : list of str
        Seasons that were actually loaded.
    """
    base_url = (
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
        "master/data/{}/gws/merged_gw.csv"
    )

    dfs = []
    loaded = []

    for season in seasons:
        url = base_url.format(season)
        try:
            df_season = pd.read_csv(url, low_memory=False)
            df_season["season"] = season

            # Ensure numeric gw column
            if "GW" in df_season.columns:
                df_season["gw"] = pd.to_numeric(df_season["GW"], errors="coerce").astype("Int64")
            elif "gw" in df_season.columns:
                df_season["gw"] = pd.to_numeric(df_season["gw"], errors="coerce").astype("Int64")

            dfs.append(df_season)
            loaded.append(season)
        except Exception as e:
            print(f"Warning: Could not load season {season}: {e}")

    if not dfs:
        raise ValueError("No data loaded from any season")

    df = pd.concat(dfs, ignore_index=True)
    return df, loaded


# -------------------------------------------------------------------
# 2. IDs + sorting
# -------------------------------------------------------------------

def create_player_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a stable 'player_id' column.

    Uses 'code' if available; otherwise falls back to name+season.
    """
    df = df.copy()
    if "code" in df.columns:
        df["player_id"] = df["code"].astype(str)
    else:
        if "season" in df.columns:
            df["player_id"] = df["name"].astype(str) + "_" + df["season"].astype(str)
        else:
            df["player_id"] = df["name"].astype(str)
    return df


def sort_for_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame for clean time-series operations."""
    df = df.copy()
    return df.sort_values(["player_id", "season", "gw"]).reset_index(drop=True)


# -------------------------------------------------------------------
# 3. Player FORM features (last 4 games only)
# -------------------------------------------------------------------

def add_player_form_features(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Add leakage-free rolling player form features using last `window` games.

    For each player_id, season, gw (sorted), and for window=4, we create:
      - FORM_total_points_mean_4 / sum_4
      - FORM_minutes_mean_4 / sum_4
      - FORM_goals_scored_sum_4
      - FORM_assists_sum_4
      - FORM_start_rate_4  (fraction of last 4 games with minutes >= 60)
    """
    df = df.copy()
    group_cols = ["player_id", "season"]
    base_stats = ["total_points", "minutes", "goals_scored", "assists"]

    for stat in base_stats:
        if stat not in df.columns:
            continue

        temp = f"_shifted_{stat}"
        df[temp] = df.groupby(group_cols)[stat].shift(1)

        # Mean for total_points and minutes
        if stat in ["total_points", "minutes"]:
            df[f"FORM_{stat}_mean_{window}"] = (
                df.groupby(group_cols)[temp]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=group_cols, drop=True)
            )

        # Sum for all stats
        df[f"FORM_{stat}_sum_{window}"] = (
            df.groupby(group_cols)[temp]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=group_cols, drop=True)
        )

        df = df.drop(columns=[temp])

    # Start rate: fraction of last N games with minutes >= 60
    if "minutes" in df.columns:
        m_temp = "_shifted_minutes"
        started_temp = "_started"
        df[m_temp] = df.groupby(group_cols)["minutes"].shift(1)
        df[started_temp] = (df[m_temp] >= 60).astype(int)

        df[f"FORM_start_rate_{window}"] = (
            df.groupby(group_cols)[started_temp]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=group_cols, drop=True)
        )

        df = df.drop(columns=[m_temp, started_temp])

    return df


# -------------------------------------------------------------------
# 4. Team form (ATT/DEF, last 4 games) + ELO
# -------------------------------------------------------------------

def build_team_form_table(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Build per-team, per-gw rolling form stats using last `window` games.

    Input: player-level df with at least:
        - 'team', 'season', 'gw'
        - 'goals_scored', 'goals_conceded'
        - 'opponent_team', 'was_home', 'team_h_score', 'team_a_score' (if available)

    Output: team_df with columns:
        - team, season, gw, opponent_team, was_home, team_h_score, team_a_score
        - TEAM_ATT_goals_scored_mean_4
        - TEAM_DEF_goals_conceded_mean_4
    """
    # Aggregate to team-gw level
    agg_dict = {}

    if "goals_scored" in df.columns:
        agg_dict["goals_scored"] = "sum"
    if "goals_conceded" in df.columns:
        agg_dict["goals_conceded"] = "first"
    if "opponent_team" in df.columns:
        agg_dict["opponent_team"] = "first"
    if "was_home" in df.columns:
        agg_dict["was_home"] = "first"
    if "team_h_score" in df.columns:
        agg_dict["team_h_score"] = "first"
    if "team_a_score" in df.columns:
        agg_dict["team_a_score"] = "first"

    if not agg_dict:
        raise ValueError("No suitable columns found for team aggregation")

    team_df = (
        df.groupby(["team", "season", "gw"], as_index=False)
        .agg(agg_dict)
    )

    team_df = team_df.sort_values(["team", "season", "gw"]).reset_index(drop=True)

    group_cols = ["team", "season"]

    # Team attacking form (goals scored)
    if "goals_scored" in team_df.columns:
        temp = "_shifted_gs"
        team_df[temp] = team_df.groupby(group_cols)["goals_scored"].shift(1)
        team_df[f"TEAM_ATT_goals_scored_mean_{window}"] = (
            team_df.groupby(group_cols)[temp]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=group_cols, drop=True)
        )
        team_df = team_df.drop(columns=[temp])

    # Team defensive form (goals conceded)
    if "goals_conceded" in team_df.columns:
        temp = "_shifted_gc"
        team_df[temp] = team_df.groupby(group_cols)["goals_conceded"].shift(1)
        team_df[f"TEAM_DEF_goals_conceded_mean_{window}"] = (
            team_df.groupby(group_cols)[temp]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=group_cols, drop=True)
        )
        team_df = team_df.drop(columns=[temp])

    return team_df


def compute_team_elo(
    team_df: pd.DataFrame,
    base_rating: float = 1500.0,
    k: float = 20.0,
) -> pd.DataFrame:
    """
    Compute a simple ELO rating per team per gameweek.

    We:
      - Take one row per match: the home team perspective (was_home == True).
      - For each match, record pre-game ratings for both teams, then update
        ratings based on match result (team_h_score vs team_a_score).

    Returns:
        elo_df with columns: season, gw, team, TEAM_elo
    """
    required = {"team", "season", "gw", "opponent_team", "was_home", "team_h_score", "team_a_score"}
    if not required.issubset(set(team_df.columns)):
        raise ValueError("team_df missing required columns for ELO computation")

    # Only process each match once: rows where this team is at home
    matches = team_df[team_df["was_home"] == True].copy()
    matches = matches.sort_values(["season", "gw"]).reset_index(drop=True)

    ratings: Dict[Tuple[str, str], float] = {}
    records = []

    for _, row in matches.iterrows():
        season = row["season"]
        gw = int(row["gw"])
        home_team = str(row["team"])
        away_team = str(row["opponent_team"])
        home_goals = row["team_h_score"]
        away_goals = row["team_a_score"]

        # Fetch current ratings (default base_rating)
        home_key = (season, home_team)
        away_key = (season, away_team)
        R_home = ratings.get(home_key, base_rating)
        R_away = ratings.get(away_key, base_rating)

        # Store pre-game ratings for this GW
        records.append({"season": season, "gw": gw, "team": home_team, "TEAM_elo": R_home})
        records.append({"season": season, "gw": gw, "team": away_team, "TEAM_elo": R_away})

        # Determine actual result
        if home_goals > away_goals:
            S_home, S_away = 1.0, 0.0
        elif home_goals < away_goals:
            S_home, S_away = 0.0, 1.0
        else:
            S_home, S_away = 0.5, 0.5

        # Expected probabilities (logistic)
        E_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))
        E_away = 1 - E_home

        # Update ratings
        R_home_new = R_home + k * (S_home - E_home)
        R_away_new = R_away + k * (S_away - E_away)

        ratings[home_key] = R_home_new
        ratings[away_key] = R_away_new

    elo_df = pd.DataFrame(records).drop_duplicates(subset=["season", "gw", "team"])
    return elo_df


def add_team_and_opponent_context_with_elo(
    df: pd.DataFrame,
    window: int = 4,
) -> pd.DataFrame:
    """
    Add team ATT/DEF form and opponent ATT/DEF form (last `window` games),
    plus TEAM_elo and OPP_elo.

    Returns:
        player-level df with:
          - TEAM_ATT_goals_scored_mean_4
          - TEAM_DEF_goals_conceded_mean_4
          - OPP_ATT_goals_scored_mean_4
          - OPP_DEF_goals_conceded_mean_4
          - TEAM_elo
          - OPP_elo
    """
    df = df.copy()

    # Build team form table
    team_df = build_team_form_table(df, window=window)

    # --- DTYPE NORMALIZATION (fixes merge error) --------------------
    # Ensure team/opponent identifiers have the SAME type everywhere
    # Convert to string on both sides
    if "team" in df.columns:
        df["team"] = df["team"].astype(str)
    if "opponent_team" in df.columns:
        df["opponent_team"] = df["opponent_team"].astype(str)

    if "team" in team_df.columns:
        team_df["team"] = team_df["team"].astype(str)
    if "opponent_team" in team_df.columns:
        team_df["opponent_team"] = team_df["opponent_team"].astype(str)

    # Also make gw numeric (safe) so merges on gw/season are clean
    if "gw" in df.columns:
        df["gw"] = pd.to_numeric(df["gw"], errors="coerce")
    if "gw" in team_df.columns:
        team_df["gw"] = pd.to_numeric(team_df["gw"], errors="coerce")
    # ----------------------------------------------------------------

    # Compute ELO on team_df
    elo_df = compute_team_elo(team_df)

    # TEAM_ features to merge back
    team_features = [
        "TEAM_ATT_goals_scored_mean_4",
        "TEAM_DEF_goals_conceded_mean_4",
    ]
    merge_team = team_df[["team", "season", "gw"] + team_features].copy()

    # Merge TEAM_ form features back to player df
    df = df.merge(merge_team, on=["team", "season", "gw"], how="left")

    # Build opponent form by renaming team -> opponent_team and TEAM_* -> OPP_*
    opp_df = merge_team.copy()
    opp_df = opp_df.rename(columns={"team": "opponent_team"})

    rename_cols = {}
    for col in team_features:
        if col.startswith("TEAM_ATT_"):
            rename_cols[col] = col.replace("TEAM_ATT_", "OPP_ATT_")
        elif col.startswith("TEAM_DEF_"):
            rename_cols[col] = col.replace("TEAM_DEF_", "OPP_DEF_")
    opp_df = opp_df.rename(columns=rename_cols)

    opp_features = list(rename_cols.values())

    # Ensure opponent_team types are aligned before merge (extra safety)
    opp_df["opponent_team"] = opp_df["opponent_team"].astype(str)

    df = df.merge(
        opp_df[["opponent_team", "season", "gw"] + opp_features],
        on=["opponent_team", "season", "gw"],
        how="left",
    )

    # Merge TEAM_elo
    df = df.merge(elo_df, on=["season", "gw", "team"], how="left")

    # Merge OPP_elo
    opp_elo = elo_df.rename(columns={"team": "opponent_team", "TEAM_elo": "OPP_elo"})
    opp_elo["opponent_team"] = opp_elo["opponent_team"].astype(str)

    df = df.merge(
        opp_elo[["season", "gw", "opponent_team", "OPP_elo"]],
        on=["season", "gw", "opponent_team"],
        how="left",
    )

    return df

# -------------------------------------------------------------------
# 5. Per-minute rates + availability + meta
# -------------------------------------------------------------------

def add_per_minute_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-minute rate features (goals, assists, xG, xA per minute).
    """
    df = df.copy()
    if "minutes" not in df.columns:
        return df

    minutes_safe = df["minutes"].replace(0, np.nan)

    if "goals_scored" in df.columns:
        df["RATE_goals_per_min"] = df["goals_scored"] / minutes_safe
    if "assists" in df.columns:
        df["RATE_assists_per_min"] = df["assists"] / minutes_safe
    if "expected_goals" in df.columns:
        df["RATE_xG_per_min"] = df["expected_goals"] / minutes_safe
    if "expected_assists" in df.columns:
        df["RATE_xA_per_min"] = df["expected_assists"] / minutes_safe

    rate_cols = [c for c in df.columns if c.startswith("RATE_")]
    df[rate_cols] = df[rate_cols].fillna(0)

    return df


def add_did_not_play_last_game(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary 'did_not_play_last_game' based on previous GW minutes.
    """
    df = df.copy()
    group_cols = ["player_id", "season"]
    prev_minutes = df.groupby(group_cols)["minutes"].shift(1)
    df["did_not_play_last_game"] = ((prev_minutes == 0) | prev_minutes.isna()).astype(int)
    return df


def add_meta_and_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add META_* and PRICE_* features:
        META_is_home, META_position, META_gw, PRICE_now_m
    """
    df = df.copy()

    # Home / away
    if "was_home" in df.columns:
        df["META_is_home"] = df["was_home"].astype(int)

    # Position
    if "element_type" in df.columns:
        df["META_position"] = df["element_type"].astype(int)
    elif "position" in df.columns:
        position_map = {
            "GK": 1, "GKP": 1, "Goalkeeper": 1,
            "DEF": 2, "DEFENDER": 2, "Defender": 2,
            "MID": 3, "MIDFIELDER": 3, "Midfielder": 3,
            "FWD": 4, "FORWARD": 4, "Forward": 4, "ATT": 4, "ATTACKER": 4,
        }
        df["META_position"] = df["position"].map(position_map).fillna(0).astype(int)

    # Gameweek index
    if "gw" in df.columns:
        df["META_gw"] = pd.to_numeric(df["gw"], errors="coerce").fillna(0).astype(int)

    # Price
    if "now_cost" in df.columns:
        df["PRICE_now"] = pd.to_numeric(df["now_cost"], errors="coerce").fillna(0)
        df["PRICE_now_m"] = df["PRICE_now"] / 10.0

    return df


# -------------------------------------------------------------------
# 6. Target creation
# -------------------------------------------------------------------

def create_next_week_target(
    df: pd.DataFrame,
    base_col: str = "total_points",
    target_col: str = "target_points_next_gw",
) -> pd.DataFrame:
    """
    Create one-week-ahead target by shifting base_col forward (GW t+1).

    For each player_id + season:
        target_points_next_gw at gw = total_points at gw+1.

    Rows with no next-gameweek (last GW per player/season) are dropped.
    """
    df = df.copy()
    group_cols = ["player_id", "season"]

    df[target_col] = df.groupby(group_cols)[base_col].shift(-1)

    initial_rows = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows (no next GW to predict)")

    return df


# -------------------------------------------------------------------
# 7. High-level pipeline: build feature table + save CSV
# -------------------------------------------------------------------

def build_fpl_feature_table(
    seasons: List[str],
    window: int = 4,
    save_path: str = "data/fpl_features_last4_elo.csv",
) -> pd.DataFrame:
    """
    High-level function that:
      1. Loads raw FPL data from GitHub (merged_gw.csv).
      2. Adds player_id and sorts for time-series.
      3. Adds player form features (last 4 GWs).
      4. Adds team & opponent context (ATT/DEF form, last 4 GWs) + TEAM_elo / OPP_elo.
      5. Adds per-minute rate features.
      6. Adds availability flag (did_not_play_last_game).
      7. Adds meta and price features.
      8. Creates one-week-ahead target (target_points_next_gw).
      9. Saves the resulting DataFrame to CSV.

    Returns:
      df_features : pd.DataFrame
          Full feature table (one row per player-gameweek with target).
    """
    print("Loading raw data from GitHub...")
    raw_df, loaded_seasons = load_raw_data_from_urls(seasons)
    print(f"Successfully loaded seasons: {sorted(loaded_seasons)}")
    missing = set(seasons) - set(loaded_seasons)
    if missing:
        print(f"Warning: the following seasons could not be loaded and will be ignored: {sorted(missing)}")

    print(f"Loaded {len(raw_df)} rows total")

    # Filter to only loaded seasons
    raw_df = raw_df[raw_df["season"].isin(loaded_seasons)].copy()

    print("Creating player_id column...")
    df = create_player_id_column(raw_df)

    print("Sorting for time series...")
    df = sort_for_time_series(df)

    print(f"Adding player form features (last {window} games)...")
    df = add_player_form_features(df, window=window)

    print(f"Adding team and opponent context (ATT/DEF, last {window} games) + ELO...")
    df = add_team_and_opponent_context_with_elo(df, window=window)

    print("Adding per-minute rate features...")
    df = add_per_minute_rate_features(df)

    print("Adding did_not_play_last_game feature...")
    df = add_did_not_play_last_game(df)

    print("Adding meta and price features...")
    df = add_meta_and_price_features(df)

    print("Creating next-gameweek target (GW t+1)...")
    df = create_next_week_target(df)

    # Save to CSV
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nSaved feature table to: {save_path}")

    print(f"\nFinal feature table shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# -------------------------------------------------------------------
# 8. Example usage
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Building FPL Feature Table (Last 4 GWs + Team ELO)")
    print("=" * 60)

    # Example seasons (you can adjust)
    seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

    df_features = build_fpl_feature_table(
        seasons=seasons,
        window=4,
        save_path="data/fpl_features_last4_elo.csv",
    )

    # Quick sanity print
    print("\nSample columns:")
    print(df_features.columns.tolist()[:25], "..." if df_features.shape[1] > 25 else "")
    print("\nDone.")