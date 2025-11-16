import numpy as np
import math

def rolling_calculator(df, weeks):
    result_df = df.copy()

    stat_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    if "GW" in stat_cols:
        stat_cols.remove("GW")

    # Sort by player name and then GW
    result_df = result_df.sort_values(['name', 'GW'])
    
    # Loop through each stat column and calculate rolling totals per player
    for col in stat_cols:
        result_df[f'{col}_last_{weeks}_games'] = (
            result_df.groupby('name')[col]
            .rolling(window=weeks, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
    
    return result_df, stat_cols


def probability(rating1, rating2):
    return 1.0 / (1 + math.pow(10, (rating1 - rating2) / 400.0))

def elo_update(win_rating, los_rating, K=32):
    """
    Update Elo ratings where one side 'wins' (offense scores or defense prevents)
    Returns: (new_winner_rating, new_loser_rating)
    """
    # Probability that the 'winner' would win
    P_winner = probability(los_rating, win_rating)
    # Probability that the 'loser' would win
    P_loser = probability(win_rating, los_rating)
    
    # Update ratings - use the parameter names!
    win_rating = win_rating + K * (1 - P_winner)
    los_rating = los_rating + K * (0 - P_loser)
    
    return win_rating, los_rating

def add_elo_columns(df, initial_elo=1500, k=32):
    df = df.sort_values('MatchDate').reset_index(drop=True)

    elo_ratings_offense = {}
    elo_ratings_defense = {}
    teams = set(df["Team"].unique()) | set(df["Opponent"].unique())

    for team in teams:
        elo_ratings_offense[team] = initial_elo 
        elo_ratings_defense[team] = initial_elo 
       
    df["Team_Off_Elo"] = 0.0
    df["Team_Def_Elo"] = 0.0
    df["Opp_Off_Elo"] = 0.0
    df["Opp_Def_Elo"] = 0.0

    for idx in df.index:
        team = df.loc[idx, "Team"]
        opponent = df.loc[idx, "Opponent"]

        df.loc[idx, "Team_Off_Elo"] = elo_ratings_offense[team]
        df.loc[idx, "Team_Def_Elo"] = elo_ratings_defense[team]
        df.loc[idx, "Opp_Off_Elo"] = elo_ratings_offense[opponent]
        df.loc[idx, "Opp_Def_Elo"] = elo_ratings_defense[opponent]

        team_goals = df.loc[idx, "team_goals"]
        opp_goals = df.loc[idx, "team_goals_allowed"]

        for _ in range(int(team_goals)):
            elo_ratings_offense[team], elo_ratings_defense[opponent] = elo_update(
                elo_ratings_offense[team], 
                elo_ratings_defense[opponent], 
                k
            )

        for _ in range(int(opp_goals)):
            elo_ratings_offense[opponent], elo_ratings_defense[team] = elo_update(
                elo_ratings_offense[opponent], 
                elo_ratings_defense[team], 
                k
            )
    return df
 