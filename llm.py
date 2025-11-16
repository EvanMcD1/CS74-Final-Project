"""
FPL LLM Evaluation Script

Compares a Large Language Model (LLM) to numeric baselines
on the SAME engineered feature table (last 4 GWs + ELO + ATT/DEF).

- Uses feature_engineering.build_fpl_feature_table(seasons, window=4)
- Evaluates only on the test season (2023-24)
- Computes MAE / RMSE for the LLM on a sampled subset (for cost control)
"""

import os
import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from langchain_dartmouth.llms import ChatDartmouth

from feature_engineering import build_fpl_feature_table


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class FeatureConfig:
    target_points_column: str = "target_points_next_gw"
    player_name_column: str = "name"
    gameweek_column: str = "gw"
    season_column: str = "season"
    # optional manual feature subset – leave [] to use all numeric features
    selected_feature_columns: List[str] = field(default_factory=list)


@dataclass
class LlmConfig:
    model_name: str = "meta.llama-3.2-11b-vision-instruct"
    temperature: float = 0.0
    max_output_tokens: int = 128
    max_rows_to_score: int = 50   # cost/time control
    clamp_min: float = 0.0
    clamp_max: float = 40.0
    max_retries: int = 3
    retry_sleep_seconds: float = 0.6


# ============================================================
# 1. Build feature matrix + target from engineered table
# ============================================================

def build_feature_matrix_and_target(
    df_features: pd.DataFrame,
    feature_config: FeatureConfig,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    From engineered table (last-4 + ELO), build:
      - X: numeric feature matrix
      - y: target (next-week points)
      - feature_columns: list of feature names used
    """
    target_col = feature_config.target_points_column
    player_col = feature_config.player_name_column
    gw_col = feature_config.gameweek_column
    season_col = feature_config.season_column

    if target_col not in df_features.columns:
        raise ValueError(f"Target column '{target_col}' not found in feature table.")

    y = df_features[target_col]

    # Columns that should NOT be used as features
    exclude_cols = {
        target_col,
        player_col,
        gw_col,
        season_col,
        "player_id",
        "code",
    }

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

    if feature_config.selected_feature_columns:
        feature_columns = [
            c for c in feature_config.selected_feature_columns
            if c in numeric_cols and c not in exclude_cols
        ]
    else:
        feature_columns = [c for c in numeric_cols if c not in exclude_cols]

    X = df_features[feature_columns].copy()
    return X, y, feature_columns


# ============================================================
# 2. LLM wrapper
# ============================================================

class SimpleLLMCaller:
    def __init__(self, llm: ChatDartmouth, max_tokens: int):
        self.llm = llm
        self.max_tokens = max_tokens

    def _format_prompt(self, stats_json: str) -> str:
        return (
            "You are an elite Fantasy Premier League (FPL) analytics model.\n"
            "Your task is to predict a player's fantasy 'total_points' for the next gameweek (GW t+1),\n"
            "using ONLY the structured features provided in JSON.\n\n"
            "The JSON row comes from an engineered feature table with:\n"
            "- FORM_*: last 4 games of player form (minutes, total points, goals, assists, start rate).\n"
            "- TEAM_*: last 4 games of team attack/defence (goals scored/conceded, ELO).\n"
            "- OPP_*: opponent strength over the last 4 games (defence, ELO).\n"
            "- RATE_*: per-minute scoring/assisting rates.\n"
            "- META_*: position, gameweek index, home/away.\n"
            "- did_not_play_last_game: indicator they did not appear last match.\n\n"
            "GUIDELINES:\n"
            "- Base your prediction strictly on these features. Do NOT use external football knowledge.\n"
            "- Heavily weight recent minutes, start rate, recent total_points, and team/opponent strength.\n"
            "- Produce realistic FPL predictions: most players 0–6 points, with occasional higher scores.\n\n"
            "OUTPUT FORMAT (MANDATORY):\n"
            "Return ONLY a valid JSON object with exactly these keys:\n"
            "  \"predicted_total_points\": <number>,\n"
            "  \"rationale\": <short explanation string>\n\n"
            "No markdown, no code blocks, no extra commentary.\n\n"
            "PLAYER FEATURE ROW (JSON):\n"
            f"{stats_json}\n"
        )

    def invoke(self, stats_json: str):
        prompt = self._format_prompt(stats_json)
        return self.llm.invoke(prompt)


def build_llm_chain(llm_config: LlmConfig) -> SimpleLLMCaller:
    chat_llm = ChatDartmouth(
        model_name=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_output_tokens,
    )
    return SimpleLLMCaller(chat_llm, max_tokens=llm_config.max_output_tokens)


# ============================================================
# 3. Parsing helpers
# ============================================================

def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove backticked blocks if any
        value = re.sub(r"```.*?```", "", value, flags=re.DOTALL)
        try:
            return float(value)
        except ValueError:
            m = re.search(r"-?\d+(?:\.\d+)?", value)
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    return None
    return None


def _coerce_and_clamp(
    value: object,
    lo: Optional[float],
    hi: Optional[float],
) -> Optional[float]:
    v = _safe_float(value)
    if v is None:
        return None
    if lo is not None:
        v = max(v, float(lo))
    if hi is not None:
        v = min(v, float(hi))
    return float(v)


# ============================================================
# 4. LLM scoring on test set
# ============================================================

def predict_points_with_llm(
    engine_chain: SimpleLLMCaller,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    feature_columns: List[str],
    feature_config: FeatureConfig,
    llm_config: LlmConfig,
) -> Dict[str, Any]:
    """
    Score a subset of test rows with the LLM and compute MAE/RMSE on that subset.
    """
    max_rows = llm_config.max_rows_to_score
    if len(X_test) <= max_rows:
        subset_indices = X_test.index
    else:
        subset_indices = X_test.sample(n=max_rows, random_state=42).index

    X_sub = X_test.loc[subset_indices]
    y_sub = y_test.loc[subset_indices]
    df_sub = df_test.loc[subset_indices]

    rows_json: List[str] = []
    for idx, row in X_sub.iterrows():
        # Numeric feature payload
        row_dict: Dict[str, Any] = {
            k: float(row[k]) if pd.notnull(row[k]) else 0.0
            for k in feature_columns
        }
        # Human-readable context (NOT used as numeric features)
        row_dict["_player_name"] = str(df_sub.loc[idx, feature_config.player_name_column])
        row_dict["_gameweek"] = int(df_sub.loc[idx, feature_config.gameweek_column])
        row_dict["_season"] = str(df_sub.loc[idx, feature_config.season_column])
        rows_json.append(pd.Series(row_dict).to_json())

    preds: List[float] = []
    rationales: List[str] = []
    lo, hi = llm_config.clamp_min, llm_config.clamp_max
    fallback = float(y_sub.mean())

    for row_json in rows_json:
        raw_text = ""
        parsed_value: Optional[float] = None
        rationale = ""

        for attempt in range(llm_config.max_retries):
            try:
                llm_out = engine_chain.invoke(row_json)
                raw_text = getattr(llm_out, "content", None) or str(llm_out)

                try:
                    obj = json.loads(raw_text)
                    candidate = obj.get("predicted_total_points", obj.get("prediction"))
                    rationale = obj.get("rationale", "")
                except Exception:
                    candidate = raw_text
                    rationale = ""

                parsed_value = _coerce_and_clamp(candidate, lo, hi)
                if parsed_value is not None:
                    break
            except Exception as e:
                raw_text = f"[error: {e}]"
                time.sleep(llm_config.retry_sleep_seconds * (attempt + 1))

        if parsed_value is None:
            parsed_value = _coerce_and_clamp(fallback, lo, hi)

        preds.append(float(parsed_value))
        rationales.append(rationale)

    preds = np.array(preds, dtype=float)
    mae = mean_absolute_error(y_sub, preds)
    rmse = root_mean_squared_error(y_sub, preds)

    # Preview table for poster / inspection
    preview = df_sub[[feature_config.player_name_column,
                      feature_config.season_column,
                      feature_config.gameweek_column,
                      feature_config.target_points_column]].copy()
    preview["llm_pred_total_points"] = preds

    return {
        "test_mae": mae,
        "test_rmse": rmse,
        "preview": preview.head(10),
        "rationales": rationales[:5],  # first few rationales (qualitative)
        "n_rows_scored": len(y_sub),
    }


# ============================================================
# 5. Main
# ============================================================

def main():
    load_dotenv()
    if not os.getenv("DARTMOUTH_CHAT_API_KEY"):
        raise RuntimeError("Please set DARTMOUTH_CHAT_API_KEY in your environment.")

    print("=" * 60)
    print("LLM Evaluation on Engineered FPL Features (Last 4 GWs + ELO)")
    print("Task: Predict GW t+1 FPL points from numeric features")
    print("=" * 60)

    # Use the same seasons / test season as XGBoost
    seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    test_seasons = ["2023-24"]

    # 1) Build engineered feature table once
    print("\nBuilding engineered feature table via build_fpl_feature_table()...")
    df_features = build_fpl_feature_table(seasons=seasons, window=4, save_path=None)
    print(f"Feature table: {df_features.shape[0]} rows, {df_features.shape[1]} columns")

    feature_config = FeatureConfig()

    # 2) Filter to test season only (we compare LLM to XGBoost on this season)
    df_test = df_features[df_features[feature_config.season_column].isin(test_seasons)].reset_index(drop=True)

    # 3) Build X_test, y_test, feature_columns from test table
    X_test_all, y_test_all, feature_columns = build_feature_matrix_and_target(df_test, feature_config)

    print(f"\nTest season rows (for potential LLM scoring): {len(df_test)}")
    print(f"Number of numeric features: {len(feature_columns)}")

    # 4) Build LLM client
    llm_config = LlmConfig()
    llm_chain = build_llm_chain(llm_config)

    # 5) Score a subset with the LLM
    print("\nScoring test rows with LLM...")
    results = predict_points_with_llm(
        engine_chain=llm_chain,
        X_test=X_test_all,
        y_test=y_test_all,
        df_test=df_test,
        feature_columns=feature_columns,
        feature_config=feature_config,
        llm_config=llm_config,
    )

    print("\n============================================================")
    print("LLM Performance on Test Season (subset)")
    print("============================================================")
    print(f"Rows scored by LLM: {results['n_rows_scored']}")
    print(f"LLM Test MAE (points):  {results['test_mae']:.3f}")
    print(f"LLM Test RMSE (points): {results['test_rmse']:.3f}")

    print("\nPreview of scored rows (test season):")
    print(results["preview"].to_string(index=False))

    print("\nSample rationales:")
    for i, r in enumerate(results["rationales"], start=1):
        print(f"  [{i}] {r}")


if __name__ == "__main__":
    main()