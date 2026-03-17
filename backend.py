import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
import plotly.express as px
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
DIFFICULTY_MULTIPLIER = {
    1: 1.20,  # Very Easy
    2: 1.10,  # Easy
    3: 1.00,  # Average
    4: 0.85,  # Hard
    5: 0.70   # Very Hard
}
POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
POSITION_CONSTRAINTS = {1: 2, 2: 5, 3: 5, 4: 3}  # required count per position
BUDGET = 100.0
SQUAD_SIZE = 15
XI_SIZE = 11
MAX_PER_TEAM = 3
MIN_CHANCE_OF_PLAYING = 70
MIN_TOTAL_POINTS = 10

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    """GET with basic error handling."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_player_history(player_id: int) -> list[dict]:
    data = _get(f"{FPL_BASE_URL}element-summary/{player_id}/")
    return data.get("history", [])

def download_fpl_data(save=True):
    """Load FPL data from API and save as Parquet."""
    
    data = requests.get(f"{FPL_BASE_URL}bootstrap-static/").json()
    
    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])

    # --- Team mapping ---
    team_name_map = teams_df.set_index('id')['name'].to_dict()
    players_df['team'] = players_df['team'].map(team_name_map)

    # --- Position mapping ---
    players_df['position_name'] = players_df['element_type'].map(POSITION_MAP)

    # --- Player name ---
    players_df['name'] = players_df['first_name'] + ' ' + players_df['second_name']

    # --- Age ---
    players_df['birth_date'] = pd.to_datetime(players_df['birth_date'], errors='coerce')
    players_df['age'] = (pd.Timestamp.now() - players_df['birth_date']).dt.days // 365

    # --- Status mapping ---
    status_map = {
        "a": "Available",
        "d": "Doubtful",
        "i": "Injured",
        "s": "Suspended",
        "u": "Unavailable",
        "n": "Not in Squad"
    }

    players_df["status_readable"] = players_df["status"].map(status_map)

    # --- Injury table ---
    injuries_df = (
        players_df
        .query("status not in ['a','u','n']")
        [["name", "team", "position_name", "status_readable", "news", "chance_of_playing_this_round"]]
        .sort_values(["team", "chance_of_playing_this_round"])
    )

    # --- SAVE ---
    if save:
        input_path = Path("./input")
        input_path.mkdir(exist_ok=True)

        players_df.to_parquet(input_path / "players.parquet", index=False)
        injuries_df.to_parquet(input_path / "injuries.parquet", index=False)
        teams_df.to_parquet(input_path / "teams.parquet", index=False)

    return players_df, injuries_df, data['element_types']



##################
###Optimazation###
##################
# ── Forecast ───────────────────────────────────────────────────────────────────

def fetch_and_forecast_players() -> pd.DataFrame:
    """Fetch players, build recent-form score, apply fixture multiplier."""
    log.info("Fetching bootstrap + fixtures...")
    bootstrap  = _get(f"{FPL_BASE_URL}bootstrap-static/")
    players_df = pd.DataFrame(bootstrap["elements"])
    teams_df   = pd.DataFrame(bootstrap["teams"])
    events_df  = pd.DataFrame(bootstrap["events"])

    team_name_map = teams_df.set_index("id")["name"].to_dict()
    players_df["team_id"]   = players_df["team"]
    players_df["team_name"] = players_df["team_id"].map(team_name_map)
    players_df["name"]      = players_df["first_name"] + " " + players_df["second_name"]

    next_event = events_df[events_df["is_next"] == True]
    next_gw_id = int(next_event.iloc[0]["id"]) if not next_event.empty else 38
    log.info(f"Forecasting for GW{next_gw_id}...")

    # Build fixture multiplier per team
    fixtures = _get(f"{FPL_BASE_URL}fixtures/")
    team_fixtures: dict[int, list[float]] = defaultdict(list)
    for f in fixtures:
        if f["event"] == next_gw_id:
            team_fixtures[f["team_h"]].append(DIFFICULTY_MULTIPLIER[f["team_h_difficulty"]])
            team_fixtures[f["team_a"]].append(DIFFICULTY_MULTIPLIER[f["team_a_difficulty"]])

    team_multiplier = {
        tid: sum(team_fixtures[tid]) if team_fixtures.get(tid) else 0.0
        for tid in range(1, 21)
    }

    # Filter: only available players with enough history
    players_df["chance_of_playing_this_round"] = (
        players_df["chance_of_playing_this_round"].fillna(100)
    )
    players_df = players_df[
        (players_df["chance_of_playing_this_round"] > MIN_CHANCE_OF_PLAYING) &
        (players_df["total_points"] >= MIN_TOTAL_POINTS)
    ]

    # Fetch last-3 GW history in parallel
    log.info(f"Fetching history for {len(players_df)} players...")
    records = []
    with requests.Session():
        for pid in tqdm(players_df["id"].tolist(), desc="Player history"):
            for rank, gw in enumerate(reversed(fetch_player_history(pid)[-3:]), start=1):
                records.append({"player_id": pid, "gw_rank": rank, "points": gw["total_points"]})

    if records:
        pivot = (
            pd.DataFrame(records)
            .pivot(index="player_id", columns="gw_rank", values="points")
            .rename(columns={1: "gw_1_points", 2: "gw_2_points", 3: "gw_3_points"})
        )
        players_df = players_df.merge(pivot, left_on="id", right_index=True, how="left")

    for col in ["gw_1_points", "gw_2_points", "gw_3_points"]:
        players_df[col] = players_df.get(col, pd.Series(0, index=players_df.index)).fillna(0)

    # Weighted recent form
    players_df["base_form_points"] = (
        0.45 * players_df["gw_1_points"] +
        0.35 * players_df["gw_2_points"] +
        0.20 * players_df["gw_3_points"]
    )
    players_df["expected_minutes_factor"] = players_df["chance_of_playing_this_round"] / 100
    players_df["fixture_multiplier"]      = players_df["team_id"].map(team_multiplier)
    players_df["projected_points"]        = np.floor(
        players_df["base_form_points"] *
        players_df["fixture_multiplier"] *
        players_df["expected_minutes_factor"]
    )
    players_df["now_cost"] = players_df["now_cost"] / 10
    players_df["team"]     = players_df["team_name"]

    return players_df.sort_values("projected_points", ascending=False).reset_index(drop=True)


# ── Squad Optimizer ────────────────────────────────────────────────────────────

def optimize_fpl_team(
    players_df: pd.DataFrame,
    players_to_keep: list[str] | None = None,
    players_to_exclude: list[str] | None = None,
) -> tuple[list[str], object, object]:
    """LP optimisation: select 15-man squad maximising projected points."""
    players_to_keep    = players_to_keep    or []
    players_to_exclude = players_to_exclude or []

    df = players_df.copy()
    df["selected"] = df.apply(lambda r: LpVariable(f"p_{r.id}", cat="Binary"), axis=1)

    model = LpProblem("FPL_Squad_Optimizer", LpMaximize)
    model += lpSum(df["selected"] * df["projected_points"])
    model += lpSum(df["selected"] * df["now_cost"]) <= BUDGET
    model += lpSum(df["selected"]) == SQUAD_SIZE

    for pos, count in POSITION_CONSTRAINTS.items():
        model += lpSum(df.loc[df["element_type"] == pos, "selected"]) == count

    for team in df["team"].unique():
        model += lpSum(df.loc[df["team"] == team, "selected"]) <= MAX_PER_TEAM

    for name in players_to_keep:
        mask = df["name"] == name
        if mask.any():
            model += df.loc[mask, "selected"].values[0] == 1
        else:
            log.warning(f"players_to_keep: '{name}' not found — skipping.")

    for name in players_to_exclude:
        mask = df["name"] == name
        if mask.any():
            model += df.loc[mask, "selected"].values[0] == 0
        else:
            log.warning(f"players_to_exclude: '{name}' not found — skipping.")

    model.solve()
    log.info(f"Optimisation status: {LpStatus[model.status]}")

    df["is_picked"] = df["selected"].apply(lambda v: v.varValue)
    selected_df = df[df["is_picked"] == 1].copy()
    fig = _plot_optimization_results(df, selected_df)

    return selected_df["name"].tolist(), model, fig


def _plot_optimization_results(all_df: pd.DataFrame, selected_df: pd.DataFrame):
    CATEGORY_ORDER = ["GKP", "DEF", "MID", "FWD", "Unselected Pool"]
    COLOR_MAP = {"GKP": "deepskyblue", "DEF": "green", "MID": "gold",
                 "FWD": "red", "Unselected Pool": "lightgrey"}
    SIZE_MAP  = {k: 6 for k in ["GKP","DEF","MID","FWD"]} | {"Unselected Pool": 3}

    df = all_df.copy()
    df["position_name"]      = df["element_type"].map(POSITION_MAP)
    df["is_selected_status"] = np.where(df["is_picked"] == 1, df["position_name"], "Unselected Pool")
    df["marker_size"]        = df["is_selected_status"].map(SIZE_MAP)

    fig = px.scatter(
        df, x="now_cost", y="projected_points",
        color="is_selected_status", size="marker_size", size_max=8,
        color_discrete_map=COLOR_MAP,
        category_orders={"is_selected_status": CATEGORY_ORDER},
        hover_name="name",
        hover_data={"now_cost": ":.1f", "projected_points": ":.1f",
                    "team": True, "total_points": True, "form": ":.1f"},
        labels={"now_cost": "Cost (£m)", "projected_points": "Projected Points",
                "is_selected_status": "Squad Status"},
        title="Optimised Squad Selection: Cost vs Projected Points",
    )
    fig.update_traces(opacity=0.35, selector=dict(name="Unselected Pool"))
    fig.update_traces(opacity=1.0,  selector=lambda t: t.name != "Unselected Pool")
    fig.update_layout(hovermode="closest", legend_title_text="Selection",
                      xaxis=dict(range=[3.5, 20]), template="plotly_white")
    return fig


# ── Starting XI Optimizer ──────────────────────────────────────────────────────

OUTFIELD_CONSTRAINTS = [(2, 3, 5), (3, 2, 5), (4, 1, 3)]  # (pos, min, max)
STARTING_XI_COLS = [
    "name", "position_name", "role", "team", "now_cost", "selected_by_percent",
    "projected_points", "fixture_multiplier", "form", "points_per_game",
    "total_points", "gw_1_points", "gw_2_points", "gw_3_points",
    "chance_of_playing_this_round",
]


def optimize_starting_11(squad_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the best XI from a 15-man squad via LP."""
    if len(squad_df) != SQUAD_SIZE:
        log.warning(f"Expected {SQUAD_SIZE} players, got {len(squad_df)}.")
        return pd.DataFrame()

    df = squad_df.copy()
    df["starter"] = df["name"].apply(
        lambda n: LpVariable(f"starter_{n.replace(' ', '_')}", cat="Binary")
    )

    model = LpProblem("FPL_Starting_XI", LpMaximize)
    model += lpSum(df["starter"] * df["projected_points"])
    model += lpSum(df["starter"]) == XI_SIZE
    model += lpSum(df[df["element_type"] == 1]["starter"]) == 1   # exactly 1 GK

    for pos, min_c, max_c in OUTFIELD_CONSTRAINTS:
        starters = df[df["element_type"] == pos]["starter"]
        model += lpSum(starters) >= min_c
        model += lpSum(starters) <= max_c

    model.solve()

    df["is_starter"]    = df["starter"].apply(lambda x: x.varValue)
    df["position_name"] = df["element_type"].map(POSITION_MAP)

    starters = (
        df[df["is_starter"] == 1]
        .copy()
        .sort_values("projected_points", ascending=False)
    )
    starters["role"] = "Starter"
    if len(starters) >= 1: starters.iloc[0, starters.columns.get_loc("role")] = "Captain (C)"
    if len(starters) >= 2: starters.iloc[1, starters.columns.get_loc("role")] = "Vice-Captain (VC)"

    starters = starters.sort_values(["element_type", "projected_points"], ascending=[True, False])
    starters["projected_points"] = np.floor(starters["projected_points"])
    return starters[STARTING_XI_COLS]


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_and_save_base_optimization(output_path: str = "base_squad_result.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: forecast → optimise → save CSV."""
    log.info("── Step 1/3: Fetching & forecasting players ──")
    forecast_df = fetch_and_forecast_players()

    log.info("── Step 2/3: Running squad optimisation ──")
    selected_names, model, _ = optimize_fpl_team(forecast_df)

    log.info("── Step 3/3: Selecting optimal Starting XI ──")
    selected_df = (
        forecast_df[forecast_df["name"].isin(selected_names)]
        .copy()
        .assign(position_name=lambda d: d["element_type"].map(POSITION_MAP))
    )
    starting_11 = optimize_starting_11(selected_df)
    selected_df["in_starting_11"] = selected_df["name"].isin(starting_11["name"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_parquet(output_path, index=False)

    log.info(f"✅ Status     : {LpStatus[model.status]}")
    log.info(f"💾 Saved to   : {output_path}")
    log.info(f"   Squad size : {len(selected_df)}/{SQUAD_SIZE}")
    log.info(f"   Squad cost : £{selected_df['now_cost'].sum():.1f}m")
    log.info(f"   Total pts  : {selected_df['projected_points'].sum():.0f}")
    log.info(f"   XI pts     : {starting_11['projected_points'].sum():.0f}")

    return selected_df, starting_11


if __name__ == "__main__":
    download_fpl_data(save=True)
    run_and_save_base_optimization("./input/base_squad_result.parquet")