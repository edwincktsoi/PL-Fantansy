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

try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import GoogleGenerativeAI
    from langchain_community.utilities import GoogleSearchAPIWrapper
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
DIFFICULTY_MULTIPLIER = {1: 1.20, 2: 1.10, 3: 1.00, 4: 0.85, 5: 0.70}
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_CONSTRAINTS = {1: 2, 2: 5, 3: 5, 4: 3}
BUDGET = 100.0
SQUAD_SIZE = 15
XI_SIZE = 11
MAX_PER_TEAM = 3
MIN_CHANCE_OF_PLAYING = 70
MIN_TOTAL_POINTS = 10

# Points awarded per goal by position
GOAL_POINTS = {1: 6, 2: 6, 3: 5, 4: 4}
# Points for keeping a clean sheet by position
CS_POINTS   = {1: 4, 2: 4, 3: 1, 4: 0}
# Fixed exponential decay weights for GW-1 … GW-5 (most recent first, sum=1)
FORM_DECAY_WEIGHTS = (0.35, 0.25, 0.20, 0.12, 0.08)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    """GET with basic error handling."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_player_history(player_id: int) -> list[dict]:
    data = _get(f"{FPL_BASE_URL}element-summary/{player_id}/")
    return data.get("history", [])


# ── Data Download ──────────────────────────────────────────────────────────────

def load_fpl_data(input_dir="./input"):
    """Read FPL data from Parquet files."""
    
    input_path = Path(input_dir)

    # --- Validation ---
    required_files = ["players.parquet", "injuries.parquet", "teams.parquet"]
    for file in required_files:
        if not (input_path / file).exists():
            raise FileNotFoundError(
                f"{file} not found in {input_dir}. Run load_fpl_data() first."
            )

    # --- Read ---
    players_df = pd.read_parquet(input_path / "players.parquet")
    injuries_df = pd.read_parquet(input_path / "injuries.parquet")
    teams_df = pd.read_parquet(input_path / "teams.parquet")

    return players_df, injuries_df, teams_df

    


# ── Forecast ───────────────────────────────────────────────────────────────────

def get_next_gw() -> int:
    """Return the next gameweek ID from the FPL API."""
    bootstrap = _get(f"{FPL_BASE_URL}bootstrap-static/")
    events_df = pd.DataFrame(bootstrap["events"])
    next_event = events_df[events_df["is_next"]]
    return int(next_event.iloc[0]["id"]) if not next_event.empty else 38


def get_next_fixture_info() -> pd.DataFrame:
    """Return one row per team for the next GW: team_id, team_name, opponent, difficulty, venue.
    Double-gameweek teams get one row with both fixtures concatenated."""
    bootstrap = _get(f"{FPL_BASE_URL}bootstrap-static/")
    teams_df  = pd.DataFrame(bootstrap["teams"])
    events_df = pd.DataFrame(bootstrap["events"])
    team_name_map = teams_df.set_index("id")["name"].to_dict()

    next_event = events_df[events_df["is_next"]]
    next_gw_id = int(next_event.iloc[0]["id"]) if not next_event.empty else 38

    fixtures = _get(f"{FPL_BASE_URL}fixtures/")
    rows: dict[int, dict] = {}
    for f in fixtures:
        if f["event"] != next_gw_id:
            continue
        for tid, opp_id, diff, venue in [
            (f["team_h"], f["team_a"], f["team_h_difficulty"], "H"),
            (f["team_a"], f["team_h"], f["team_a_difficulty"], "A"),
        ]:
            opp_name = team_name_map.get(opp_id, "?")
            entry = f"{opp_name} ({venue})"
            if tid in rows:
                rows[tid]["next_match"] += f" + {entry}"
                rows[tid]["difficulty"] = max(rows[tid]["difficulty"], diff)
            else:
                rows[tid] = {
                    "team_id":   tid,
                    "team_name": team_name_map.get(tid, "?"),
                    "next_match": entry,
                    "difficulty": diff,
                }
    if not rows:
        return pd.DataFrame(columns=["team_id", "team_name", "next_match", "difficulty"])
    return pd.DataFrame(rows.values())


def _gw_xp_vec(
    xg_s: pd.Series,
    xa_s: pd.Series,
    xgc_s: pd.Series,
    mins_s: pd.Series,
    etype_s: pd.Series,
) -> pd.Series:
    """Vectorised expected-points estimate for a single gameweek.

    Combines appearance bonus, attacking contribution (xG × goal pts + xA × 3),
    and for GKP/DEF a clean-sheet probability (Poisson model) minus concession
    penalty.  Returns 0 for players who didn't play that week.
    """
    appearance = np.where(mins_s >= 60, 2.0, np.where(mins_s > 0, 1.0, 0.0))
    goal_pts   = etype_s.map(GOAL_POINTS).fillna(4).astype(float)
    cs_pts     = etype_s.map(CS_POINTS).fillna(0).astype(float)
    is_def     = etype_s.isin([1, 2])

    attack  = xg_s * goal_pts + xa_s * 3.0
    # P(clean sheet) = Poisson P(0 goals conceded) = e^{-xGC}
    cs_prob = np.exp(-xgc_s.clip(lower=0))
    defense = np.where(is_def, cs_prob * cs_pts - (xgc_s / 2.0).clip(lower=-4), 0.0)

    xp = np.where(mins_s > 0, (appearance + attack + defense).clip(lower=0), 0.0)
    return pd.Series(xp, index=etype_s.index)


def fetch_and_forecast_players(
    progress_callback=None,
    xg_blend: float = 0.35,
    baseline_blend: float = 0.25,
) -> pd.DataFrame:
    """Fetch players and forecast next-GW points using a multi-signal model.

    Improvements over the simple 3-GW average:
    - 5-gameweek history with exponential decay weights
    - Per-GW xG/xA/xGC blended with raw points to smooth luck noise
    - Position-aware fixture multipliers (FWD/MID use opponent defensive
      strength; GKP/DEF use their own defensive strength vs opponent attack)
    - Minutes-per-start playing-time factor (replaces binary chance_of_playing)
    - Season PPG baseline blend to avoid overreacting to short hot/cold runs

    Args:
        progress_callback: optional callable(pct, text) for UI progress bar.
        xg_blend:        weight given to xP estimate vs raw points per GW [0-1].
        baseline_blend:  weight given to season PPG vs recent form [0-1].
    """
    log.info("Fetching bootstrap + fixtures...")
    bootstrap  = _get(f"{FPL_BASE_URL}bootstrap-static/")
    players_df = pd.DataFrame(bootstrap["elements"])
    teams_df   = pd.DataFrame(bootstrap["teams"])
    events_df  = pd.DataFrame(bootstrap["events"])

    team_name_map = teams_df.set_index("id")["name"].to_dict()
    players_df["team_id"]   = players_df["team"]
    players_df["team_name"] = players_df["team_id"].map(team_name_map)
    players_df["name"]      = players_df["first_name"] + " " + players_df["second_name"]

    next_event = events_df[events_df["is_next"]]
    next_gw_id = int(next_event.iloc[0]["id"]) if not next_event.empty else 38
    log.info(f"Forecasting for GW{next_gw_id}...")

    # ── Position-aware fixture multipliers ────────────────────────────────────
    # Use FPL team strength ratings to separately quantify how good a fixture is
    # for ATTACKERS (FWD/MID) vs DEFENDERS (GKP/DEF).
    #   attack_mult  = own_attack_strength / opponent_defence_strength
    #   defence_mult = own_defence_strength / opponent_attack_strength
    # Ratios > 1.0 → favourable; < 1.0 → unfavourable.  Clipped to [0.70, 1.30].
    def _str(col: str) -> dict:
        return teams_df.set_index("id")[col].fillna(1200).to_dict()

    t_atk_h = _str("strength_attack_home")
    t_atk_a = _str("strength_attack_away")
    t_def_h = _str("strength_defence_home")
    t_def_a = _str("strength_defence_away")

    fixtures = _get(f"{FPL_BASE_URL}fixtures/")
    atk_mults: dict[int, list[float]] = defaultdict(list)
    def_mults: dict[int, list[float]] = defaultdict(list)
    raw_mults: dict[int, list[float]] = defaultdict(list)   # for BGW detection

    for f in fixtures:
        if f["event"] != next_gw_id:
            continue
        h, a = f["team_h"], f["team_a"]

        h_atk = np.clip(t_atk_h.get(h, 1200) / max(t_def_a.get(a, 1200), 1), 0.70, 1.30)
        h_def = np.clip(t_def_h.get(h, 1200) / max(t_atk_a.get(a, 1200), 1), 0.70, 1.30)
        a_atk = np.clip(t_atk_a.get(a, 1200) / max(t_def_h.get(h, 1200), 1), 0.70, 1.30)
        a_def = np.clip(t_def_a.get(a, 1200) / max(t_atk_h.get(h, 1200), 1), 0.70, 1.30)

        atk_mults[h].append(float(h_atk)); def_mults[h].append(float(h_def))
        atk_mults[a].append(float(a_atk)); def_mults[a].append(float(a_def))
        raw_mults[h].append(DIFFICULTY_MULTIPLIER[f["team_h_difficulty"]])
        raw_mults[a].append(DIFFICULTY_MULTIPLIER[f["team_a_difficulty"]])

    all_team_ids = {f["team_h"] for f in fixtures} | {f["team_a"] for f in fixtures}
    # Sum over fixtures so DGW players get doubled value
    atk_mult_map = {tid: sum(atk_mults.get(tid, [0.0])) for tid in all_team_ids}
    def_mult_map = {tid: sum(def_mults.get(tid, [0.0])) for tid in all_team_ids}
    raw_mult_map = {tid: sum(raw_mults.get(tid, [0.0])) for tid in all_team_ids}

    # ── Filter players ─────────────────────────────────────────────────────────
    players_df["chance_of_playing_this_round"] = (
        players_df["chance_of_playing_this_round"].fillna(100)
    )
    # Early season: no points history yet, so relax the total-points gate
    pts_threshold = 0 if next_gw_id <= 5 else MIN_TOTAL_POINTS
    players_df = players_df[
        (players_df["chance_of_playing_this_round"] >= MIN_CHANCE_OF_PLAYING) &
        (players_df["total_points"] >= pts_threshold)
    ].copy()

    # ── Fetch GW history (last 5 GWs) ─────────────────────────────────────────
    n_gws      = len(FORM_DECAY_WEIGHTS)
    player_ids = players_df["id"].tolist()
    total      = len(player_ids)
    log.info(f"Fetching {n_gws}-GW history for {total} players...")

    records = []
    with requests.Session():
        for i, pid in enumerate(tqdm(player_ids, desc="Player history")):
            history = fetch_player_history(pid)
            recent  = list(reversed(history[-n_gws:]))   # rank 1 = most recent GW
            for rank, gw in enumerate(recent, start=1):
                records.append({
                    "player_id": pid,
                    "gw_rank":   rank,
                    "raw_pts":   int(gw.get("total_points", 0) or 0),
                    "xg":        float(gw.get("expected_goals", 0) or 0),
                    "xa":        float(gw.get("expected_assists", 0) or 0),
                    "xgc":       float(gw.get("expected_goals_conceded", 0) or 0),
                    "mins":      int(gw.get("minutes", 0) or 0),
                })
            if progress_callback:
                pct = int((i + 1) / total * 100)
                progress_callback(pct, f"🔍 Fetching player histories... {i + 1}/{total}")

    if records:
        rec_df = pd.DataFrame(records)

        def _pivot(col: str, suffix: str) -> pd.DataFrame:
            p = rec_df.pivot_table(index="player_id", columns="gw_rank",
                                   values=col, aggfunc="first")
            p.columns = [f"gw_{r}_{suffix}" for r in p.columns]
            return p

        for col, suffix in [("raw_pts", "points"), ("xg", "xg"),
                             ("xa", "xa"), ("xgc", "xgc"), ("mins", "mins")]:
            players_df = players_df.merge(_pivot(col, suffix),
                                          left_on="id", right_index=True, how="left")

    # Fill missing history with zeros
    for rank in range(1, n_gws + 1):
        for suffix in ("points", "xg", "xa", "xgc", "mins"):
            col = f"gw_{rank}_{suffix}"
            if col not in players_df.columns:
                players_df[col] = 0.0
            else:
                players_df[col] = pd.to_numeric(players_df[col], errors="coerce").fillna(0)

    # ── Blended form score (decay-weighted, xG-smoothed) ──────────────────────
    form_score = pd.Series(0.0, index=players_df.index)
    for rank, w in enumerate(FORM_DECAY_WEIGHTS, start=1):
        xp      = _gw_xp_vec(
            players_df[f"gw_{rank}_xg"],
            players_df[f"gw_{rank}_xa"],
            players_df[f"gw_{rank}_xgc"],
            players_df[f"gw_{rank}_mins"],
            players_df["element_type"],
        )
        blended = (1 - xg_blend) * players_df[f"gw_{rank}_points"] + xg_blend * xp
        form_score += w * blended

    players_df["base_form_points"] = form_score

    # ── Playing-time factor (minutes-per-start, not binary chance_of_playing) ──
    starts   = pd.to_numeric(players_df.get("starts", 0), errors="coerce").fillna(0)
    minutes  = pd.to_numeric(players_df["minutes"], errors="coerce").fillna(0)
    avg_mins = np.where(starts > 0, minutes / starts, np.where(minutes > 0, 90.0, 0.0))
    players_df["playing_time_factor"] = np.clip(avg_mins / 90.0, 0.0, 1.0)

    # ── Position-aware fixture multiplier ─────────────────────────────────────
    is_attacker = players_df["element_type"].isin([3, 4])
    players_df["_pos_fixture_mult"] = np.where(
        is_attacker,
        players_df["team_id"].map(atk_mult_map).fillna(0),
        players_df["team_id"].map(def_mult_map).fillna(0),
    )
    # Keep the raw combined multiplier for BGW detection and display
    players_df["fixture_multiplier"] = players_df["team_id"].map(raw_mult_map).fillna(0)

    # ── Season PPG baseline ────────────────────────────────────────────────────
    players_df["points_per_game"] = pd.to_numeric(
        players_df["points_per_game"], errors="coerce"
    ).fillna(0)

    # ── Count actual GWs of data per player ───────────────────────────────────
    # A GW counts only if the player registered minutes (not a blank row)
    n_avail = pd.Series(0, index=players_df.index)
    for rank in range(1, n_gws + 1):
        n_avail += (players_df[f"gw_{rank}_mins"] > 0).astype(int)
    players_df["n_gw_available"] = n_avail

    # ── Adaptive baseline blend ────────────────────────────────────────────────
    # Fewer real GWs → lean more on PPG (scales from baseline_blend up to 1.0)
    adaptive_blend = np.clip(
        baseline_blend + (1 - n_avail / n_gws) * (1 - baseline_blend),
        baseline_blend, 1.0,
    )

    # ── ep_next weight (FPL's own model, fades out once we have ≥3 GWs) ───────
    # 1.0 at GW0, 0.67 at GW1, 0.33 at GW2, 0.0 at GW3+
    ep_weight = np.clip((3 - n_avail) / 3, 0.0, 1.0)
    ep_next   = pd.to_numeric(players_df.get("ep_next", 0), errors="coerce").fillna(0)

    # ── Final projection ──────────────────────────────────────────────────────
    form_proj  = (
        players_df["base_form_points"]
        * players_df["playing_time_factor"]
        * players_df["_pos_fixture_mult"]
    )
    ppg_proj   = players_df["points_per_game"] * players_df["_pos_fixture_mult"]
    # Combined model signal with per-player adaptive blend
    model_proj = (1 - adaptive_blend) * form_proj + adaptive_blend * ppg_proj

    # ep_next is already fixture-adjusted by FPL; blend it in for early-season GWs
    players_df["projected_points"] = (1 - ep_weight) * model_proj + ep_weight * ep_next

    players_df["now_cost"] = players_df["now_cost"] / 10
    players_df["team"]     = players_df["team_name"]

    return players_df.sort_values("projected_points", ascending=False).reset_index(drop=True)


# ── Squad Optimizer ────────────────────────────────────────────────────────────

def optimize_fpl_team(
    players_df: pd.DataFrame,
    players_to_keep: list[str] | None = None,
    players_to_exclude: list[str] | None = None,
    budget: float = BUDGET,
) -> tuple[list[str], object, object]:
    """LP optimisation: select 15-man squad maximising projected points."""
    players_to_keep    = players_to_keep    or []
    players_to_exclude = players_to_exclude or []

    df = players_df.copy()
    df["selected"] = df.apply(lambda r: LpVariable(f"p_{r.id}", cat="Binary"), axis=1)

    model = LpProblem("FPL_Squad_Optimizer", LpMaximize)
    model += lpSum(df["selected"] * df["projected_points"])
    model += lpSum(df["selected"] * df["now_cost"]) <= budget
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

OUTFIELD_CONSTRAINTS = [(2, 3, 3), (3, 4, 5), (4, 2, 3)]  # (pos, min, max) — 3-back, mid-focus (3-5-2 or 3-4-3)
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


# ── Transfer Planner ───────────────────────────────────────────────────────────

def optimize_transfers(
    forecast_df: pd.DataFrame,
    current_squad_names: list[str],
    max_transfers: int = 1,
    budget: float = BUDGET,
) -> tuple[list[str], list[str], list[str]]:
    """
    Given a current 15-man squad, find the best squad achievable within
    `max_transfers` transfers while staying within budget.

    Returns:
        (new_squad_names, transfers_in, transfers_out)
    """
    df = forecast_df.copy()
    current_set = set(current_squad_names)

    df["selected"] = df.apply(lambda r: LpVariable(f"t_{r.id}", cat="Binary"), axis=1)
    df["is_current"] = df["name"].isin(current_set).astype(int)

    # Buy-in indicator: new player not in current squad
    df["buy_in"] = df.apply(
        lambda r: LpVariable(f"buy_{r.id}", cat="Binary"), axis=1
    )

    model = LpProblem("FPL_Transfer_Planner", LpMaximize)
    model += lpSum(df["selected"] * df["projected_points"])

    # Standard squad constraints
    model += lpSum(df["selected"] * df["now_cost"]) <= budget
    model += lpSum(df["selected"]) == SQUAD_SIZE
    for pos, count in POSITION_CONSTRAINTS.items():
        model += lpSum(df.loc[df["element_type"] == pos, "selected"]) == count
    for team in df["team"].unique():
        model += lpSum(df.loc[df["team"] == team, "selected"]) <= MAX_PER_TEAM

    # Transfer limit: a player is a "buy" only if selected AND not in current squad
    for _, row in df.iterrows():
        model += row["buy_in"] >= row["selected"] - row["is_current"]
        model += row["buy_in"] <= row["selected"]
        model += row["buy_in"] <= 1 - row["is_current"]
    model += lpSum(df["buy_in"]) <= max_transfers

    model.solve()
    log.info(f"Transfer optimisation status: {LpStatus[model.status]}")

    df["is_picked"] = df["selected"].apply(lambda v: v.varValue)
    new_squad = set(df[df["is_picked"] == 1]["name"].tolist())

    transfers_in  = sorted(new_squad - current_set)
    transfers_out = sorted(current_set - new_squad)
    return sorted(new_squad), transfers_in, transfers_out


# ── Season Trajectory ──────────────────────────────────────────────────────────

def plot_player_performance_timeseries(player_names: list, players_df: pd.DataFrame):
    """Fetch GW-by-GW history and return (fig_cumulative, fig_weekly) for selected players."""
    id_map = (
        players_df[players_df["name"].isin(player_names)]
        .set_index("name")["id"]
        .to_dict()
    )

    records = []
    with requests.Session() as session:
        for name, pid in id_map.items():
            try:
                resp = session.get(f"{FPL_BASE_URL}element-summary/{int(pid)}/", timeout=10)
                resp.raise_for_status()
                for gw in resp.json().get("history", []):
                    records.append({
                        "name": name,
                        "gw": gw["round"],
                        "points": gw["total_points"],
                    })
            except Exception:
                continue

    if not records:
        return None, None

    hist_df = pd.DataFrame(records).sort_values(["name", "gw"])
    hist_df["cumulative"] = hist_df.groupby("name")["points"].cumsum()

    fig_weekly = px.bar(
        hist_df, x="gw", y="points", color="name",
        barmode="group",
        labels={"gw": "Gameweek", "points": "Points", "name": "Player"},
        template="plotly_white", height=400,
    )

    fig_cum = px.line(
        hist_df, x="gw", y="cumulative", color="name",
        markers=True,
        labels={"gw": "Gameweek", "cumulative": "Cumulative Points", "name": "Player"},
        template="plotly_white", height=400,
    )

    return fig_cum, fig_weekly


# ── AI Advisor ────────────────────────────────────────────────────────────────

def _check_api_keys() -> bool:
    if not os.environ.get("GOOGLE_API_KEY"):
        log.warning("GOOGLE_API_KEY not set.")
        return False
    if not os.environ.get("GOOGLE_CSE_ID"):
        log.warning("GOOGLE_CSE_ID not set.")
        return False
    return True


def fpl_langchain_advisor(question: str, my_team_df: pd.DataFrame) -> str:
    """Answer a question about the user's squad using Gemini via LangChain."""
    if not _LANGCHAIN_AVAILABLE:
        return "AI Advisor requires langchain packages. Run: pip install langchain langchain-google-genai langchain-community"
    if not _check_api_keys():
        return "API keys missing. Set GOOGLE_API_KEY in your environment or Streamlit secrets."

    cols = [c for c in ["name", "total_points", "now_cost", "minutes", "form"] if c in my_team_df.columns]
    team_csv = my_team_df[cols].to_csv(index=False)

    template = PromptTemplate(
        input_variables=["question", "team_data"],
        template=(
            "You are an FPL expert. Given this team data:\n{team_data}\n\n"
            "Answer this question: {question}\n"
            "Provide concise, data-backed advice."
        ),
    )
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=template)
    return chain.run({"question": question, "team_data": team_csv})


if __name__ == "__main__":
    load_fpl_data(save=True)
    run_and_save_base_optimization("./input/base_squad_result.parquet")