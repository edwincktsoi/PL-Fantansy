import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
import plotly.express as px

# --- Constants ---
FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
DIFFICULTY_MULTIPLIER = {
    1: 1.20,  # Very Easy
    2: 1.10,  # Easy
    3: 1.00,  # Average
    4: 0.85,  # Hard
    5: 0.70   # Very Hard
}
POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

pd.options.mode.chained_assignment = None
def plot_player_performance_timeseries(player_names, players_df):
    """
    Fetches historical gameweek data for selected players and 
    returns cumulative and weekly performance charts.
    """
    all_history = []
    
    for name in player_names:
        # Find the player ID from the main dataframe
        player_row = players_df[players_df['name'] == name]
        if player_row.empty:
            continue
            
        p_id = player_row.iloc[0]['id']
        
        # Fetch individual player history from FPL API
        url = f"https://fantasy.premierleague.com/api/element-summary/{p_id}/"
        try:
            r = requests.get(url)
            data = r.json()
            history = pd.DataFrame(data['history'])
            
            if not history.empty:
                history['player_name'] = name
                # Create cumulative points column
                history['cumulative_points'] = history['total_points'].cumsum()
                all_history.append(history)
        except Exception as e:
            st.error(f"Error fetching data for {name}: {e}")

    if not all_history:
        return None, None

    df_plot = pd.concat(all_history)

    # 1. Cumulative Points Chart
    fig_cum = px.line(
        df_plot, x='round', y='cumulative_points', color='player_name',
        title="Season Points Progression",
        labels={'round': 'Gameweek', 'cumulative_points': 'Total Points'},
        markers=True, template="plotly_white"
    )

    # 2. Weekly Points Chart
    fig_weekly = px.bar(
        df_plot, x='round', y='total_points', color='player_name',
        title="Weekly Performance", barmode='group',
        labels={'round': 'Gameweek', 'total_points': 'GW Points'},
        template="plotly_white"
    )

    return fig_cum, fig_weekly


def plot_points_vs_cost(players_df, positions_df):
    """Plot total FPL points vs cost separated by position using facets."""
    df = players_df.copy()
    df['cost_m'] = df['now_cost'] / 10
    
    # Define the order for the facets (better presentation)
    POSITION_ORDER = ['GKP', 'DEF', 'MID', 'FWD']
    df['position_name'] = df['element_type'].map(POSITION_MAP)
    df['position_name'] = pd.Categorical(df['position_name'], categories=POSITION_ORDER, ordered=True)
    df = df.sort_values('position_name')
    
    # Create a faceted figure, separated by position
    fig = px.scatter(
        df,
        x='cost_m',
        y='total_points',
        color='position_name', 
        facet_col='position_name', # <-- KEY CHANGE: Facet the plot by position
        facet_col_wrap=2,          # Display 2 plots per row
        hover_name='name',
        trendline='ols',
        labels={'cost_m': 'Cost (£m)', 'total_points': 'Total Points'},
        title='Total Points vs Cost - Separated by Position'
    )
    
    # Clean up the facet titles (remove 'position_name=')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    # Set the same y-axis range for all plots for easy comparison
    fig.update_yaxes(matches=None, showticklabels=True)
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=800) # Increase height for better viewing of 4 plots
    
    return fig

def plot_cumulative_points(age_limit=26, position='Midfielder', min_cum_points=50):
    """Plot cumulative points history for specific player demographics."""
    players_df, _ = load_fpl_data()
    # Note: Streamlit cannot display this plot unless fig.show() is replaced with 'return fig'
    st.warning("This function uses fig.show() and cannot be displayed directly in Streamlit without modification.")
    return None # Return None to prevent crashing
# ---------------------------
# 1️⃣ DATA FETCHING & CACHING
# ---------------------------

from functools import lru_cache

@lru_cache(maxsize=1024)
def fetch_player_history(player_id):
    """Cached fetch of player history to speed up repeated calls."""
    try:
        res = requests.get(f"{FPL_BASE_URL}element-summary/{player_id}/", timeout=10)
        if res.status_code != 200:
            return []
        return res.json().get("history", [])
    except Exception:
        return []

def download_fpl_data(save_dir="data/fpl", save_csv=True):
    """Downloads FPL bootstrap data and saves locally."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    url = f"{FPL_BASE_URL}bootstrap-static/"
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    # Save raw JSON
    json_path = save_path / "bootstrap_static.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    if save_csv:
        pd.DataFrame(data["elements"]).to_csv(save_path / "players.csv", index=False)
        pd.DataFrame(data["teams"]).to_csv(save_path / "teams.csv", index=False)
        pd.DataFrame(data["element_types"]).to_csv(save_path / "positions.csv", index=False)

    print(f"✅ FPL data downloaded to: {save_path.resolve()}")
    return data

import requests
import pandas as pd

def load_fpl_data():
    """Load FPL data from API."""
    
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

    # --- Injury table (optional but useful for medical tab) ---
    injuries_df = (
        players_df
        .query("status not in ['a','u','n']")
        [["name", "team", "position_name", "status_readable", "news", "chance_of_playing_this_round"]]
        .sort_values(["team", "chance_of_playing_this_round"])
    )

    return players_df, injuries_df, data['element_types']

# ---------------------------
# 2️⃣ PLAYER FORECASTING (DGW + BGW aware)
# ---------------------------

def fetch_and_forecast_players():
    """Fetch FPL players, calculate form, and apply DGW/BGW logic."""
    bootstrap = requests.get(f"{FPL_BASE_URL}bootstrap-static/").json()
    players_df = pd.DataFrame(bootstrap["elements"])
    teams_df   = pd.DataFrame(bootstrap["teams"])
    events_df  = pd.DataFrame(bootstrap["events"])

    # Team mapping
    team_name_map = teams_df.set_index("id")["name"].to_dict()
    players_df["team_id"] = players_df["team"]
    players_df["team_name"] = players_df["team_id"].map(team_name_map)
    players_df["name"] = players_df["first_name"] + " " + players_df["second_name"]

    # Next GW
    next_event = events_df[events_df["is_next"] == True]
    next_gw_id = int(next_event.iloc[0]["id"]) if not next_event.empty else 38
    print(f"Forecasting for GW{next_gw_id}...")

    # Fixtures
    fixtures = requests.get(f"{FPL_BASE_URL}fixtures/").json()
    team_fixtures = defaultdict(list)
    for f in fixtures:
        if f["event"] == next_gw_id:
            team_fixtures[f["team_h"]].append(DIFFICULTY_MULTIPLIER[f["team_h_difficulty"]])
            team_fixtures[f["team_a"]].append(DIFFICULTY_MULTIPLIER[f["team_a_difficulty"]])

    # DGW / BGW logic
    team_expected_multiplier = {}
    for team_id in range(1, 21):
        diffs = team_fixtures.get(team_id, [])
        if not diffs:  # Blank GW
            team_expected_multiplier[team_id] = 0.0
        else:  # Sum over all fixtures (DGW safe)
            team_expected_multiplier[team_id] = sum(diffs)

    # Player filtering
    players_df["chance_of_playing_this_round"] = players_df["chance_of_playing_this_round"].fillna(100)
    players_df = players_df[players_df["chance_of_playing_this_round"] > 70]
    relevant_players = players_df[players_df["total_points"] >= 10]["id"].tolist()

    # Last 3 GWs
    all_recent_points = []
    with requests.Session() as session:
        for pid in tqdm(relevant_players, desc="Processing Players"):
            history = fetch_player_history(pid)
            last_3 = history[-3:]
            for i, gw in enumerate(reversed(last_3)):
                all_recent_points.append({
                    "player_id": pid,
                    "gw_rank": i + 1,
                    "points": gw["total_points"]
                })

    if all_recent_points:
        history_df = pd.DataFrame(all_recent_points)
        pivot = history_df.pivot(index="player_id", columns="gw_rank", values="points").rename(
            columns={1:"gw_1_points",2:"gw_2_points",3:"gw_3_points"}
        )
        players_df = players_df.merge(pivot, left_on="id", right_index=True, how="left")

    for col in ["gw_1_points","gw_2_points","gw_3_points"]:
        players_df[col] = players_df.get(col, 0).fillna(0)

    # Form
    players_df["base_form_points"] = (
        0.45 * players_df["gw_1_points"] +
        0.35 * players_df["gw_2_points"] +
        0.20 * players_df["gw_3_points"]
    )

    # Minutes probability + DGW/BGW multiplier
    players_df["expected_minutes_factor"] = players_df["chance_of_playing_this_round"] / 100
    players_df["fixture_multiplier"] = players_df["team_id"].map(team_expected_multiplier)

    # Projected points
    players_df["projected_points"] = (
        players_df["base_form_points"] *
        players_df["fixture_multiplier"] *
        players_df["expected_minutes_factor"]
    ).apply(np.floor)

    players_df["now_cost"] = players_df["now_cost"] / 10
    players_df["team"] = players_df["team_name"]

    return players_df.sort_values(by="projected_points", ascending=False).reset_index(drop=True)

# ---------------------------
# 3️⃣ OPTIMIZER
# ---------------------------

def optimize_fpl_team(players_df, players_to_keep=None, players_to_exclude=None):
    """Optimizes 15-man FPL squad using LP."""
    players_to_keep = players_to_keep or []
    players_to_exclude = players_to_exclude or []

    # Use safe LP variable names
    players_df['selected'] = players_df.apply(
        lambda row: LpVariable(f"p_{row.id}", cat='Binary'), axis=1
    )

    model = LpProblem("FPL_Squad_Optimizer", LpMaximize)
    model += lpSum(players_df['selected'] * players_df['projected_points'])

    # Budget & squad size
    model += lpSum(players_df['selected'] * players_df['now_cost']) <= 100
    model += lpSum(players_df['selected']) == 15

    # Positional constraints: 2 GKP, 5 DEF, 5 MID, 3 FWD
    for pos, count in {1:2,2:5,3:5,4:3}.items():
        model += lpSum(players_df.loc[players_df['element_type']==pos,'selected']) == count

    # Max 3 players per team
    for team_id in players_df['team'].unique():
        model += lpSum(players_df.loc[players_df['team']==team_id,'selected']) <= 3

    # Forced Keep / Exclude
    for name in players_to_keep:
        if name in players_df['name'].values:
            model += players_df.loc[players_df['name']==name,'selected'].values[0] == 1
    for name in players_to_exclude:
        if name in players_df['name'].values:
            model += players_df.loc[players_df['name']==name,'selected'].values[0] == 0

    model.solve()
    print("Optimization Status:", LpStatus[model.status])

    players_df['is_picked'] = players_df['selected'].apply(lambda var: var.varValue)
    selected_df = players_df[players_df['is_picked']==1].copy()

    fig = _plot_optimization_results(players_df, selected_df)
    return selected_df['name'].tolist(), model, fig

def _plot_optimization_results(all_players, selected_players):
    """Plot optimized squad results."""
    POSITION_MAP = {1:'GKP',2:'DEF',3:'MID',4:'FWD'}
    CATEGORY_ORDER = ['GKP','DEF','MID','FWD','Unselected Pool']
    COLOR_MAP = {'GKP':'deepskyblue','DEF':'green','MID':'gold','FWD':'red','Unselected Pool':'lightgrey'}
    SIZE_MAP = {'GKP':6,'DEF':6,'MID':6,'FWD':6,'Unselected Pool':3}

    df = all_players.copy()
    df['position_name'] = df['element_type'].map(POSITION_MAP)
    df['is_selected_status'] = np.where(df['is_picked']==1, df['position_name'], 'Unselected Pool')
    df['marker_size'] = df['is_selected_status'].map(SIZE_MAP)

    fig = px.scatter(
        df,
        x='now_cost', y='projected_points',
        color='is_selected_status',
        size='marker_size',
        size_max=8,
        color_discrete_map=COLOR_MAP,
        category_orders={'is_selected_status': CATEGORY_ORDER},
        hover_name='name',
        hover_data={'now_cost':':.1f','projected_points':':.1f','team':True,'total_points':True,'form':':.1f'},
        labels={'now_cost':'Cost (£m)','projected_points':'Projected Points','is_selected_status':'Squad Status'},
        title='Optimized Squad Selection: Cost vs Projected Points'
    )

    fig.update_traces(opacity=0.35, selector=dict(name='Unselected Pool'))
    fig.update_traces(opacity=1.0, selector=lambda t: t.name!='Unselected Pool')
    fig.update_layout(hovermode='closest', legend_title_text='Selection', xaxis=dict(range=[3.5,20]), template='plotly_white')
    return fig

# ---------------------------
# 4️⃣ STARTING XI
# ---------------------------

def optimize_starting_11(squad_df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes starting XI from selected 15-man squad."""
    if len(squad_df) != 15:
        print(f"Warning: Squad has {len(squad_df)} players. Optimization requires 15.")
        return pd.DataFrame()

    squad_df['starter'] = squad_df['name'].apply(lambda name: LpVariable(f"starter_{name.replace(' ','_')}", cat='Binary'))
    model = LpProblem("FPL_Starting_XI", LpMaximize)
    model += lpSum(squad_df['starter']*squad_df['projected_points'])

    # Constraints
    model += lpSum(squad_df['starter'])==11
    model += lpSum(squad_df[squad_df['element_type']==1]['starter'])==1  # GKP
    for pos, min_c, max_c in [(2,3,5),(3,2,5),(4,1,3)]:
        starters = squad_df[squad_df['element_type']==pos]['starter']
        model += lpSum(starters)>=min_c
        model += lpSum(starters)<=max_c

    model.solve()
    squad_df['is_starter'] = squad_df['starter'].apply(lambda x: x.varValue)
    squad_df['position_name'] = squad_df['element_type'].map(POSITION_MAP)

    starters_df = squad_df[squad_df['is_starter']==1].copy()
    starters_df = starters_df.sort_values(by='projected_points', ascending=False)
    starters_df['role'] = 'Starter'
    if len(starters_df)>=2:
        starters_df.iloc[0, starters_df.columns.get_loc('role')] = 'Captain (C)'
        starters_df.iloc[1, starters_df.columns.get_loc('role')] = 'Vice-Captain (VC)'

    starters_df = starters_df.sort_values(by=['element_type','projected_points'], ascending=[True,False])
    starters_df['projected_points'] = np.floor(starters_df['projected_points'])
    return starters_df[[
        'name','position_name','role','team','now_cost','selected_by_percent',
        'projected_points','fixture_multiplier','form','points_per_game',
        'total_points','gw_1_points','gw_2_points','gw_3_points','chance_of_playing_this_round'
    ]]
