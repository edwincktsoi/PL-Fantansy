import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from collections import defaultdict

# --- Constants ---
FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
DIFFICULTY_MULTIPLIER = {
    1: 1.20,  # Very Easy: +20%
    2: 1.10,  # Easy: +10%
    3: 1.00,  # Average: No change
    4: 0.85,  # Hard: -15%
    5: 0.70   # Very Hard: -30%
}
POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

# Suppress ChainedAssignment warnings
pd.options.mode.chained_assignment = None 

import requests
import pandas as pd
from pathlib import Path

FPL_BASE_URL = "https://fantasy.premierleague.com/api/"

import requests


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

def download_fpl_data(save_dir="data/fpl", save_csv=True):
    """
    Downloads FPL bootstrap data and saves it locally.
    
    Parameters
    ----------
    save_dir : str
        Directory to store downloaded files
    save_csv : bool
        Whether to also save parsed CSV files
        
    Returns
    -------
    dict
        Raw JSON response from FPL API
    """
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




def load_fpl_data():
    """Fetch data from FPL API and return player and position dataframes."""
    url = f"{FPL_BASE_URL}bootstrap-static/"
    data = requests.get(url).json()
    
    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    
    # Map Team Names
    team_name_map = teams_df.set_index('id')['name'].to_dict()
    players_df['team'] = players_df['team'].map(team_name_map)
    
    # Map Position Names (standardizing on 'position_name')
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players_df['position_name'] = players_df['element_type'].map(pos_map)
    
    # CREATE THE NAME COLUMN HERE
    players_df['name'] = players_df['first_name'] + ' ' + players_df['second_name']
    
    # Calculate Age
    players_df['birth_date'] = pd.to_datetime(players_df['birth_date'], errors='coerce')
    players_df['age'] = (pd.Timestamp.now() - players_df['birth_date']).dt.days // 365
    
    return players_df, data['element_types']

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
        hover_name='web_name',
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




def fetch_and_forecast_players():
    """
    Fetches FPL players, calculates recent form (last 3 GWs),
    and adjusts projected points using correctly applied fixture difficulty.
    """

    print("Fetching bootstrap data...")
    bootstrap = requests.get(f"{FPL_BASE_URL}bootstrap-static/").json()

    players_df = pd.DataFrame(bootstrap["elements"])
    teams_df   = pd.DataFrame(bootstrap["teams"])
    events_df  = pd.DataFrame(bootstrap["events"])

    # ─────────────────────────────────────────────
    # 1. Team mapping
    # ─────────────────────────────────────────────
    team_name_map = teams_df.set_index("id")["name"].to_dict()
    players_df["team_id"]   = players_df["team"]
    players_df["team_name"] = players_df["team_id"].map(team_name_map)

    players_df["name"] = players_df["first_name"] + " " + players_df["second_name"]

    # ─────────────────────────────────────────────
    # 2. Identify next GW
    # ─────────────────────────────────────────────
    next_event = events_df[events_df["is_next"] == True]
    next_gw_id = int(next_event.iloc[0]["id"]) if not next_event.empty else 38

    print(f"Forecasting for GW{next_gw_id}...")

    # ─────────────────────────────────────────────
    # 3. Fixture difficulty (CORRECTED)
    # ─────────────────────────────────────────────
    print("Fetching fixtures...")
    fixtures = requests.get(f"{FPL_BASE_URL}fixtures/").json()

    team_fixtures = defaultdict(list)

    for f in fixtures:
        if f["event"] == next_gw_id:
            team_fixtures[f["team_h"]].append(
                DIFFICULTY_MULTIPLIER[f["team_h_difficulty"]]
            )
            team_fixtures[f["team_a"]].append(
                DIFFICULTY_MULTIPLIER[f["team_a_difficulty"]]
            )

    # Final team multipliers
    team_multipliers = {}
    for team_id in range(1, 21):
        if team_id not in team_fixtures:
            # Blank GW
            team_multipliers[team_id] = 0.30
        else:
            # Average difficulty across fixtures (DGW-safe)
            team_multipliers[team_id] = np.mean(team_fixtures[team_id])

    # ─────────────────────────────────────────────
    # 4. Player filtering
    # ─────────────────────────────────────────────
    players_df["chance_of_playing_this_round"] = (
        players_df["chance_of_playing_this_round"].fillna(100)
    )

    players_df = players_df[
        players_df["chance_of_playing_this_round"] > 70
    ]

    # Only fetch history for active players
    relevant_players = players_df[
        players_df["total_points"] >= 10
    ]["id"].tolist()

    # ─────────────────────────────────────────────
    # 5. Fetch last 3 GWs history
    # ─────────────────────────────────────────────
    print(f"Fetching history for {len(relevant_players)} players...")
    all_recent_points = []

    with requests.Session() as session:
        for player_id in tqdm(relevant_players, desc="Processing Players"):
            try:
                res = session.get(f"{FPL_BASE_URL}element-summary/{player_id}/")
                if res.status_code != 200:
                    continue

                history = res.json().get("history", [])
                last_3  = history[-3:]

                for i, gw in enumerate(reversed(last_3)):
                    all_recent_points.append({
                        "player_id": player_id,
                        "gw_rank": i + 1,  # 1 = most recent
                        "points": gw["total_points"]
                    })

            except Exception:
                continue

    # ─────────────────────────────────────────────
    # 6. Merge history
    # ─────────────────────────────────────────────
    if all_recent_points:
        history_df = pd.DataFrame(all_recent_points)

        pivot = history_df.pivot(
            index="player_id",
            columns="gw_rank",
            values="points"
        ).rename(columns={
            1: "gw_1_points",
            2: "gw_2_points",
            3: "gw_3_points"
        })

        players_df = players_df.merge(
            pivot, left_on="id", right_index=True, how="left"
        )

    for col in ["gw_1_points", "gw_2_points", "gw_3_points"]:
        players_df[col] = players_df.get(col, 0).fillna(0)

    # ─────────────────────────────────────────────
    # 7. Form calculation
    # ─────────────────────────────────────────────
    players_df["base_form_points"] = (
        0.45 * players_df["gw_1_points"] +
        0.35 * players_df["gw_2_points"] +
        0.20 * players_df["gw_3_points"]
    )

    # ─────────────────────────────────────────────
    # 8. Apply fixture multiplier
    # ─────────────────────────────────────────────
    players_df["fixture_multiplier"] = (
        players_df["team_id"].map(team_multipliers)
    )

    players_df["projected_points"] = (
        players_df["base_form_points"] *
        players_df["fixture_multiplier"]
    ).apply(np.floor)

    # ─────────────────────────────────────────────
    # 9. Final cleanup
    # ─────────────────────────────────────────────
    players_df["now_cost"] = players_df["now_cost"] / 10
    players_df["team"]     = players_df["team_name"]

    return players_df.sort_values(
        by="projected_points", ascending=False
    ).reset_index(drop=True)

def optimize_fpl_team(players_df, players_to_keep=None, players_to_exclude=None):
    """Runs Linear Programming to select the best 15-man squad."""
    players_to_keep = players_to_keep or []
    players_to_exclude = players_to_exclude or []

    # Setup Variables
    players_df['selected'] = players_df['name'].apply(lambda name: LpVariable(name.replace(" ", "_"), cat='Binary'))
    model = LpProblem("FPL_Squad_Optimizer", LpMaximize)

    # Objective: Maximize Projected Points
    model += lpSum(players_df['selected'] * players_df['projected_points'])

    # Constraints
    model += lpSum(players_df['selected'] * players_df['now_cost']) <= 100.0, "Budget"
    model += lpSum(players_df['selected']) == 15, "Squad_Size"

    # Positional Constraints
    for pos, count in {1: 2, 2: 5, 3: 5, 4: 3}.items():
        model += lpSum(players_df.loc[players_df['element_type'] == pos, 'selected']) == count, f"Pos_{pos}_Count"

    # Max 3 players per team
    for team_id in players_df['team'].unique():
        model += lpSum(players_df.loc[players_df['team'] == team_id, 'selected']) <= 3

    # Forced Keep/Exclude
    for name in players_to_keep:
        if name in players_df['name'].values:
            model += players_df.loc[players_df['name'] == name, 'selected'].values[0] == 1
    for name in players_to_exclude:
        if name in players_df['name'].values:
            model += players_df.loc[players_df['name'] == name, 'selected'].values[0] == 0

    model.solve()
    print("Optimization Status:", LpStatus[model.status])

    # Extract Results
    players_df['is_picked'] = players_df['selected'].apply(lambda var: var.varValue)
    selected_df = players_df[players_df['is_picked'] == 1].copy()
    
    # Store the plot object in a separate variable
    fig = _plot_optimization_results(players_df, selected_df)
    
    # Returning the plot figure along with the results
    return selected_df['name'].tolist(), model, fig 

def _plot_optimization_results(all_players, selected_players):
    """
    Optimized Plotly visualization for FPL squad optimization.
    Selected players are highlighted by position; unselected pool is subtle.
    """

    # --- Constants ---
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    CATEGORY_ORDER = ['GKP', 'DEF', 'MID', 'FWD', 'Unselected Pool']

    COLOR_MAP = {
        'GKP': 'deepskyblue',
        'DEF': 'green',
        'MID': 'gold',
        'FWD': 'red',
        'Unselected Pool': 'lightgrey'
    }

    SIZE_MAP = {
        'GKP': 6,
        'DEF': 6,
        'MID': 6,
        'FWD': 6,
        'Unselected Pool': 3
    }

    # --- Work on a copy (avoid side effects) ---
    df = all_players.copy()

    # --- Vectorized feature engineering ---
    df['position_name'] = df['element_type'].map(POSITION_MAP)

    df['is_selected_status'] = np.where(
        df['is_picked'] == 1,
        df['position_name'],
        'Unselected Pool'
    )

    df['marker_size'] = df['is_selected_status'].map(SIZE_MAP)

    # --- Plot ---
    fig = px.scatter(
        df,
        x='now_cost',
        y='projected_points',
        color='is_selected_status',
        size='marker_size',
        size_max=8,
        color_discrete_map=COLOR_MAP,
        category_orders={'is_selected_status': CATEGORY_ORDER},
        hover_name='name',
        hover_data={
            'now_cost': ':.1f',
            'projected_points': ':.1f',
            'team': True,
            'total_points': True,
            'form': ':.1f',
            'position_name': False,
            'marker_size': False
        },
        labels={
            'now_cost': 'Cost (£m)',
            'projected_points': 'Projected Points (Next GW)',
            'is_selected_status': 'Squad Status'
        },
        title='Optimized Squad Selection: Cost vs Projected Points'
    )

    # --- Opacity tuning ---
    fig.update_traces(
        opacity=0.35,
        selector=dict(name='Unselected Pool')
    )

    fig.update_traces(
        opacity=1.0,
        selector=lambda t: t.name != 'Unselected Pool'
    )

    # --- Layout polish ---
    fig.update_layout(
        hovermode='closest',
        legend_title_text='Selection',
        xaxis=dict(range=[3.5, 20]),
        template='plotly_white'
    )

    return fig # KEY CHANGE: Return the figure for Streamlit

# -----------------------------------------------------------

def optimize_starting_11(squad_df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes Starting XI from the selected 15-man squad."""
    if len(squad_df) != 15:
        print(f"Warning: Squad has {len(squad_df)} players. Optimization requires 15.")
        return pd.DataFrame()

    squad_df['starter'] = squad_df['name'].apply(lambda name: LpVariable(f"starter_{name.replace(' ','_')}", cat='Binary'))
    model = LpProblem("FPL_Starting_XI", LpMaximize)

    # Objective
    model += lpSum(squad_df['starter'] * squad_df['projected_points'])

    # Constraints
    model += lpSum(squad_df['starter']) == 11
    model += lpSum(squad_df[squad_df['element_type'] == 1]['starter']) == 1  # GKP
    
    # Flexible formation constraints
    # Def: 3-5, Mid: 2-5, Fwd: 1-3
    for pos, min_c, max_c in [(2, 3, 5), (3, 2, 5), (4, 1, 3)]:
        starters = squad_df[squad_df['element_type'] == pos]['starter']
        model += lpSum(starters) >= min_c
        model += lpSum(starters) <= max_c

    model.solve()
    
    # Process Results
    squad_df['is_starter'] = squad_df['starter'].apply(lambda x: x.varValue)
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    squad_df['position_name'] = squad_df['element_type'].map(POSITION_MAP)
    
    starters_df = squad_df[squad_df['is_starter'] == 1].copy()
    
    # Captaincy
    starters_df = starters_df.sort_values(by='projected_points', ascending=False)
    starters_df['role'] = 'Starter'
    if len(starters_df) >= 2:
        starters_df.iloc[0, starters_df.columns.get_loc('role')] = 'Captain (C)'
        starters_df.iloc[1, starters_df.columns.get_loc('role')] = 'Vice-Captain (VC)'

    # Sort by position for display (GKP -> FWD)
    starters_df = starters_df.sort_values(by=['element_type', 'projected_points'], ascending=[True, False])
    starters_df['projected_points'] = np.floor(starters_df['projected_points'])
    return starters_df[[
    'name', 
    'position_name',
    'role',
    'team',
    'now_cost',
    'selected_by_percent',
    'projected_points',
    'fixture_multiplier',
    'form',
    'points_per_game',
    'total_points',
    'gw_1_points',
    'gw_2_points',
    'gw_3_points',
    'chance_of_playing_this_round',
]]

# --- AI / LLM Features ---
def _check_api_keys():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables.")
        return False
    # Google Search API is also required for injury news
    if not os.environ.get("GOOGLE_CSE_ID") or not os.environ.get("GOOGLE_API_KEY"):
        print("⚠️ Warning: GOOGLE_CSE_ID or GOOGLE_API_KEY not found for Google Search.")
        return False
    return True

def fpl_langchain_advisor(question: str, my_team_df) -> str:
    """Analyze team using Gemini."""
    # NOTE: You must set GOOGLE_API_KEY in your environment for this to work.
    if not _check_api_keys(): return "API Keys missing. Please set GOOGLE_API_KEY."

    subset = my_team_df[['name', 'total_points', 'now_cost', 'minutes', 'form']].to_csv(index=False)
    
    template = PromptTemplate(
        input_variables=["question", "team_data"],
        template="""
        You are an FPL expert. Given this team data:
        {team_data}
        
        Answer this question: {question}
        Provide concise, data-backed advice.
        """
    )
    
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=template)
    return chain.run({"question": question, "team_data": subset})

def get_fpl_injury_news():
    """Fetch injury news summary."""
    # NOTE: You must set GOOGLE_API_KEY and GOOGLE_CSE_ID for the Google Search API.
    if not _check_api_keys(): return "API Keys missing. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID."

    search = GoogleSearchAPIWrapper()
    try:
        results = search.run("Fantasy Premier League injury news site:fantasyfootballscout.co.uk")
    except Exception as e:
        return f"Error running Google Search: {e}. Check your GOOGLE_CSE_ID and GOOGLE_API_KEY."
    
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        input_variables=["news"],
        template="Summarize these FPL injury news items for the upcoming Gameweek:\n{news}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(news=results)