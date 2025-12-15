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

def load_fpl_data():
    """Fetch data from FPL API and return player and position dataframes."""
    url = f"{FPL_BASE_URL}bootstrap-static/"
    data = requests.get(url).json()
    
    players_df = pd.DataFrame(data['elements'])
    positions_df = pd.DataFrame(data['element_types'])
    
    # Map positions
    players_df['position'] = players_df['element_type'].map(positions_df.set_index('id')['singular_name'])
    
    # Calculate Age
    players_df['birth_date'] = pd.to_datetime(players_df['birth_date'], errors='coerce')
    players_df['age'] = (pd.Timestamp.now() - players_df['birth_date']).dt.days // 365
    
    return players_df, positions_df

def plot_points_vs_cost(players_df, positions_df):
    """Plot total FPL points vs cost for each position."""
    df = players_df.copy()
    df['cost_m'] = df['now_cost'] / 10

    for pos in df['position'].unique():
        subset = df[df['position'] == pos]
        fig = px.scatter(
            subset,
            x='cost_m',
            y='total_points',
            hover_name='web_name',
            trendline='ols',
            labels={'cost_m': 'Cost (£m)', 'total_points': 'Total Points'},
            title=f'Total Points vs Cost — {pos}s'
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.show()

def plot_cumulative_points(age_limit=26, position='Midfielder', min_cum_points=50):
    """Plot cumulative points history for specific player demographics."""
    players_df, _ = load_fpl_data()
    filtered = players_df[(players_df['position'] == position) & (players_df['age'] < age_limit)]

    if filtered.empty:
        print(f"No players found for position {position} under age {age_limit}.")
        return

    all_history = []
    
    # Use Session for faster repeated requests
    with requests.Session() as session:
        for _, player in tqdm(filtered.iterrows(), total=filtered.shape[0], desc="Fetching History"):
            try:
                url = f"{FPL_BASE_URL}element-summary/{player['id']}/"
                resp = session.get(url)
                if resp.status_code == 200:
                    history = resp.json().get('history', [])
                    if history:
                        df = pd.DataFrame(history)
                        df['player_name'] = player['web_name']
                        all_history.append(df)
            except Exception:
                continue

    if not all_history:
        print("No historical data available.")
        return

    df_history = pd.concat(all_history)
    fig = go.Figure()

    # Create traces only for players meeting the point threshold
    for name in df_history['player_name'].unique():
        player_data = df_history[df_history['player_name'] == name].sort_values('round')
        player_data['cumulative_points'] = player_data['total_points'].cumsum()
        
        if player_data['cumulative_points'].max() >= min_cum_points:
            fig.add_trace(go.Scatter(
                x=player_data['round'],
                y=player_data['cumulative_points'],
                mode='lines+markers',
                name=name
            ))

    fig.update_layout(
        title=f"Cumulative Points — {position}s under {age_limit}",
        xaxis_title="Gameweek",
        yaxis_title="Cumulative Total Points",
        hovermode="x unified"
    )
    fig.show()

def fetch_and_forecast_players():
    """
    Fetches FPL players, calculates form from last 3 GWs, 
    and adjusts projected points based on upcoming fixture difficulty.
    """
    print("Fetching bootstrap data...")
    bootstrap = requests.get(f"{FPL_BASE_URL}bootstrap-static/").json()
    players_df = pd.DataFrame(bootstrap['elements'])
    events_df = pd.DataFrame(bootstrap['events'])
    
    # Identify Next Gameweek
    next_event = events_df[events_df['is_next'] == True]
    next_gw_id = next_event.iloc[0]['id'] if not next_event.empty else 38
    print(f"Forecasting for GW{next_gw_id}...")

    # Calculate Team Difficulty Multipliers
    print("Fetching fixtures...")
    fixtures = requests.get(f"{FPL_BASE_URL}fixtures/").json()
    
    # Initialize multipliers (Default 0.30 for blank GWs)
    team_multipliers = {i: 0.30 for i in range(1, 21)}
    
    for f in fixtures:
        if f['event'] == next_gw_id:
            # Add difficulty multiplier (Handles Double GWs automatically)
            team_multipliers[f['team_h']] += DIFFICULTY_MULTIPLIER.get(f['team_h_difficulty'], 1.0)
            team_multipliers[f['team_a']] += DIFFICULTY_MULTIPLIER.get(f['team_a_difficulty'], 1.0)

    # Filter and Prep Player Data
    players_df['name'] = players_df['first_name'] + ' ' + players_df['second_name']
    
    # Clean chance_of_playing (None usually means 100%)
    players_df['chance_of_playing_this_round'] = players_df['chance_of_playing_this_round'].fillna(100)
    players_df = players_df[players_df['chance_of_playing_this_round'] > 70]

    # Fetch Recent History (Last 3 GWs)
    relevant_players = players_df[players_df['total_points'] >= 10]['id'].tolist()
    all_recent_points = []
    
    print(f"Fetching history for {len(relevant_players)} active players...")
    with requests.Session() as session:
        for player_id in tqdm(relevant_players, desc="Processing Players"):
            try:
                res = session.get(f"{FPL_BASE_URL}element-summary/{player_id}/")
                if res.status_code == 200:
                    history = res.json().get('history', [])
                    last_3 = history[-3:] if len(history) >= 3 else history
                    
                    if last_3:
                        for i, gw in enumerate(reversed(last_3)):
                            all_recent_points.append({
                                'player_id': player_id,
                                'gw_rank': i + 1, # 1 = most recent
                                'total_points': gw['total_points']
                            })
            except Exception:
                continue

    # Merge History
    if all_recent_points:
        history_df = pd.DataFrame(all_recent_points)
        pivot = history_df.pivot(index='player_id', columns='gw_rank', values='total_points')
        pivot.columns = [f'gw_{col}_points' for col in pivot.columns]
        players_df = players_df.merge(pivot, left_on='id', right_index=True, how='left')
    
    # Fill missing history with 0
    for col in ['gw_1_points', 'gw_2_points', 'gw_3_points']:
        if col not in players_df.columns:
            players_df[col] = 0
        players_df[col] = players_df[col].fillna(0)

    # Calculate Projection
    players_df['base_form_points'] = (
        0.45 * players_df['gw_1_points'] + 
        0.35 * players_df['gw_2_points'] + 
        0.20 * players_df['gw_3_points']
    )
    
    players_df['fixture_multiplier'] = players_df['team'].map(team_multipliers)
    players_df['projected_points'] = players_df['base_form_points'] * players_df['fixture_multiplier']
    players_df['now_cost'] = players_df['now_cost'] / 10

    return players_df.sort_values(by='projected_points', ascending=False)

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
    
    _plot_optimization_results(players_df, selected_df)
    
    return selected_df['name'].tolist(), model

# In pl.py

#def _plot_optimization_results(all_players, selected_players):
#    """Helper function to visualize the optimization results."""
#    
#    # ISSUE: 'position_name' was only created on 'all_players' (which is players_df)
#    # SOLUTION: Apply the position map to the selected_players DataFrame directly.
#    
#    # 1. Map Position Names to BOTH DataFrames (or at least the one being filtered)
#    all_players['position_name'] = all_players['element_type'].map(POSITION_MAP)
#    
#    # Apply the same mapping to the selected_players DataFrame to ensure the column exists
#    selected_players['position_name'] = selected_players['element_type'].map(POSITION_MAP) 
#    
#    colors = {'GKP': 'blue', 'DEF': 'green', 'MID': 'orange', 'FWD': 'purple'}
#    
#    plt.figure(figsize=(12, 7))
#    
#    # Plot unselected
#    unselected = all_players[all_players['is_picked'] == 0]
#    plt.scatter(unselected['now_cost'], unselected['projected_points'], 
#                color='lightgray', alpha=0.5, label='Pool')
#
#    # Plot selected
#    for pos, color in colors.items():
#        # This line now works because 'position_name' exists on 'selected_players'
#        subset = selected_players[selected_players['position_name'] == pos] 
#        plt.scatter(subset['now_cost'], subset['projected_points'], 
#                    color=color, label=pos, s=100, edgecolor='black', zorder=3)
#        for _, row in subset.iterrows():
#            plt.text(row['now_cost'], row['projected_points'] + 0.2, 
#                     row['name'], fontsize=8, ha='center')
#
#    plt.xlabel('Cost (£m)')
#    plt.ylabel('Projected Points')
#    plt.title('Optimized Squad Selection')
#    plt.legend()
#    plt.grid(True, alpha=0.3)
#    plt.tight_layout()
#    plt.show()

# In pl.py, replace the existing _plot_optimization_results function
import plotly.express as px
import numpy as np

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

    fig.show()

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

    return starters_df[[
    'name', 
    'position_name',           # Position (e.g., Goalkeeper, Defender)
    'role',               # Role (Captain, Vice-Captain, Starter)
    'team',               # Club
    'now_cost',
    'selected_by_percent',            # Price (£m)
    'projected_points',   # The single most important column for prediction
    'fixture_multiplier', # The factor used to adjust the points (context for projection)
    'form',               # Official FPL form score (last 5 GWs)
    'points_per_game',    # Consistency metric
    'total_points',       # Season total points
    'gw_1_points',        # Points from most recent GW (Crucial for form)
    'gw_2_points',        # Points from 2nd most recent GW
    'gw_3_points',        # Points from 3rd most recent GW
    'chance_of_playing_this_round', # Injury/Suspension risk (100 is best)

]]

# --- AI / LLM Features ---
# Ensure you set these keys in your environment variables before running!
def _check_api_keys():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables.")
        return False
    return True

def fpl_langchain_advisor(question: str, my_team_df) -> str:
    """Analyze team using Gemini."""
    if not _check_api_keys(): return "API Keys missing."

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
    if not _check_api_keys(): return "API Keys missing."

    search = GoogleSearchAPIWrapper()
    results = search.run("Fantasy Premier League injury news site:fantasyfootballscout.co.uk")
    
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        input_variables=["news"],
        template="Summarize these FPL injury news items for the upcoming Gameweek:\n{news}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(news=results)