import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

def load_fpl_data():
    """Fetch data from FPL API and return player and position dataframes."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url).json()
    players_df = pd.DataFrame(data['elements'])
    positions_df = pd.DataFrame(data['element_types'])
    players_df['position'] = players_df['element_type'].map(positions_df.set_index('id')['singular_name'])
    players_df['birth_date'] = pd.to_datetime(players_df['birth_date'], errors='coerce')
    today = pd.to_datetime('today')
    players_df['age'] = (today - players_df['birth_date']).dt.days // 365
    return players_df, positions_df

def plot_points_vs_cost(players_df, positions_df):
    """Plot total FPL points vs cost for each position using scatter plots with trendlines."""
    players_df = players_df.copy()
    players_df['cost_m'] = players_df['now_cost'] / 10

    for pos in players_df['position'].unique():
        subset = players_df[players_df['position'] == pos]
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
    """Plot cumulative points over gameweeks for players under given age and above a point threshold."""
    players_df, _ = load_fpl_data()
    filtered = players_df[(players_df['position'] == position) & (players_df['age'] < age_limit)]

    all_history = []
    for _, player in filtered.iterrows():
        url = f"https://fantasy.premierleague.com/api/element-summary/{player['id']}/"
        history = requests.get(url).json().get('history', [])
        if history:
            df = pd.DataFrame(history)
            df['player_name'] = player['web_name']
            all_history.append(df)

    if not all_history:
        print("No player data found.")
        return

    df_history = pd.concat(all_history)
    fig = go.Figure()

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


import requests
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
from tqdm import tqdm

def fetch_and_forecast_players():
    base_url = "https://fantasy.premierleague.com/api/"
    data = requests.get(base_url + "bootstrap-static/").json()

    players_df = pd.DataFrame(data['elements'])

    # Keep only key fields you want
    players_df = players_df[[
        'id', 'first_name', 'second_name', 'team', 'element_type',
        'now_cost', 'minutes', 'selected_by_percent',
        'total_points', 'expected_goals', 'expected_assists',
        'form', 'points_per_game', 'in_dreamteam', 'chance_of_playing_this_round'
    ]]

    # Add full player name for convenience
    players_df['name'] = players_df['first_name'] + ' ' + players_df['second_name']

    # Filter players likely to play (>70%)
    players_df = players_df[players_df['chance_of_playing_this_round'] > 70]

    # Fetch last 3 GWs points for each player
    all_recent_points = []
    for player_id in tqdm(players_df['id'], desc="Fetching GW history"):
        res = requests.get(f"{base_url}element-summary/{player_id}/")
        if res.status_code != 200:
            continue
        history = res.json().get('history', [])
        last_3_gws = history[-3:] if len(history) >= 3 else history
        for i, gw in enumerate(reversed(last_3_gws)):
            all_recent_points.append({
                'player_id': player_id,
                'gw_rank': i + 1,  # 1 = most recent
                'total_points': gw['total_points']
            })

    history_df = pd.DataFrame(all_recent_points)
    pivot = history_df.pivot(index='player_id', columns='gw_rank', values='total_points')
    pivot.columns = [f'gw_{col}_points' for col in pivot.columns]
    pivot = pivot.reset_index()

    players_df = players_df.merge(pivot, left_on='id', right_on='player_id', how='left')

    for col in ['gw_1_points', 'gw_2_points', 'gw_3_points']:
        players_df[col] = players_df[col].fillna(0)

    # Weighted projected points
    players_df['projected_points'] = (
        0.5 * players_df['gw_1_points'] +
        0.3 * players_df['gw_2_points'] +
        0.2 * players_df['gw_3_points']
    )

    # Convert cost to millions
    players_df['now_cost'] = players_df['now_cost'] / 10

    return players_df


def optimize_fpl_team(players_df, players_to_keep=None, players_to_exclude=None):
    if players_to_keep is None:
        players_to_keep = []
    if players_to_exclude is None:
        players_to_exclude = []

    # Create binary decision variables for each player
    players_df['selected'] = players_df['name'].apply(lambda name: LpVariable(name, cat='Binary'))

    # Initialize LP problem
    model = LpProblem("FPL_Team_Optimizer", LpMaximize)

    # Objective: maximize projected points
    model += lpSum(players_df['selected'] * players_df['projected_points'])

    # Constraints
    model += lpSum(players_df['selected'] * players_df['now_cost']) <= 100  # Budget
    model += lpSum(players_df['selected']) == 15  # Squad size

    # Position constraints
    position_counts = {1: 2, 2: 5, 3: 5, 4: 3}  # GKP, DEF, MID, FWD
    for pos, count in position_counts.items():
        model += lpSum(players_df.loc[players_df['element_type'] == pos, 'selected']) == count

    # Max 3 players per team
    for team_id in players_df['team'].unique():
        model += lpSum(players_df.loc[players_df['team'] == team_id, 'selected']) <= 3

    # Keep specified players
    for player_name in players_to_keep:
        if player_name in players_df['name'].values:
            model += players_df.loc[players_df['name'] == player_name, 'selected'].values[0] == 1

    # Exclude specified players
    for player_name in players_to_exclude:
        if player_name in players_df['name'].values:
            model += players_df.loc[players_df['name'] == player_name, 'selected'].values[0] == 0

    # Position-wise budget constraints (optional, can be adjusted)
    budget_constraints = {
        1: (8.0, 8.5),
        2: (25.0, 26.5),
        3: (36.0, 38.0),
        4: (27.0, 29.0),
    }
    for pos, (min_budget, max_budget) in budget_constraints.items():
        pos_players = players_df[players_df['element_type'] == pos]
        total_cost = lpSum(pos_players['selected'] * pos_players['now_cost'])
        model += total_cost >= min_budget
        model += total_cost <= max_budget

    # Solve
    model.solve()

    print("Status:", LpStatus[model.status])
    # Map positions
    position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players_df['position'] = players_df['element_type'].map(position_map)
    selected = players_df[players_df['selected'].apply(lambda var: var.varValue == 1)].copy()
    unselected = players_df[players_df['selected'].apply(lambda var: var.varValue == 0)].copy()
    
    # Define colors for selected player positions
    colors = {'GKP': 'blue', 'DEF': 'green', 'MID': 'orange', 'FWD': 'purple'}
    
    # Filter valid players (exclude 0 projected points and extreme low-cost)
    valid_players = players_df[ (players_df['now_cost'] > 2)]
    
    # Calculate average cost and points
    avg_cost = valid_players['now_cost'].mean()
    avg_points = valid_players['projected_points'].mean()
    
    plt.figure(figsize=(12, 7))
    
    # Plot unselected players in light gray
    plt.scatter(unselected['now_cost'], unselected['projected_points'],
                color='lightgray', alpha=0.5, label='Unselected Players')
    
    # Plot selected players by position with annotations
    for pos, color in colors.items():
        subset = selected[selected['position'] == pos]
        plt.scatter(subset['now_cost'], subset['projected_points'],
                    color=color, label=pos, s=100, edgecolor='black', zorder=3)
        for _, row in subset.iterrows():
            plt.text(row['now_cost'] + 0.1, row['projected_points'], row['name'],
                     fontsize=8, zorder=4)
    
    # Draw average lines
    #plt.axvline(avg_cost, color='darkblue', linestyle='--', linewidth=2,
    #            label=f'Avg Cost ≈ {avg_cost:.2f}')
    #plt.axhline(avg_points, color='darkred', linestyle='--', linewidth=2,
    #            label=f'Avg Projected Points ≈ {avg_points:.2f}')
    
    plt.xlabel('Player Cost (now_cost)')
    plt.ylabel('Projected Points')
    plt.title('FPL Team Selection: Projected Points vs Cost')
    plt.legend(title='Selected Positions & Averages')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    selected_players = players_df.loc[players_df['selected'].apply(lambda var: var.varValue == 1), 'name'].tolist()

    return selected_players, model
