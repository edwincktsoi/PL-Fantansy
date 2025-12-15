import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus

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

import requests
import pandas as pd
from tqdm import tqdm

def fetch_and_forecast_players():
    """
    Fetches FPL players, calculates form from last 3 GWs, 
    and adjusts projected points based on upcoming fixture difficulty.
    """
    base_url = "https://fantasy.premierleague.com/api/"
    
    # 1. Fetch Bootstrap Data
    print("Fetching bootstrap data...")
    bootstrap = requests.get(base_url + "bootstrap-static/").json()
    players_df = pd.DataFrame(bootstrap['elements'])
    events_df = pd.DataFrame(bootstrap['events'])
    
    # Identify the NEXT Gameweek
    next_event = events_df[events_df['is_next'] == True]
    if not next_event.empty:
        next_gw_id = next_event.iloc[0]['id']
        print(f"Next Gameweek is: GW{next_gw_id}")
    else:
        print("Season finished or no next GW found. Defaulting to GW 38.")
        next_gw_id = 38

    # 2. Define Difficulty Discount Factors (The "Multiplier")
    # 1 (Easy) -> Boost points | 5 (Hard) -> Reduce points
    difficulty_multiplier = {
        1: 1.20,  # Very Easy: +20%
        2: 1.10,  # Easy: +10%
        3: 1.00,  # Average: No change
        4: 0.85,  # Hard: -15%
        5: 0.70   # Very Hard: -30%
    }

    # 3. Fetch Fixtures for the Next Gameweek
    print(f"Fetching fixtures for GW{next_gw_id}...")
    fixtures = requests.get(base_url + "fixtures/").json()
    
    # Initialize a map for team multipliers (Default 0.0 for Blank GWs)
    # Bench value for Blank Gameweeks (player still has some value)
    BENCH_MULTIPLIER = 0.30
    # 20 teams, indexed by ID (1-20)
    team_multipliers = {i: BENCH_MULTIPLIER for i in range(1, 21)}
    
    # Process fixtures to build the multiplier map
    # This automatically handles Double Gameweeks (adds both games) and Blanks (stays 0)
    for f in fixtures:
        if f['event'] == next_gw_id:
            h_team = f['team_h']
            a_team = f['team_a']
            h_diff = f['team_h_difficulty']
            a_diff = f['team_a_difficulty']
            
            # Add multiplier for Home Team
            team_multipliers[h_team] += difficulty_multiplier.get(h_diff, 1.0)
            
            # Add multiplier for Away Team
            team_multipliers[a_team] += difficulty_multiplier.get(a_diff, 1.0)

    # 4. Process Player Data
    # Keep only key fields
    players_df = players_df[[
        'id', 'first_name', 'second_name', 'team', 'element_type',
        'now_cost', 'minutes', 'selected_by_percent',
        'total_points', 'expected_goals', 'expected_assists',
        'form', 'points_per_game', 'in_dreamteam', 'chance_of_playing_this_round'
    ]]

    players_df['name'] = players_df['first_name'] + ' ' + players_df['second_name']
    
    # Filter for active players
    # Use fillna(100) because "None" often means fully fit in FPL API
    players_df['chance_of_playing_this_round'] = players_df['chance_of_playing_this_round'].fillna(100)
    players_df = players_df[players_df['chance_of_playing_this_round'] > 70]

    # 5. Fetch Recent Form (Last 3 GWs)
    all_recent_points = []
    
    # Optimization: Only fetch history for relevant players to save time
    # (e.g. players with > 10 total points or cost > 4.0)
    relevant_players = players_df[players_df['total_points'] >= 10]['id'].tolist()
    
    print(f"Fetching history for {len(relevant_players)} players...")
    
    # Using a session for faster connection reuse
    with requests.Session() as session:
        for player_id in tqdm(relevant_players, desc="Fetching GW history"):
            try:
                res = session.get(f"{base_url}element-summary/{player_id}/")
                if res.status_code != 200: continue
                
                history = res.json().get('history', [])
                last_3 = history[-3:] if len(history) >= 3 else history
                
                # If player has no history, skip
                if not last_3: continue
                
                for i, gw in enumerate(reversed(last_3)):
                    all_recent_points.append({
                        'player_id': player_id,
                        'gw_rank': i + 1, # 1 = most recent
                        'total_points': gw['total_points']
                    })
            except Exception:
                continue

    history_df = pd.DataFrame(all_recent_points)
    
    if not history_df.empty:
        pivot = history_df.pivot(index='player_id', columns='gw_rank', values='total_points')
        pivot.columns = [f'gw_{col}_points' for col in pivot.columns]
        pivot = pivot.reset_index()
        players_df = players_df.merge(pivot, left_on='id', right_on='player_id', how='left')
    else:
        # Fallback if history fetch fails or season just started
        for col in ['gw_1_points', 'gw_2_points', 'gw_3_points']:
            players_df[col] = 0

    # Fill NaN history with 0
    for col in ['gw_1_points', 'gw_2_points', 'gw_3_points']:
        if col in players_df.columns:
            players_df[col] = players_df[col].fillna(0)
        else:
            players_df[col] = 0

    # 6. Calculate Base Form Points (Weighted)
    players_df['base_form_points'] = (
        0.45 * players_df['gw_1_points'] +  # 50% weight to most recent
        0.35 * players_df['gw_2_points'] +  # 30% weight to 2nd recent
        0.2 * players_df['gw_3_points']    # 20% weight to 3rd recent
    )

    # 7. Apply Fixture Difficulty Discount/Boost
    # Map team ID to the multiplier we calculated earlier
    players_df['fixture_multiplier'] = players_df['team'].map(team_multipliers)
    
    # Final Projection Formula
    players_df['projected_points'] = players_df['base_form_points'] * players_df['fixture_multiplier']

    # Handle Cost
    players_df['now_cost'] = players_df['now_cost'] / 10

    # Clean up columns
    final_df = players_df.sort_values(by='projected_points', ascending=False)
    
    return final_df


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


from langchain_community.utilities import GoogleSearchAPIWrapper
def fpl_langchain_advisor(question: str, my_team_df) -> str:
    """Use LLM to analyze your current FPL team and answer the question."""

    # Set environment variables for LangChain and Google API
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = ''
    os.environ['GOOGLE_API_KEY'] = ''

    # Select relevant columns from your FPL team DataFrame
    subset = my_team_df[['name', 'position', 'total_points', 'now_cost', 'minutes', 'form', 'selected_by_percent']]
    team_data_str = subset.to_csv(index=False)

    # Create prompt
    template = PromptTemplate(
        input_variables=["question", "team_data"],
        template="""
You are a top-tier Fantasy Premier League (FPL) expert.

Below is data about my FPL team:

{team_data}

Using this data, answer the following question:

{question}

Provide specific advice using the stats. Consider form, minutes played, cost, and ownership.
"""
    )

    # Use Gemini Flash model
    llm = GoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=template)

    # Run the chain
    response = chain.run({
        "question": question,
        "team_data": team_data_str
    })

    return response

def get_fpl_injury_news():
    # Set environment variables for LangChain and Google API
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = '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'
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyC3mD-iVxmgexEwdNjR0MqFfdhyBpLnApY'
    # 1. Search the latest news
    search = GoogleSearchAPIWrapper()
    results = search.run("Fantasy Premier League injury news site:fantasyfootballscout.co.uk OR premierleague.com")

    # 2. Prompt LLM to summarize
    llm = GoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    prompt = PromptTemplate(
        input_variables=["news"],
        template="""
You're a Fantasy Premier League advisor. Summarize the latest injury and team news from the content below for this Gameweek:

{news}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(news=results)
    return summary




from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
import pandas as pd

def optimize_starting_11(squad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the optimal starting 11 from the 15-player squad based on projected points,
    and then selects the Captain (C) and Vice-Captain (VC) from that 11.
    
    Args:
        squad_df (pd.DataFrame): DataFrame containing the 15 selected players.

    Returns:
        pd.DataFrame: DataFrame of the 11 selected players, sorted by position, 
                      including C/VC status and rich player data.
    """
    if len(squad_df) != 15:
        print("Warning: Input DataFrame does not contain exactly 15 players. Please check the squad optimization.")
        return pd.DataFrame()

    # Create binary decision variables for the starting 11 selection from the squad
    squad_df['starter'] = squad_df['name'].apply(lambda name: LpVariable(f"starter_{name}", cat='Binary'))

    # Initialize LP problem for starting 11
    model_xi = LpProblem("FPL_Starting_11_Optimizer", LpMaximize)

    # Objective: maximize projected points of the starting 11
    model_xi += lpSum(squad_df['starter'] * squad_df['projected_points']), "Maximize_Starting_XI_Points"

    # --- Starting 11 Constraints (Positional Rules) ---
    model_xi += lpSum(squad_df['starter']) == 11, "Total_XI_Size"
    
    gkp_starters = squad_df[squad_df['element_type'] == 1]['starter']
    def_starters = squad_df[squad_df['element_type'] == 2]['starter']
    mid_starters = squad_df[squad_df['element_type'] == 3]['starter']
    fwd_starters = squad_df[squad_df['element_type'] == 4]['starter']
    
    model_xi += lpSum(gkp_starters) == 1, "GKP_Count"
    model_xi += lpSum(def_starters) >= 3, "Min_DEF_Count"
    model_xi += lpSum(def_starters) <= 5, "Max_DEF_Count"
    model_xi += lpSum(mid_starters) >= 2, "Min_MID_Count"
    model_xi += lpSum(mid_starters) <= 5, "Max_MID_Count"
    model_xi += lpSum(fwd_starters) >= 1, "Min_FWD_Count"
    model_xi += lpSum(fwd_starters) <= 3, "Max_FWD_Count"

    # Solve the problem
    model_xi.solve()

    print("\nStarting 11 Status:", LpStatus[model_xi.status])

    # 1. EXTRACT THE SELECTED PLAYERS AS A DATAFRAME
    starter_df = squad_df.loc[squad_df['starter'].apply(lambda var: var.varValue == 1)].copy()
    
    # --- CAPTAINCY SELECTION LOGIC ---
    starter_df['role'] = 'Starter'
    
    if len(starter_df) >= 2:
        # Sort the starting XI by projected points in descending order
        sorted_starters = starter_df.sort_values(
            by='projected_points', 
            ascending=False
        ).reset_index(drop=True)

        # Captain: Player with the highest projected points
        captain_name = sorted_starters.iloc[0]['name']
        
        # Vice-Captain: Player with the second highest projected points
        vice_captain_name = sorted_starters.iloc[1]['name']
        
        # Update the 'role' column
        starter_df.loc[starter_df['name'] == captain_name, 'role'] = 'Captain (C)'
        starter_df.loc[starter_df['name'] == vice_captain_name, 'role'] = 'Vice-Captain (VC)'
        
        print("\n--- Captaincy Selection ---")
        print(f"👑 Captain (C): {captain_name} ({sorted_starters.iloc[0]['projected_points']:.1f} pts)")
        print(f"© Vice-Captain (VC): {vice_captain_name} ({sorted_starters.iloc[1]['projected_points']:.1f} pts)")
        print("-" * 30)
    
    # 2. Sort by 'element_type' (1 to 4) to get GKP -> FWD order
    # Secondary sort by projected_points to order within position
    starter_df = starter_df.sort_values(
        by=['element_type', 'projected_points'], 
        ascending=[True, False]
    )

    # 3. SELECT THE DESIRED COLUMNS on the DataFrame
    # Note: 'role' has been added here
    try:
        starting_11 = starter_df[[
            'name', 'position', 'role', 'projected_points', 'now_cost',
            'form', 'selected_by_percent', 'minutes', 'total_points', 'gw_1_points','gw_2_points', 'gw_3_points',
            'expected_goals', 'expected_assists', 'points_per_game', 'in_dreamteam',
             'chance_of_playing_this_round',
        ]]
    except KeyError as e:
        print(f"\nError: Column {e} not found in the DataFrame. Please check the column list.")
        # Return a simplified version if the rich data columns are missing
        starting_11 = starter_df[['name', 'position', 'role', 'projected_points', 'now_cost']].copy()
    
    return starting_11


