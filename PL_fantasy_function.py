#%%

import requests
import pandas as pd

# Load FPL data
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
data = requests.get(url).json()

players_df = pd.DataFrame(data['elements'])
teams = pd.DataFrame(data['teams'])
positions_df = pd.DataFrame(data['element_types'])

#%%
# Add position name to player data
players_df['position'] = players_df['element_type'].map(positions_df.set_index('id')['singular_name'])

# Define relevant attributes for each position
common_attributes = [
    'web_name', 'team', 'now_cost', 'total_points', 'minutes', 'form', 'selected_by_percent',
    'points_per_game', 'value_form', 'value_season'
]

keeper_attributes = common_attributes + [
    'saves', 'saves_per_90', 'clean_sheets', 'clean_sheets_per_90', 'expected_goals_conceded', 
    'goals_conceded', 'penalties_saved', 'bonus', 'bps'
]

defender_attributes = common_attributes + [
    'clean_sheets', 'clean_sheets_per_90', 'expected_goals_conceded_per_90',
    'goals_scored', 'assists', 'expected_goal_involvements_per_90', 'bonus', 'bps'
]

midfielder_attributes = common_attributes + [
    'goals_scored', 'assists', 'expected_goals_per_90', 'expected_assists_per_90',
    'expected_goal_involvements_per_90', 'penalties_order', 'bonus', 'bps'
]

forward_attributes = common_attributes + [
    'goals_scored', 'expected_goals_per_90', 'expected_goal_involvements_per_90',
    'assists', 'starts', 'bonus', 'bps'
]

# Filter and select relevant features per position
keepers = players_df[players_df['position'] == 'Goalkeeper'][keeper_attributes]
defenders = players_df[players_df['position'] == 'Defender'][defender_attributes]
midfielders = players_df[players_df['position'] == 'Midfielder'][midfielder_attributes]
forwards = players_df[players_df['position'] == 'Forward'][forward_attributes]

# Show top 5 from each for quick inspection
print("🧤 Goalkeepers:\n", keepers.head(), "\n")
print("🛡️ Defenders:\n", defenders.head(), "\n")
print("🎯 Midfielders:\n", midfielders.head(), "\n")
print("⚽ Forwards:\n", forwards.head(), "\n")

#%%
import requests
import pandas as pd
import plotly.graph_objects as go

# Step 1: Load main data and filter forwards under 26
url_main = "https://fantasy.premierleague.com/api/bootstrap-static/"
data_main = requests.get(url_main).json()

players_df = pd.DataFrame(data_main['elements'])
positions_df = pd.DataFrame(data_main['element_types'])

# Map position names
players_df['position'] = players_df['element_type'].map(positions_df.set_index('id')['singular_name'])


# Convert birth_date to datetime
players_df['birth_date'] = pd.to_datetime(players_df['birth_date'], errors='coerce')

# Calculate age in years
today = pd.to_datetime('today')
players_df['age'] = (today - players_df['birth_date']).dt.days // 365


# Filter forwards under 26
forwards_u26 = players_df[(players_df['position'] == 'Forward') & (players_df['age'] < 26)]

# Step 2: Fetch weekly data for each forward and aggregate
all_forwards_history = []

for _, player in forwards_u26.iterrows():
    player_id = player['id']
    url_summary = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    data_summary = requests.get(url_summary).json()
    history = pd.DataFrame(data_summary['history'])
    history['player_name'] = player['web_name']
    all_forwards_history.append(history)

# Combine all players' week-by-week history data
df_history = pd.concat(all_forwards_history)

# Filter players to plot - those who have total cumulative points >= 50 at any point
players_to_plot = df_history['player_name'].unique()

fig = go.Figure()

for player in players_to_plot:
    player_data = df_history[df_history['player_name'] == player].sort_values('round')
    
    # Calculate cumulative total points
    player_data['cumulative_points'] = player_data['total_points'].cumsum()
    
    # Filter to players whose cumulative points ever reached at least 50
    if player_data['cumulative_points'].max() >= 50:
        fig.add_trace(go.Scatter(
            x=player_data['round'],
            y=player_data['cumulative_points'],
            mode='lines+markers',
            name=player
        ))

fig.update_layout(
    title="Cumulative Total Points by Forwards Under 26 Over Gameweeks",
    xaxis_title="Gameweek",
    yaxis_title="Cumulative Total Points",
    hovermode="x unified",
    legend_title="Player"
)

fig.show()

#%%
import requests
import pandas as pd
import plotly.graph_objects as go

# Step 1: Load main data and filter midfielders under 26
url_main = "https://fantasy.premierleague.com/api/bootstrap-static/"
data_main = requests.get(url_main).json()

players_df = pd.DataFrame(data_main['elements'])
positions_df = pd.DataFrame(data_main['element_types'])

# Map position names
players_df['position'] = players_df['element_type'].map(positions_df.set_index('id')['singular_name'])

# Convert birth_date to datetime
players_df['birth_date'] = pd.to_datetime(players_df['birth_date'], errors='coerce')

# Calculate age in years
today = pd.to_datetime('today')
players_df['age'] = (today - players_df['birth_date']).dt.days // 365

# Filter midfielders under 26
midfielders_u26 = players_df[(players_df['position'] == 'Midfielder') & (players_df['age'] < 26)]

# Step 2: Fetch weekly data for each midfielder and aggregate
all_midfielders_history = []

for _, player in midfielders_u26.iterrows():
    player_id = player['id']
    url_summary = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    data_summary = requests.get(url_summary).json()
    history = pd.DataFrame(data_summary['history'])
    history['player_name'] = player['web_name']
    all_midfielders_history.append(history)

# Combine all players' week-by-week history data
df_history = pd.concat(all_midfielders_history)

# Filter players to plot - those who have total cumulative points >= 50 at any point
players_to_plot = df_history['player_name'].unique()

fig = go.Figure()

for player in players_to_plot:
    player_data = df_history[df_history['player_name'] == player].sort_values('round')
    
    # Calculate cumulative total points
    player_data['cumulative_points'] = player_data['total_points'].cumsum()
    
    # Filter to players whose cumulative points ever reached at least 50
    if player_data['cumulative_points'].max() >= 50:
        fig.add_trace(go.Scatter(
            x=player_data['round'],
            y=player_data['cumulative_points'],
            mode='lines+markers',
            name=player
        ))

fig.update_layout(
    title="Cumulative Total Points by Midfielders Under 26 Over Gameweeks",
    xaxis_title="Gameweek",
    yaxis_title="Cumulative Total Points",
    hovermode="x unified",
    legend_title="Player"
)

fig.show()







#%%
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Map position names
players_df['position'] = players_df['element_type'].map(positions_df.set_index('id')['singular_name'])
players_df['cost_m'] = players_df['now_cost'] / 10  # Cost in millions

# Loop over each position and create a separate plot
for pos in players_df['position'].unique():
    subset = players_df[players_df['position'] == pos]

    fig = px.scatter(
        subset,
        x='cost_m',
        y='total_points',
        hover_name='web_name',
        hover_data={'cost_m': True, 'total_points': True},
        trendline='ols',
        labels={'cost_m': 'Cost (Millions £)', 'total_points': 'Total Points'},
        title=f'FPL Total Points vs Cost — {pos}s'
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.show()




##%%
## Merge team and position names
#players['team'] = players['team'].map(teams.set_index('id')['name'])
#players['position'] = players['element_type'].map(positions.set_index('id')['singular_name'])
#
## Filter: only players with decent minutes played
#filtered = players[players['minutes'] > 800].copy()
#
## Calculate value metrics
#filtered['value_ppm'] = filtered['total_points'] / filtered['now_cost']  # PPM
#filtered['points_per_90'] = filtered['total_points'] / filtered['minutes'] * 90
#
## Top 10 Moneyball Picks (value-based)
#moneyball = filtered.sort_values(by='value_ppm', ascending=False).head(10)
#
#print("💰 Top 10 Moneyball Picks by Points per Million (PPM):")
#print(moneyball[['web_name', 'team', 'position', 'now_cost', 'total_points', 'value_ppm']])
#
# %%
