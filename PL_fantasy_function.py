#%%
# Import your custom FPL analysis package
import pl  

#%% Load FPL data
# Load player and position data from the FPL API or source
players_df, positions_df = pl.load_fpl_data()

#%% 1. Visualize Points vs. Cost per Position
# Helps identify value picks by plotting total points against cost for each position
pl.plot_points_vs_cost(players_df, positions_df)

#%% 2. Plot Cumulative Points for Goalkeepers under 30 with at least 50 points
# You can change the position or age limit for different visualizations
pl.plot_cumulative_points(
    age_limit=30, 
    position='Goalkeeper', 
    min_cum_points=50
)

#%% 3. Forecast FPL player performance and run team optimization
# This step loads forecasted performance and suggests an optimized team

# Refresh data with forecasted points
players_df = pl.fetch_and_forecast_players()

# Optional filters for players to definitely include or exclude
players_to_keep = []#['Erling Haaland']
players_to_exclude = []  # Add player names if any should be avoided

# Get the optimized team
selected_players, model = pl.optimize_fpl_team(
    players_df, 
    players_to_keep, 
    players_to_exclude
)

# Print the optimized FPL team
print("⚽ Optimized 15-player squad:", selected_players)
print("-" * 30)

#%%
# --- NEW ADDITION: Optimize the Starting 11 ---
# Filter the original players_df to only include the 15 selected players
squad_df = players_df[players_df['name'].isin(selected_players)].copy()

# Get the optimized Starting 11 from the squad
starting_11 = pl.optimize_starting_11(squad_df)

# Print the Optimized Starting 11

print('total cost: ', starting_11['now_cost'].sum())

print("🌟 Optimized Starting 11:")
starting_11.head(11)
#%% 4. Query model using LangChain assistant
# Ask questions in natural language, and the assistant will respond based on the player data
#players_df['selected'] = players_df['selected'].replace('_', ' ')  # Replace underscores with spaces for better readability
#my_team_df = players_df[players_df['name'].isin(selected_players)]
#q = "Question 1: Who should be captain for my team? \
#Question 2: Should I use Wildcards & Chips for next week?" \
#
#response = pl.fpl_langchain_advisor(q, my_team_df)
#print("🤖 Advisor Response:", response)


#%% 5. Get FPL injury news
# Fetch the latest injury news for players in the FPL
#news = pl.get_fpl_injury_news()
#print(news)
#%%
# [OPTIONAL] Manual Moneyball-style filtering (Commented Out)
# This block manually calculates Points Per Million (PPM) and Points per 90 minutes
# You can uncomment this for deeper manual analysis.

"""
# Merge team and position names
players_df['team'] = players_df['team'].map(teams.set_index('id')['name'])
players_df['position'] = players_df['element_type'].map(positions_df.set_index('id')['singular_name'])

# Filter: only players with at least 800 minutes played
filtered = players_df[players_df['minutes'] > 800].copy()

# Calculate value metrics
filtered['value_ppm'] = filtered['total_points'] / filtered['now_cost']  # Points Per Million
filtered['points_per_90'] = filtered['total_points'] / filtered['minutes'] * 90

# Top 10 Moneyball picks based on PPM
moneyball = filtered.sort_values(by='value_ppm', ascending=False).head(10)

print("💰 Top 10 Moneyball Picks by Points per Million (PPM):")
print(moneyball[['web_name', 'team', 'position', 'now_cost', 'total_points', 'value_ppm']])
"""
