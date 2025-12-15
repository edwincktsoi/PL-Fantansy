#%%
# 1. Setup
import os
import pl  # Import our cleaned custom package

# SECURITY NOTE: Set your API keys here or in your system environment variables.
# Do NOT upload this file with real keys to public repositories.
os.environ['GOOGLE_API_KEY'] = "YOUR_GOOGLE_KEY_HERE"
os.environ['LANGCHAIN_API_KEY'] = "YOUR_LANGCHAIN_KEY_HERE"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

#%%
# 2. Load Data & Visualize
# Load basic player data
players_df, positions_df = pl.load_fpl_data()

# Visualize: Value for Money (Points vs Cost)
pl.plot_points_vs_cost(players_df, positions_df)

#%%
# 3. Forecast & Optimize Squad (15 Players)
# This step fetches recent form and upcoming fixtures to project points
forecast_df = pl.fetch_and_forecast_players()

# Define constraints
players_to_keep = [] # e.g. ['Erling Haaland']
players_to_exclude = []

# Run Optimization
selected_names, model = pl.optimize_fpl_team(
    forecast_df, 
    players_to_keep, 
    players_to_exclude
)

print(f"⚽ Optimized 15-player squad: {selected_names}")

#%%
# 4. Optimize Starting 11 (Best XI + Captain)
# Filter dataframe to only our selected squad
squad_df = forecast_df[forecast_df['name'].isin(selected_names)].copy()

# Run Lineup Optimization
starting_11 = pl.optimize_starting_11(squad_df)

print(f"💰 Starting XI Cost: £{starting_11['now_cost'].sum():.1f}m")
print(f"🚀 Projected Points: {starting_11['projected_points'].sum():.1f}")
print("\n🌟 Optimized Starting 11:")
#print(starting_11)  # Use print(starting_11) if not in Jupyter/Spyder
starting_11.head(15)

#%%
# 5. AI Advisor (Optional)
# Ask questions about your specific team
#my_team_df = forecast_df[forecast_df['name'].isin(selected_names)]
#
#questions = "Who should be captain? Should I use a wildcard?"
#response = pl.fpl_langchain_advisor(questions, my_team_df)
#
#print("\n🤖 AI Advisor Response:")
#print(response)
#
##%%
## 6. Injury News (Optional)
#news = pl.get_fpl_injury_news()
#print("\n🚑 Injury News Summary:")
#print(news)