#%%
import pl 

# Load data
players_df, positions_df = pl.load_fpl_data()
#%%
# 1. Plot points vs. cost per position
pl.plot_points_vs_cost(players_df, positions_df)
#%%
# 2. Plot cumulative performance of midfielders under 26 with at least 50 points
#Defender, Forward, Midfielders,  - select the players you want to keep
pl.plot_cumulative_points(age_limit=30, position='Goalkeeper', min_cum_points=50)

#%%
players_df = pl.fetch_and_forecast_players()

players_to_keep = ['Erling Haaland', 'Alexander Isak']  # example
players_to_exclude = []  # example

selected_players, model = pl.optimize_fpl_team(players_df, players_to_keep, players_to_exclude)

print("Optimized team:", selected_players)


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
