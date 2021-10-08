# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:31:50 2021

@author: Rasmus
"""
# Import packages
import json
from pandas import json_normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from difflib import get_close_matches
from mplsoccer import Pitch, VerticalPitch
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as path_effects

### Define functions

# Find all events in the possessions and matches pair, where the player is involved
def filter_pair(player_df, event_df, event_pair = ["possession", "match_id"]):    
    # Convert all pairs of possessions and match id's to tuples
    pos_id_tuples = player_df[event_pair].apply(tuple, axis=1)
    # Convert pandas series of tuples to list of tuples
    keep_tuples = pos_id_tuples.to_list()
    # Construct multiindex dataframe for speed increase
    tuples_in_df = pd.MultiIndex.from_frame(event_df[event_pair])
    # Filter in the temporary df to return the events in the possessions
    return event_df[tuples_in_df.isin(keep_tuples)]

# Plot a chain of events in a possession
def plot_event(dataframe, team_name, event_name = None, 
               women = False, player = None, player_filter = False,
               arrow = False, extra_text = "", half = False):
    
    # Create new figure
    
    if half: 
    # Vertical pitch
        pitch = VerticalPitch(pad_bottom=10,  # pitch extends slightly below halfway line
                              half=True,  # half of a pitch
                              goal_type='box',
                              line_zorder=2,
                              pitch_color='#22312b', line_color='white',
                              shade_color='#22312b',
                              
                              )
        # Draw the pitch
        fig, ax = pitch.draw(figsize=(12, 8), constrained_layout=True, tight_layout=False)
    else:
        pitch = Pitch(pitch_type='statsbomb', 
                  pitch_color='#22312b', line_color='#c7d5cc',
                  #pitch_color='grass', line_color='white'
                  half = half
                  )
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
    fig.set_facecolor('#22312b')
    
    # Filter the event of interest    
    if event_name is not None:
        event_df = dataframe[dataframe.type_name == event_name].set_index('id')
    else: 
        event_df = dataframe.set_index('id')

    # If the data is from the women's world cup
    if women: 
        team_name += " Women's"
    event_df_filter = event_df.loc[event_df["team_name"] == team_name]
    
    # If a specific player should be filtered
    if player != None and player_filter: 
        event_df_filter = event_df_filter.loc[event_df_filter["player_name"] == player]
        
    # Dictionary of colors for possesions
    color_dict = {1: "tab:green", 2: "tab:orange", 3: "tab:red", 
                  4: "tab:blue", 5: "tab:pink", 6: "tab:brown", 
                  7: "tab:purple", 8: "tab:gray", 9: "tab:olive", 
                  10: "tab:cyan", 11: "lime", 12: "gold", 
                  13: "black"}
    
    # Count for colors
    count_index = 0
    # List of symbols to use
    symbol_list = ["o", "x", "s"]
    
    for possession in set(event_df_filter["possession"]):
        # Filter for one possession at a time
        event_df_filter_pos = event_df_filter[event_df_filter["possession"] == possession]
        # Filter out events not used
        event_df_filter_pos = event_df_filter_pos[
            event_df_filter_pos["type_name"].isin(["Shot", "Pass", 
                                                   "Carry", "Dribble",
                                                   "Ball Recipt*"])
            ].reset_index()
        event_df_filter_pos["index"] = event_df_filter_pos.index

        # Count index for colors increment increase     
        count_index += 1
        # Loop over all events in the data 
        for i, event in event_df_filter_pos.iterrows():

            # Coordinates of event start location
            x_start = event['location'][0]
            y_start = event['location'][1]
            # Other information to keep track of  at current event
            event_id = event["id"]
            index = event["index"]
            max_index = max(event_df_filter_pos["index"])
            
            # If we don't want to filter on one event
            if event_name is None:
                event_name = event["type_name"]
                change_event_name_to_none = True
            else: 
                change_event_name_to_none = False

            # Get coordinates of next event for smoother arrows
            if index < max_index: 
                x_end = event_df_filter_pos.loc[event_df_filter_pos["index"] == index+1,
                                                    "location"][index+1][0]
                y_end = event_df_filter_pos.loc[event_df_filter_pos["index"] == index+1,
                                                    "location"][index+1][1]
                label = event_name
            
            elif index == max_index:
                x_end = event[event_name.lower() + "_end_location"][0]
                y_end = event[event_name.lower() + "_end_location"][1]
                label = event_name
            
            # Get color for the specific possession
            color = color_dict[count_index]
            
            # Set default transparency value
            alpha = 0.4
            if player == event["player_name"]:
                # Change if the player of interest is involved
                alpha = 1
            
            # Draw points with a star for shots ending in goal
            if event_name == "Shot" and event_df_filter.loc[event_id, 'shot_outcome_name']=='Goal':
                pitch.scatter(x_start, y_start, ax = ax, s = 300,
                              color = color, alpha = alpha,
                              marker = "o", label = label, zorder = 2)   
                pitch.scatter(x_start, y_start, ax = ax, s = 300,
                              color = "yellow", alpha = alpha,
                              marker = "*", label = label, zorder = 2)  
            
            # Draw arrows for passes, carries and shots
            if arrow:
                # Add an arrow for pass, carries and shots
                if event_name in ["Pass", "Carry", "Shot"]:
                    pitch.arrows(x_start, y_start, 
                                x_end, y_end,
                                lw=10, alpha = alpha, 
                                color= color, ax=ax, zorder = 1
                                )     
            
            # Reset event name if it has been changed
            if change_event_name_to_none:
                event_name = None
    
    # Create non-existent graph elements and use for "custom" legend labels
    pitch.scatter(None, None, ax = ax, marker = symbol_list[0], label = "Pass",
                  color = "black")
    pitch.scatter(None, None, ax = ax, marker = symbol_list[1], label = "Shot",
                  color = "black")
    pitch.scatter(None, None, ax = ax, marker = symbol_list[2], label = "Carry",
                  color = "black")
    pitch.scatter(None, None, ax = ax, marker = "*", label = "Goal",
                  color = "black")

    # Get the actual handles and labels for the legend (one per color + symbol combo)
    handles, labels = ax.get_legend_handles_labels()
    
    # Add legend with "custom" handles and labels
    #ax.legend(handles = handles[-1:], labels = labels[-1:], #handles = handles[-4:], labels = labels[-4:],
    #          labelcolor = "black", prop = {"size": 25}, loc = (0.03, 0.225))#"upper left")
    
    # Set title of graph
    ax.set_title(f"Goal involvement for {player_of_interest_name} {extra_text} during World Cup 2018", 
             fontsize=13.5, pad = 0, color = "white", x = 0.5, fontweight='bold')

    # Show plot
    plt.show()



# ID for male world cup 2018
competition_id = 43
season_id = 3

#Load the list of matches for this competition
path = f"Statsbomb/data/matches/{competition_id}/{season_id}.json"
with open(path) as f:
    matches = json.load(f)

# Select the nation and player to analyze
team_required = "Argentina"
player_of_interest = 'Lionel Andres Messi'
player_of_interest_name = "Lionel Messi"

# Find match_ids for the required team
match_ids = [ match["match_id"] for match in matches if 
 match['home_team']['home_team_name'] == team_required or 
 match['away_team']['away_team_name'] == team_required
 ]

# Empty data frame to be appended to with all play by play events
df = pd.DataFrame()

# Select the given matches and append to data frame
for match_id in match_ids:
    match_path = f"Statsbomb/data/events/{match_id}.json"
    with open(match_path, encoding='utf-8') as data_file:
        data = json.load(data_file)
        temp_df = json_normalize(data, sep = "_").assign(match_id = match_id)
        df = df.append(temp_df)

# Fix player name if it is incorrect (most of the time it works)
if player_of_interest not in set(df["player_name"].dropna()):
    player_of_interest = get_close_matches(player_of_interest, set(df["player_name"].dropna()))[0]

# Filter only events directly involving the player (i.e. performing an action)
direct_involvement = df.loc[(df["player_name"] != player_of_interest)]

# Get possesions that inclue the player
unique_pos = set(direct_involvement.possession)

# Get the possession events where the player is invloved
player_possesion_event = filter_pair(direct_involvement, df)

# Find the 5 opportunities with highest expected goals where the player is involved
xg_5 = player_possesion_event[(player_possesion_event['shot_statsbomb_xg'].notna()) &
                    ((player_possesion_event["team_name"] == team_required)) &
                    (player_possesion_event["shot_type_name"] != "Penalty")].shot_statsbomb_xg.sort_values(
                        ascending = False)[0:5]
# Print xG (player)
print(xg_5)

# Find the 5 opportunities with highest expected goals where the player is not 
# necessarily involved
total_xg_5 = df[(df['shot_statsbomb_xg'].notna()) &
          ((df["team_name"] == team_required)) &
          (df["shot_type_name"] != "Penalty")].shot_statsbomb_xg.sort_values(
              ascending = False)[0:5]
# Print xG (team)
print(total_xg_5)

# Get the lowest xG for top 5
min_xg = min(xg_5)

# Get the indices of the goals
goal_indices = player_possesion_event[(player_possesion_event["shot_outcome_name"] == "Goal") & 
                      (player_possesion_event["shot_type_name"] != "Penalty") & 
                      (player_possesion_event["team_name"] == team_required)].player_name.index

# Get the possesion number and match_id which ends in a goal scoring opportunity
possession_match_goal = player_possesion_event.iloc[goal_indices][["possession", "match_id"]]

# Filter in the temporary df
possessions_end_in_goal = filter_pair(possession_match_goal, df)

# Match_id: 7564 = Nigeria, 7531 = Iceland, 7580 = France
# Plot possesions ending in a goal
plot_event(possessions_end_in_goal[possessions_end_in_goal["match_id"] == 7580],
           team_name = team_required, player = player_of_interest, arrow = True, 
           extra_text = "in Argentina vs. France", half = True)


### Shots involving/not involving the player
# Possession with a shot
team_shots_df = df[(df["team_name"] == team_required) &
                   (df["type_name"] == "Shot") &
                   (df["shot_type_name"] != "Penalty")]
team_shots = filter_pair(team_shots_df, df)

# Empty dataframe to contain the events within the alloted timespan before shot
events_prior_to_shot = pd.DataFrame()

# Keep only events 15 seconds before a shot by looping over unique possesion &
# match_id combinations
for time, possession, match in zip(team_shots_df.timestamp,
                                   team_shots_df.possession, 
                                   team_shots_df.match_id):
    # Extract the events for the given possession & match combination
    temp = team_shots[(team_shots["possession"].isin([possession])) & 
                      (team_shots["match_id"].isin([match]) 
                       )]

    # Get the timestamp of the shot
    shot_time = temp.loc[temp["type_name"] == "Shot", ["minute", "second"]]
    # Allowed time interval between event and shot
    diff = 30
    # If there are multiple shots in the possession
    # NOTE: This adds duplicates of each events, so they must be dropped later
    if len(shot_time) > 1:
        # Get the difference (in seconds) between the two shots
        two_shots_time_diff = abs(shot_time.diff().second.iloc[-1])
        # Add the time between the two shots as buffer to include both
        diff += two_shots_time_diff
        # Extract minute and second
        minute = shot_time.iloc[-1].to_list()[0]
        second = shot_time.iloc[-1].to_list()[1]
    else:   
        # Extract minute and second
        minute = shot_time.minute.to_list()[0]
        second = shot_time.second.to_list()[0]
    
    # Append possession events to the dataframe
    events_prior_to_shot = events_prior_to_shot.append(
        temp[(
            (temp["minute"]-minute == 0) & # Same minute
            (abs(temp["second"]-second) <= diff) # No more than 15 second difference
            ) | 
            (
            (temp["minute"]-minute == -1) & # Previous minute
            (60-abs(temp["second"]-second) <= diff) & # No more than 15 second difference
            (second <= diff) # The seconds are less than 16
            )] 
        )
    
# The above produces duplicates for possessions with 2 (or more) shots. 
# Therefore, drop duplicates rows:
events_prior_to_shot.drop_duplicates(subset = "id", inplace = True)
    
# events_prior_to_shot.type_name.value_counts() := 56 shots
# Remove events without a location (camera on/off etc.)
events_prior_to_shot = events_prior_to_shot.loc[(events_prior_to_shot["location"].notna())]

# Get player-wise events prior to a shot
player_prior_shot = events_prior_to_shot[events_prior_to_shot["player_name"] == player_of_interest]

# Find possesions prior to shot for the player
player_prior_shot = filter_pair(player_prior_shot, events_prior_to_shot)
# Find the shots the player is involved with
team_prior_shot = player_prior_shot[player_prior_shot["team_name"] == team_required]

# Extract events of the player which are not ball receipt
player_prior_shot_pos = player_prior_shot[
      (player_prior_shot["player_name"] == player_of_interest) &
      (~player_prior_shot["type_name"].isin(["Ball Receipt*"]))]

# Extract shots by the team
team_shots = team_prior_shot[#(player_prior_shot["location"].notna()) & 
      (team_prior_shot["team_name"] == team_required) &
      (team_prior_shot["type_name"].isin(["Shot"]))]

# Filter so only game against France is included
player_prior_shot_pos_subset = player_prior_shot_pos[player_prior_shot_pos["match_id"] == 7580]
team_shots = team_shots[team_shots["match_id"] == 7580]
# Extract (x,y) coordinates for all events which the player is involved and is 
# not a ball receipt

# All games
# x = [i[0] for i in player_prior_shot_pos.location]
# y = [i[1] for i in player_prior_shot_pos.location]

# Specific game 
x = [i[0] for i in player_prior_shot_pos_subset.location]
y = [i[1] for i in player_prior_shot_pos_subset.location]


# Get (x,y) coordinates for shots by the team
shot_x = [row.location[0] for i, row in team_shots.iterrows() if 
     row["type_name"] == "Shot"]
shot_y = [row.location[1] for i, row in team_shots.iterrows() if 
     row["type_name"] == "Shot"]

# Indices for shots not involving the player
shots_diff_index = set(team_shots_df[team_shots_df["match_id"] == 7580].index) - set(team_shots.index)

# Shots without player involved
team_shots_no_involvement = team_shots_df.loc[shots_diff_index]

# Coordinates of shots without player
shot_not_messi_x = [row.location[0] for i, row in team_shots_no_involvement.iterrows() if 
     row["type_name"] == "Shot"]
shot_not_messi_y = [row.location[1] for i, row in team_shots_no_involvement.iterrows() if 
     row["type_name"] == "Shot"]

# Heatmap for positions
path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'),
            path_effects.Normal()]

# Vertical pitch
pitch = VerticalPitch(pad_bottom=0.5,  # pitch extends slightly below halfway line
                      half=True,  # half of a pitch
                      goal_type='box',
                      line_zorder=1,
                      pitch_color='#22312b', line_color='white',
                      shade_color='#22312b'
                      )
# Draw the pitch
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#f4edf0')

# Calculate bin statistic for positional map
bin_statistic = pitch.bin_statistic_positional(x, y, statistic='count',
                                               positional='full', normalize=True)
# Draw heatmap positional
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='Blues', edgecolors='#22312b', zorder = 0)

# Add labels to the heatmap with % and remove 0's
labels = pitch.label_heatmap(bin_statistic, color = "black",
                             fontsize=28,
                             ax=ax, ha='center', va='top', 
                             str_format='{:.0%}',
                             exclude_zeros = True,
                             path_effects=path_eff)

# Add the actions by the player
pitch.scatter(x, y, ax = ax, color = "gold", s = 300, alpha = 1, 
              marker='o', edgecolor = "black", label = "Actions (Messi)")

# Add the shots by the team
pitch.scatter(shot_x, shot_y, ax = ax, color = "tab:green", s = 300, alpha = 1, zorder = 2, 
              marker='*', edgecolor = "black", label = "Shots (Argentina)\nMessi involved")
pitch.scatter(shot_not_messi_x, shot_not_messi_y, ax = ax, color = "tab:red", 
              s = 300, alpha = 1, zorder = 2,
              marker='*', edgecolor = "black", label = "Shots (Argentina)\nMessi not involved")

# Set title for graph
ax.set_title(f'Action heatmap for {player_of_interest_name} in Argentina vs. France within 15 seconds prior to a shot in World Cup 2018', 
             fontsize=17,  color = "black", fontweight='bold')

# Add legend and make readable
ax.legend(prop = {"size": 18}, loc = "upper left")
