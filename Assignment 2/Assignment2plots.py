# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 07:52:11 2021

@author: Rasmus
"""
import numpy as np
import pandas as pd
import json

from Assignment2 import tag_split, add_xg_variables

#Plotting
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch

# Custom colormaps
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Select the nation(s) of interest
nation_list = ["Italy"]
# Load information of players in a given nation
for nation in nation_list:
    with open(f'Wyscout/events/events_{nation}.json', encoding='utf-8') as f:
        temp = json.load(f)

# Save the nation into a data frame
events = pd.DataFrame(temp)

# Read information about players
with open('Wyscout/players.json', encoding = "utf-8") as f:
    temp = json.load(f)
    
# Store all players in a data frame
players = pd.DataFrame(temp)

# Find the player id of the play we wish to investigate
playerId_of_interest = 286215

# Get the name of the goalkeeper
player_of_interest = players.loc[players["wyId"] == playerId_of_interest, "shortName"].iloc[0]

# Find events with the goalie of interest
goalie = events[events["playerId"] == playerId_of_interest]

#### Information about save attempts ####
# Find goalie save attempts
save = goalie[goalie["eventName"] == "Save attempt"]

# Find shots before goalie save attempt
shots_prior = events.loc[save.index-1]
shots_prior = shots_prior[shots_prior["eventName"].isin(["Shot", "Free kick"])]

# Add variables regarding coordinates
shots_prior = add_xg_variables(shots_prior, players)

# Add columns regarding tags
shots_prior = tag_split(shots_prior)

#### Information about passing ####
# Get all passes from the goalie
passing = goalie[goalie["eventName"] == "Pass"]

# Add variables regarding coordinates
passing = add_xg_variables(passing, players)

# Add information regarding tags
passing = tag_split(passing)

### Figure 1: Shots & goals against/conceded
# Pitch
pitch = VerticalPitch(pad_bottom=0.5,  # pitch extends slightly below halfway line
                      half=True,  # half of a pitch
                      pitch_type = "wyscout",
                      goal_type='box',
                      line_zorder=1,
                      pitch_color='grass', line_color='white', stripe=True
                      )
# Draw the pitch
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#aabb97')

pitch.scatter(shots_prior.startX, shots_prior.startY, ax = ax, color = "gold", 
              s = 300, alpha = 0.5, 
              marker='o', edgecolor = "black", label = "Shots")
# Add the goals
pitch.scatter(shots_prior.loc[shots_prior["tag101"], "startX"], 
              shots_prior.loc[shots_prior["tag101"], "startY"],
              ax = ax, color = "tab:red", s = 300, alpha = 1, zorder = 2, 
              marker='*', edgecolor = "black", label = "Goals")

# Add legend
ax.legend(prop = {"size": 24}, loc = "upper left")

# Add title
plt.title(f"Shots and goals conceded for {player_of_interest} during the 2017-2018 Serie A season", 
          fontsize=24)

### Figure 2: Save percentages
# Define a new colormap based on green
greens = cm.get_cmap('Greens', 256)
newcolors = greens(np.linspace(0, 1, 256))
green = np.array([0, 255/256, 0, 0.2])
newcolors[:25, :] = green
newgreen = ListedColormap(newcolors)

# Define a new colormap based on blue
blues = cm.get_cmap('Blues', 256)
newcolors = blues(np.linspace(0, 1, 256))
blue = np.array([0, 0, 255/256, 0.2])
newcolors[:25, :] = green
newblue = ListedColormap(newcolors)

# Pitch attributes
pitch = VerticalPitch(pad_bottom = -39,
                      half=True,  # half of a pitch
                      pitch_type = "wyscout",
                      goal_type='box',
                      line_zorder=1,
                      pitch_color='grass', line_color='white', stripe=True
                      )
# Draw the pitch
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#aabb97')

# Calculate how many shots and goals occur in each bin
shot_bin = pitch.bin_statistic(shots_prior.startX, shots_prior.startY,
                               statistic='count', bins=(18, 8))
goal_bin = pitch.bin_statistic(shots_prior.loc[shots_prior["tag101"], "startX"], 
                               shots_prior.loc[shots_prior["tag101"], "startY"], 
                               statistic='count', bins=(18, 8))

# Calculate save percentage for each bin
save_percentage = 1 - goal_bin["statistic"]/shot_bin["statistic"]

# Replace NaN values with 0
shot_bin["statistic"] = np.select([np.isnan(save_percentage)], [0], save_percentage)

# Add the bins with information over save percentages
pitch.heatmap(shot_bin, ax=ax, cmap=newgreen, edgecolors=None, zorder=2)

# Add labels for the percentages
pitch.label_heatmap(shot_bin, color="white", fontsize=16,
                    ax=ax, ha='center', va='top', zorder=2,
                    str_format='{:.0%}',
                    exclude_zeros=True)

# Add title
plt.title(f"Save percentage for {player_of_interest} during the 2017-2018 Serie A season", 
          fontsize=24)

### Figure 3: Passing success rate
# Find passes that are successful
success = pitch.bin_statistic_positional(passing.loc[passing["tag1801"], "endX"], 
                                         passing.loc[passing["tag1801"], "endY"],
                                         statistic='count',
                                         positional='full')

# Find unsuccesful passes; will be used as a placeholder mainly for passes per 90 minutes
pass_per_90 = pitch.bin_statistic_positional(passing.loc[~passing["tag1801"], "endX"], 
                                      passing.loc[~passing["tag1801"], "endY"],
                                      statistic='count',
                                      positional='full')

# Calculate total amount of passes
total = pitch.bin_statistic_positional(passing.loc[:, "endX"], 
                                       passing.loc[:, "endY"],
                                       statistic='count',
                                       positional='full')

# Go through all bins created
for i in range(len(success)):
    # Calculate proportion of successful passes per bin
    prop_success = success[i]["statistic"] / (total[i]["statistic"] )
    # Replace values for total with proportion of success (with 0 instead of NaN)
    total[i]["statistic"] = np.select([np.isnan(prop_success)], [0], prop_success)
    # Calculate passes per 90 minutes
    pass_per_90[i]["statistic"] = 90 * success[i]["statistic"] / 1160 

# Pitch attributes
pitch = VerticalPitch(pad_bottom=0,  # pitch extends slightly below halfway line
                      half=False,  # half of a pitch
                      pitch_type = "wyscout",
                      goal_type='box',
                      line_zorder=1,
                      pitch_color='grass', line_color='white', stripe=True
                      )
# Draw the pitch
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#aabb97')

# Percentage of completed passes
pitch.heatmap_positional(total, ax=ax, cmap=newgreen, edgecolors='black')
pitch.label_heatmap(total, color="white", fontsize=28,
                    ax=ax, ha='center', va='bottom',
                    str_format='{:.0%}',
                    exclude_zeros = False)

# Denote number of successful passes
pitch.label_heatmap(success, color="white", fontsize=16,
                    ax=ax, ha='center', va='top',
                    str_format='\n#: {:.0f}',
                    exclude_zeros = False)

# Denote number of successful passes per 90 minutes
pitch.label_heatmap(pass_per_90, color="white", fontsize=16,
                    ax=ax, ha='center', va='top',
                    str_format='\n\n{:.2f} /90\'',
                    exclude_zeros = False)

# Set title and subtitle for graph
plt.suptitle(f"Pass success for {player_of_interest} during 2017-2018 Seria A season", 
             y=1.03, fontsize=16, fontweight='bold')

plt.title(f"Success is based on end coordinates of passes made by {player_of_interest}\n where # denotes number of successful passes to a given rectangle", 
          fontsize=14)

