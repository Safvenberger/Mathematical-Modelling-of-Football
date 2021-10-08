# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 18:31:17 2021

@author: Rasmus Säfvenberg
"""

#The basics
import pandas as pd
import numpy as np
import json
import re

#Statistical fitting of models
from sklearn.linear_model import LogisticRegression

# Suppress some warnings which do not impact the results for cleaner console output
pd.set_option("mode.chained_assignment", None)

# Function to split a column containing tags into multiple binary columns (one per tag)
def tag_split(df):  
    """ 
    Parameters:
        df: a data frame of one particular event typically (e.g. shots)
    Returns:
        a modified data frame where the "tags" column has been split into
        multiple binary ones
    """
    # Split tags into multiple columns (creates a new data frame)
    tags_df = pd.json_normalize(df["tags"])
    
    # Number of unique tags to consider
    n_tags = len(pd.json_normalize(df["tags"]).columns)
    tags_df = tags_df.rename(columns={i:f"tag{i}" for i in range(n_tags)})
    
    # Reformat the tag columns into just numerical values
    for i in range(n_tags):
        tags_df[f"tag{i}"] = [list(i.values())[0] if i is not None else None for i in tags_df[f"tag{i}"]]
    
    # This flattens the pandas column into one series
    flat_tags = pd.Series([y for x in df['tags'] for y in x])
    
    # Get a list of all the tags which occur
    flat_tags = [list(v.values())[0] for k, v in flat_tags.items()]
    
    # Add tags as separate columns sorted from lowest to highest numerical value
    for tagId in np.sort(pd.unique(flat_tags)):
        df[f"tag{tagId}"] = np.where(tags_df[[f"tag{i}" for i in range(n_tags)]
                                             ].eq(tagId).any(1), True, False)

    return df


def add_indicator_variables(df):
    """ 
    Parameters:
        df: a data frame of one particular event typically (e.g. shots)
            which has had tags split previously by tag_split()
    Returns:
        a modified data frame with new indicator variables for where the shot
        ended up and whether is was taken with the players preferred foot
    """
    
    # Shots in the center of the goal 
    center = np.where(df[['tag1201', 'tag1203', 'tag1207']].eq(True).any(1), True, False)

    # Top corners 
    top_corners = np.where(df[['tag1208', 'tag1209']].eq(True).any(1), True, False)
    
    # Bottom corners 
    bottom_corners = np.where(df[['tag1202', 'tag1205']].eq(True).any(1), True, False)
    
    # Middle corners
    middle_corners = np.where(df[['tag1204', 'tag1206']].eq(True).any(1), True, False)

    # Shot was taken with foot
    #foot = np.where(df[['tag401', 'tag402']].eq(True).any(1), True, False)
    
    # Shot was taken with preferred foot
    pref_foot = [True if ((v["foot"] == "right" and v["tag402"]) or 
                          (v["foot"] == "left" and v["tag401"]) or 
                          (v["foot"] == "noth" and (v["tag401"] or v["tag402"])) ) 
                 else False for i, v in df.iterrows()]
    
    # Add columns to the dataframe itself
    df["center"] = center
    df["pref_foot"] = pref_foot
    df["top_corners"] = top_corners
    df["bottom_corners"] = bottom_corners
    df["middle_corners"] = middle_corners
    
    return df

        
def add_xg_variables(pbp_df, players):
    """ 
    Parameters:
        xg_df: data frame containing postXG values alongside player id
        pbp_df: play by play data frame with all the events
    Returns:
        a modified version of pbp_df with extra columns useful for creating
        an xG model, namely distance, angle, and eventSecDiff
    Notes: 
        the quality of eventSecDiff depends on the quality of data collection
    """
    # Include information about a player's prefered foot
    pbp_df = pbp_df.merge(players[["wyId", "foot"]], left_on = "playerId", right_on = "wyId", how = "left")
    
    # Split the positions into one columns for each coordinate
    pbp_df[["start", "end"]] = (pd.json_normalize(pbp_df["positions"]))
    pbp_df[["startY", "startX"]] = (pd.json_normalize(pbp_df["start"]))
    pbp_df[["endY", "endX"]] = (pd.json_normalize(pbp_df["end"]))
    
    # Convert to a pitch of coordinates 105x68 with same direction as wyscout data
    # Direction of wyscout: 0 => 105 (closer to opposition goal) & 0 => 68 (top to bottom)
    pbp_df[["startX_meter", "endX_meter"]] = pbp_df[["startX", "endX"]].apply(lambda x: 105 * abs(100-100-x) / 100)
    pbp_df[["startY_meter", "endY_meter"]] = pbp_df[["startY", "endY"]].apply(lambda y: 68 * abs(100-100-y) / 100)
    
    # Calculate distance to goal
    x = (105/100*(100-pbp_df["startX"]))
    y = (65/100*abs(pbp_df["startY"]-50))
    pbp_df["distance"] = np.sqrt( x**2 + y**2)
    
    # Calculate angle of shot
    pbp_df["angle"] = np.arctan(7.32 * x / ( x**2 + y**2 - (7.32/2)**2))
    pbp_df["angle"] = pbp_df["angle"].apply(lambda a: np.pi + a if a < 0 else a)
    
    # Create a column of differences between consecutive events and shift them
    # to preceeding events to use later (get time between shot and save attempt)
    pbp_df["eventSecDiff"] = pbp_df.eventSec.diff().shift(-1)
    
    #(105 - pbp_df["startX_meter"]) / pbp_df["eventSecDiff"]
    # Replace NaN values with 0
    pbp_df["eventSecDiff"].fillna(0, inplace=True)
    
    # Replace any values of infinity with 0 to prevent model from breaking
    pbp_df.replace([np.inf, -np.inf], 0, inplace=True)
    return pbp_df


def played_minutes(matches):
    """ 
    Parameters:
        matches: a data frame containing all matches for a given season
    Returns:
        a data frame with overall information regarding games and minutes played
        as well as one column (matchId) per match with all relevant data.
    Notes: 
        this assumes that the games played were 90 minutes (i.e. only league games)
        players sent off are not adjusted for here, see red_cards()
    """
    
    # Create empty dictionary with playerId as key
    player_dict = {}
    
    # Can use matches.duration to change minutes if desired
    for i, match in matches.iterrows():
        # Get the dictionary of all match information
        teamData = match["teamsData"]
        # Get match id
        matchId = match["wyId"]
        # Loop over all entries for both teams
        for teamId, data in teamData.items():
            # Go over the three formations (bench, lineup)
            for formation in data["formation"]:
                if formation in ["bench", "lineup"]:
                    # Give players minutes played for baseline
                    minutes = ["bench", "lineup"].index(formation)*90
                    # Decide if the player played in the game or only sat on bench
                    played = ["bench", "lineup"].index(formation)
                    # Loop over all players
                    for player in data["formation"][formation]:
                        # Get all data for the player in the given match
                        matchData = {
                                matchId:{"played": played,
                                         "minutes": minutes,
                                         "goals": player["goals"], 
                                         "ownGoals": player["ownGoals"], # Little wonky?
                                         "yellowCards": player["yellowCards"],
                                         "redCards": player["redCards"],
                                         "teamId": teamId, 
                                         "side": data["side"], 
                                         "finalScore": match["label"].split(", ")[1].replace(" ", ""),
                                         "matchWinner": match["winner"]}
                                }
                        # See if player is in dictionary
                        if player["playerId"] not in player_dict:
                            # Add each player to the dictionary
                            player_dict[player["playerId"]] = matchData
                            player_dict[player["playerId"]]["gamesPlayed"] = played
                            player_dict[player["playerId"]]["minutesPlayed"] = minutes
                        else:
                            # Combine (append) the matches 
                            player_dict[player["playerId"]].update(matchData)
                            player_dict[player["playerId"]]["gamesPlayed"] += played
                            player_dict[player["playerId"]]["minutesPlayed"] += minutes
                            
                else:
                    # If there are substitutions, loop over all of them
                    if data["formation"][formation] != "null":
                        for sub in data["formation"][formation]:
                            if sub["playerIn"] != 0:
                                # Add minutes if subbed in
                                player_dict[sub["playerIn"]][matchId]["minutes"] += abs(90 - sub["minute"])
                                
                                # Change played status to 1 
                                player_dict[sub["playerIn"]][matchId]["played"] = 1
                                # Add to total games and minutes played
                                player_dict[sub["playerIn"]]["gamesPlayed"] += 1
                                player_dict[sub["playerIn"]]["minutesPlayed"] += abs(90 - sub["minute"])
                                
                                # Remove minutes if subbed out during game
                                player_dict[sub["playerOut"]][matchId]["minutes"] -= abs(90 - sub["minute"])
                                # Remove total minutes played if subbed out
                                player_dict[sub["playerOut"]]["minutesPlayed"] -= abs(90 - sub["minute"])

                        
    # Convert into dataframe
    playerDF = pd.DataFrame.from_dict(player_dict).transpose()
    # Rename the index
    playerDF = playerDF.rename(columns={'index': 'playerId'})
    
    # Reorder columns
    cols_to_order = ['gamesPlayed', 'minutesPlayed']
    new_columns = cols_to_order + (playerDF.columns.drop(cols_to_order).tolist())
    playerDF = playerDF[new_columns]
    
    return playerDF


def red_cards(pbp_df, playerDF):
    """ 
    Parameters:
        pbp_df: the play by play data frame with all events
    Returns:
        a modified version of playerDF where the minutes have been adjusted
        for players getting sent off
    Notes: 
        this assumes that the games played were 90 minutes (i.e. only league games)
    """
    # Find the fouls
    nation_foul = pbp_df[pbp_df.eventName == "Foul"]
    # Split into binary columns
    nation_foul = tag_split(nation_foul)
    # Create a boolean series of whether it was a red card or not
    red_cards = nation_foul["tag1701"]
    # Keep only red cards (i.e. true values)
    red_cards = red_cards[red_cards]
    
    # Second yellow cards
    second_yellow = nation_foul["tag1703"]
    # Keep only second yellow cards (i.e. true values)
    second_yellow = second_yellow[second_yellow]
    
    # Combine the two indices to get all players sent off
    sent_off_index = red_cards.index.union(second_yellow.index)
    
    # Find the players that got sent off
    #players_sent_off = pbp_df.iloc[red_cards.index].playerId
    
    #gks_sent_off = players_sent_off.isin(goalkeepers.wyId)
    #gks_sent_off = gks_sent_off[gks_sent_off]
        
    # Extract relevant columns to speed up looping
    sent_off = pbp_df.loc[sent_off_index, ["matchId", "playerId", "matchPeriod","eventSec"]]
    
    for i, row in sent_off.iterrows():
        # Extract relevant values
        match = row["matchId"]
        player = row["playerId"]
        half = row["matchPeriod"]
        eventSec = row["eventSec"]
        # Convert seconds to minutes
        eventMin = eventSec / 60
        # If it happened during extra time
        if eventMin > 45:
            eventMin = 45
        # Adjust minutes accordingly to which half the player got sent off in
        if half == "1H":
            game_time_lost = round(90 - eventMin)
        else:
            game_time_lost = round(90 - 2*eventMin)
        # Remove minutes after sending off for the game
        playerDF[match][player]["minutes"] -= game_time_lost
        # Remove minutes overall
        playerDF.loc[player, "minutesPlayed"] -= game_time_lost

    return playerDF


def own_goals(pbp_df, playerDF, goalkeepers):
    """ 
    Parameters:
        pbp_df: play by play data frame with all the events
        playerDF: data frame of all shots taken with feet not from a set piece
        goalkeepers: data frame with metadata of all goalkeepers (i.e from players.json)
    Returns:
        shotsXG_gk: a data frame with summarized values for postXG from shots
        headersXG_gk: a data frame with summarized values for postXG from free kicks
        freekicksXG_gk: a data frame with summarized values for postXG from penalites
    Notes: 
        no xG model is trained for penalties, since WyScout use a fixed value
        of 0.76 for this xG. Similar values apply for these data too, 
        as the number of penalties scored / total number of penalties on target
        is roughly 0.76.
    """
    # Find own goals
    first_tag = pbp_df["tags"].apply(lambda x: x[0].get('id', 0) if len(x) > 0 else 0)
    own_goal_index = first_tag[first_tag == 102].index
    own_goals = pbp_df.loc[own_goal_index, ]
    
    # NOTE: This assumes that one keeper played the enitre match.
    for i, v in own_goals.iterrows():
        # Get team and match id
        team = v["teamId"]
        match = v["matchId"]
        # Find players who were played or was on the bench
        players_in_match = playerDF[playerDF[match].notna()].index
        # Find the id of the goalkeepers in the match
        gk_id = goalkeepers[(goalkeepers["wyId"].isin(players_in_match))].wyId
        # Find the goalkeepers in the match
        goalies_in_match = playerDF.loc[(playerDF[match].notna()) &
                                         playerDF.index.isin(gk_id), match]
        
        # Add the goalkeeper who conceded the own goal to own_goals data frame
        for j in range(len(goalies_in_match)):
            goalie = goalies_in_match.iloc[j]
            if int(goalie["teamId"]) == team and int(goalie["minutes"]) > 0:
                own_goals.loc[i, "goalkeeperId"] = goalies_in_match.index[j]
                
    return own_goals


def xg_model_prep(pbp_df):
    """ 
    Parameters:
        pbp_df: play by play data frame with all the events
    Returns:
        shots: a data frame containing all tags and indicator variables over accurate shots
        headers: a data frame containing all tags and indicator variables over accurate headers
        freekicks: a data frame containing all tags and indicator variables over accurate free kicks
        penalties: a data frame containing all tags over accurate penalty kicks
    Notes: 
        indicator variables are not added for penalty kicks since the data is 
        more scarce 
    """
    # Filter based in order to create one model for different type of shots
    shots = pbp_df[pbp_df['subEventName'].isin(["Shot"])]
    penalties = pbp_df[pbp_df['subEventName'].isin(["Penalty"])]
    freekicks = pbp_df[pbp_df['subEventName'].isin(["Free kick shot"])]
    
    # Add so that each tag get its own binary columns
    shots = tag_split(shots)
    penalties = tag_split(penalties)
    freekicks = tag_split(freekicks)
    
    # Create dummy variables for each subEvent
    dummy_variables = pd.get_dummies(pbp_df.iloc[shots.index-1].subEventName.replace(" ", "_", regex = True))
    dummy_variables.reset_index(drop=True, inplace=True)
    
    # Combine the shots with dummy variables
    shots = shots.reset_index().merge(dummy_variables, left_index=True, right_index=True)
    penalties = penalties.reset_index().merge(dummy_variables, left_index=True, right_index=True)
    freekicks = freekicks.reset_index().merge(dummy_variables, left_index=True, right_index=True)
    
    # Add indicator variables (custom made)
    shots = add_indicator_variables(shots)
    #penalties = add_indicator_variables(penalties)
    freekicks = add_indicator_variables(freekicks)
    
    # Subset shots that are accurate
    headers = shots[(shots["tag1801"]) & (shots["tag403"])]
    shots = shots[(shots["tag1801"]) & ((shots["tag402"]) | (shots["tag401"]))]
    freekicks = freekicks[freekicks["tag1801"]]
    # For penalties, tag1801 specifies it was accurate which in this context
    # means that it was a goal. Thus we instead look at shots ending up on
    # target (given by tags 1201-1209)
    on_target_columns = penalties.columns
    r = re.compile("tag120.") # Find all columns starting with tag120_
    # Find rows with at least one true
    penalties_on_target = (penalties[list(filter(r.match, on_target_columns))]).any(axis=1)
    penalties = penalties[penalties_on_target]

    return shots, headers, freekicks, penalties


def postXG_model(pbp_df, event_df, event, model, model_vars):
    """ 
    Parameters:
        pbp_df: the play by play dataframe (dataframe)
        event_df: the dataframe with only events of interest (E.g. shots)
        event: the event to calculate postXG for (string)
        model: the sklearn model (sklearn logistic regression model)
        model_vars: the variables to consider (list)
    Returns:
        a data frame with postXG for each goalkeeper from the preceeding event
    """
    # Sanity check
    # pbp_df.iloc[event_df["index"]+1].eventName.value_counts()
    
    save_att_index = pbp_df[pbp_df["eventName"] == "Save attempt"].index
    event_prior = pbp_df.iloc[save_att_index-1]  

    # Split tags
    event_prior = tag_split(event_prior)
    
    # Shots in open play prior to save attempt
    if event == "Shot":
        shots_before_save_att_index = event_prior[(event_prior["subEventName"].isin([f"{event}"])) & 
                                                  ((event_prior["tag401"]) | 
                                                   (event_prior["tag402"]))].index
    elif event == "Header":
        shots_before_save_att_index = event_prior[(event_prior["subEventName"].isin(["Shot"])) & 
                                                  (event_prior["tag403"])].index
    else:
        shots_before_save_att_index = event_prior[event_prior["subEventName"].isin([f"{event}"])].index
        
    shots_before_save_att = event_df[event_df["index"].isin(shots_before_save_att_index)]
    
    # Split tags into multiple columns
    shots_before_save_att = tag_split(shots_before_save_att)
    
    # Add indicator variables for modelling
    shots_before_save_att = add_indicator_variables(shots_before_save_att)
    
    # it does like it should; predict based on time from shot to goalie event
    #print(shots_before_save_att.eventSecDiff.sort_values())
    
    # Skikit learn method
    shots_before_save_att["xG"] = model.predict_proba(shots_before_save_att[model_vars])[:, 1]
    
    ## Find the save attempts which are preceeded by a shot
    save_att = pbp_df.iloc[shots_before_save_att["index"]+1]#.reset_index()
    
    # Reset index
    save_att.reset_index(inplace=True)
    
    # Insert xG into save attempts
    save_att["xG"] = shots_before_save_att.reset_index()["xG"]
    
    return save_att


# Should this one split for each event?
def postXG_sum(xg_df, pbp_df, goalkeepers):
    """ 
    Parameters:
        xg_df: data frame containing postXG values alongside player id
        pbp_df: play by play data frame with all the events
        goalkeepers: data frame with metadata over goalkeepers
    Returns:
        a data frame with a total postXG value, alongside number of goals conceded
        per goalkeeper
    Notes: 
        not available numbers are replaced by zero
    """
    # How many goals each goalkeeper is expected to concede
    expected_goals_conceded = xg_df.groupby("playerId")["xG"].apply(sum)
    
    # Events where goalkeepers are involved
    goalkeeper_events = pbp_df[pbp_df["playerId"].isin(goalkeepers.wyId)]
    
    # Actual goals conceded (mostly correct)
    # Find save attempts
    goals_conceded = goalkeeper_events[(goalkeeper_events["eventName"] == "Save attempt")]
    # Combine save attempts with their respective tags
    goals_conceded = pd.concat([goals_conceded["playerId"].reset_index(drop=True), 
                                pd.json_normalize(goals_conceded.tags)], axis = 1)
    # Goal tags (due to tag id and numerical order)
    goals_conceded["tag"] = [list(i.values())[0] for i in goals_conceded[0]]
    
    # Goals conceded by goalkeeper
    goals_conceded = goals_conceded[goals_conceded["tag"] == 101].groupby("playerId")["tag"].size()
    
    # Combine into one data frame
    postXG = pd.concat([goals_conceded, expected_goals_conceded], axis = 1).reset_index()
        
    return postXG.fillna(0)


def fit_postXG_model(pbp_df, shots, headers, freekicks, penalties, goalkeepers, 
                     players, train_df):
    """ 
    Parameters:
        pbp_df: play by play data frame with all the events to predict on
        shots: data frame of all shots taken with feet not from a set piece
        headers: data frame of all headers (tag403)
        freekicks: data frame of all shots coming directly from a free kick
        penalties: data frame of all shots from the penalty spot
        players: data frame with meta data over players
        train_df: play by play data frame used for training the models
    Returns:
        shotsXG_gk: a data frame with summarized values for postXG from shots
        headersXG_gk: a data frame with summarized values for postXG from free kicks
        freekicksXG_gk: a data frame with summarized values for postXG from penalites
    Notes: 
        no xG model is trained for penalties, since WyScout use a fixed value
        of 0.76 for this xG. Similar values apply for these data too, 
        as the number of penalties scored / total number of penalties on target
        is roughly 0.76.
    """
    
    # Add variables used for creating the model (e.g. distance & angle)
    train_df = add_xg_variables(train_df, players)

    # Add tags as variables and custom indicator variables    
    train_shots, train_headers, train_fks, train_pens = xg_model_prep(train_df)
    
    # Model variables
    model_vars = ["angle", "distance", 
                  "top_corners", "center", #"bottom_corners", "middle_corners",
                  "eventSecDiff", 
                  "pref_foot"]
    model_vars_fk = ["angle", "distance", 
                  "top_corners", "center", #"bottom_corners", "middle_corners",
                  "eventSecDiff"
                  ]
        
    # Fit the models
    shotsModel = LogisticRegression(max_iter = 1000, 
                            penalty = "none",
                            solver = "lbfgs").fit(X = train_shots.loc[:, model_vars], 
                                                  y = train_shots["tag101"])
    
    headersModel = LogisticRegression(max_iter = 1000, 
                            penalty = "none",
                            solver = "lbfgs").fit(X = train_headers.loc[:, model_vars], 
                                                  y = train_headers["tag101"])
    
    #penaltiesModel = LogisticRegression(max_iter = 1000, 
    #                        penalty = "none",
    #                        solver = "lbfgs").fit(X = penalties.loc[:, model_vars], y = penalties["tag101"])
    
    freekicksModel = LogisticRegression(max_iter = 1000, 
                            penalty = "none",
                            solver = "lbfgs").fit(X = train_fks.loc[:, model_vars_fk], 
                                                  y = train_fks["tag101"])
    
    # Calculate postXG for the respective model
    shotsXG = postXG_model(pbp_df, shots, "Shot", shotsModel, model_vars)
    headersXG = postXG_model(pbp_df, headers, "Header", headersModel, model_vars)
    #penaltiesXG = postXG_model(pbp_df, "Penalty", penaltiesModel, model_vars)
    freekicksXG = postXG_model(pbp_df, freekicks, "Free kick shot", freekicksModel, model_vars_fk)
    
    # Summarize postXG for different types of events    
    shotsXG_gk = postXG_sum(shotsXG, pbp_df, goalkeepers)
    headersXG_gk = postXG_sum(headersXG, pbp_df, goalkeepers)
    freekicksXG_gk = postXG_sum(freekicksXG, pbp_df, goalkeepers)

    return shotsXG_gk, headersXG_gk, freekicksXG_gk


def remove_non_goalkeeper_saves(players, df):
    """ 
    Parameters:
        goalkeepers: a data frame with information of own goals conceded by the goalkeeper
        df: data frame with rows to remove (i.e. containing non-goalkeepers)
    Returns:
        a data frame with non goalkeeper rows removed
    """
    # Get index of all goalkeepers
    gk_index = [i for i, row in players.iterrows() if row["role"]["name"] == "Goalkeeper"]
    
    # Get all goalkeepers
    goalkeepers = players.loc[gk_index].reset_index(drop=True)
    
    # Find which rows contain goalkeepers
    rows_to_keep = df.playerId.isin(goalkeepers.wyId)
    rows_to_keep = rows_to_keep[rows_to_keep]
    
    # Return the data frame with only goalkeepers
    return df.loc[rows_to_keep.index]


def postXG_merge(own_goals, pbp_df, players, played_minutes,
                 shotsXG_gk, headersXG_gk, freekicksXG_gk):
    """ 
    Parameters:
        own_goals: a data frame with information of own goals conceded by the goalkeeper
        pbp_df: play by play data frame with all the events
        players: data frame with meta data over the players
        played_minutes: data frame with data containing played minutes
        shotsXG_gk: a data frame with summarized values for postXG from shots
        headersXG_gk: a data frame with summarized values for postXG from headers
        freekicksXG_gk: a data frame with summarized values for postXG from free kicks
    Returns:
        gk_rank: a data frame summarizing goalkeepers in the league and information
                 regarding total goals conceded, total postXG, own goals, 
                 penalties against and penalties allowed
    Notes: 
        penalties receive a fixde value of xG/postXG at 0.76 per WyScout glossary.
    """
    # Merge the different models
    postXG = shotsXG_gk.merge(headersXG_gk, on = ["playerId", "tag"], how = "outer").\
                        merge(freekicksXG_gk, on = ["playerId", "tag"], how = "outer")

    # Sum XG together from the different models
    postXG["xG"] = postXG.iloc[:, 2:].sum(axis = 1)
    
    # Drop unused columns
    postXG.drop(["xG_x", "xG_y"], axis = 1, inplace = True)
    
    # Combine information regarding own goals and goalkeepers     
    postXG = postXG.merge(own_goals.groupby(["goalkeeperId"]).size().rename("ownGoal"), 
                          left_on = "playerId", right_on = "goalkeeperId", how = "outer")
    
    # Rename column more appropriately
    postXG.rename(columns = {"tag": "goal"}, inplace = True)
    
    # Penalties taken
    penalty_index = pbp_df[pbp_df["subEventName"] == "Penalty"].index
    penalty = pbp_df.iloc[penalty_index]
    penalty = tag_split(penalty)
    
    # Fix index if there is a shot after the penalty, so that we only consider one shot
    penalty_index_fix = [i+2 if pbp_df.iloc[i+1].subEventName == "Shot" else i+1 for i in penalty_index]
    penalty.index = penalty_index_fix
    
    # Change some tags to make more sense from the goalkeepers perspective
    post_penalty = pbp_df.iloc[penalty_index_fix]
    post_penalty = tag_split(post_penalty)
    post_penalty["tag1802"] = penalty["tag1801"]
    post_penalty["tag1801"] = penalty["tag1802"]
    
    # Penalty kick goals per keeper, both total against and allowed
    total_pk = post_penalty.groupby("playerId").size().rename("penalties")
    goal_pk = post_penalty.groupby("playerId")["tag1802"].sum().rename("allowedPenalties")
    pk_df = pd.concat([total_pk, goal_pk], axis = 1).reset_index()
    
    # Merge postXG with information regarding penalties
    postXG = postXG.merge(pk_df, how = "outer")
    
    # Change NaN values to 0
    postXG.fillna(0, inplace = True)
    
    # Include penalties in xG column by specified WyScout value
    postXG["xG"] = postXG["xG"] + 0.76 * postXG["penalties"]
    postXG["diff"] = postXG["xG"] - (postXG["goal"] - postXG["ownGoal"])
    
    # Add player names
    league_goalies = players.loc[players["wyId"].isin(postXG.playerId), ["wyId", "shortName"]]

    # Create a data frame for ranking
    gk_rank = postXG.merge(league_goalies, left_on = "playerId", right_on = "wyId")
    
    # Remove duplicate column
    gk_rank.drop("wyId", axis = 1, inplace = True)
    
    # Remove non goalkeepers
    gk_rank = remove_non_goalkeeper_saves(players, gk_rank)
    
    # Add percentile rank
    gk_rank["percentile_rank"] = gk_rank["diff"].rank(pct = True)
    
    # Add minutes & games played
    play_time = played_minutes[["minutesPlayed", "gamesPlayed"]].reset_index()
    play_time.rename(columns = {"index": "playerId"}, inplace = True)
    gk_rank = gk_rank.merge(play_time, left_on = "playerId", right_on = "playerId")
    
    return gk_rank


def save_percentage(pbp_df, players, playerDF, goalkeepers, min_minutes_played = 1):
    """ 
    Parameters:
        pbp_df: play by play data frame with all the events
        players: data frame with metadata of all players
        playerDF: data frame over games and minutes played overall and per player
                  and match
        goalkeepers: data frame with metadata of all goalkeepers (i.e from players.json)
        min_minutes_played: how many minutes the player is required to have played
                            to be included
    Returns:
        save_attempts: a data frame with information over save percentages, 
                       both overall and per reflexes and save attempts
    Notes: 
        min_minutes_played is used as a factor to ensure percentages are not 
        inflated for players with few minutes played
    """

    # Events including goalkeepers
    goalkeepers_event = pbp_df[pbp_df["playerId"].isin(goalkeepers.wyId)]
    goalkeepers_event = tag_split(goalkeepers_event)
    
    # Filter out keepers who have not played enough minutes
    freq_gks = playerDF[playerDF["minutesPlayed"] >= min_minutes_played]
    
    # Find goalkeepers among the events who played enough minutes
    gks_of_interest = goalkeepers_event[goalkeepers_event["playerId"].isin(freq_gks.index)]
    
    # Calculate amount of successful and unsuccessful save attempts
    save_attempts = gks_of_interest[gks_of_interest["eventName"] == \
                      "Save attempt"].groupby(["playerId", "subEventName"], as_index=False)\
                        [["tag1801", "tag1802"]].sum()
                        
    # Get the total proportion of accurate for both reflexes and save attempt
    save_attempts_total_prop = save_attempts.groupby("playerId", as_index = False).sum()
    save_attempts_total_prop["totalPropAccurate"] = save_attempts_total_prop["tag1801"] / \
        (save_attempts_total_prop["tag1801"] + save_attempts_total_prop["tag1802"])
    
    # Get the accurace per subevent
    save_attempts["propAccurate"] = save_attempts["tag1801"] / \
        (save_attempts["tag1801"] + save_attempts["tag1802"])
    
    # Combine into one data frame with both subEvent and event accuracy
    save_attempts = save_attempts.merge(save_attempts_total_prop[["playerId", "totalPropAccurate"]],
                        on='playerId', how='inner')
    
    # Add player names
    league_goalies = goalkeepers.loc[goalkeepers["wyId"].isin(save_attempts.playerId), ["wyId", "shortName"]]

    # Create a data frame for ranking
    save_attempts = save_attempts.merge(league_goalies, left_on = "playerId", right_on = "wyId")
    
    # Remove duplicate column
    save_attempts.drop("wyId", axis = 1, inplace = True)
    
    # Remove non goalkeepers
    save_attempts = remove_non_goalkeeper_saves(players, save_attempts)
    
    # Add percentile rank
    save_attempts["percentile_rank_overall"] = save_attempts["totalPropAccurate"].rank(pct = True)
    
    # Note that these two percentiles ranking (reflexes and save attempts)
    # are stored in the same column. To get actual ranking, subset one of them.
    save_attempts["percentile_rank_save_type"] = save_attempts.groupby(["subEventName"])["propAccurate"].rank(pct = True)
    
    return save_attempts
        

def pass_success(pbp_df, goalkeepers, min_passes = 1):
    """ 
    Parameters:
        goalkeepers: data frame events including goalkeepers
        min_passes: how many passes a player has to have performed
    Returns:
        passes: a data frame with information over pass success
    Notes: 
        min_passes is used as a factor to ensure percentages are not 
        inflated for players with few minutes played
    """
    
    
    # Subset events with only goalkeepers
    goalkeeper_events = pbp_df[pbp_df["playerId"].isin(goalkeepers.wyId)]

    # Add tag columns
    goalkeeper_events = tag_split(goalkeeper_events)
    
    # Find passes of goalkeepers
    passes = goalkeeper_events[goalkeeper_events["eventName"] == "Pass"].\
        groupby(["playerId", "subEventName", "tag1801"], as_index=False)["subEventName"].size()
    
    passes["propAccurate"] = passes.groupby(["playerId", "subEventName"])["size"].apply(lambda x: x/sum(x))
    
    # Only players with at least 10 passes of a given pass type
    passes = passes[passes["size"] >= min_passes]
    
    # Add player names
    league_goalies = goalkeepers.loc[goalkeepers["wyId"].isin(passes.playerId), ["wyId", "shortName"]]

    # Create a data frame for ranking
    passes = passes.merge(league_goalies, left_on = "playerId", right_on = "wyId")
    
    # Remove duplicate column
    passes.drop("wyId", axis = 1, inplace = True)
    
    # Remove non goalkeepers
    passes = remove_non_goalkeeper_saves(goalkeepers, passes)
    
    # Add percentile rank
    passes["percentile_rank"] = passes.groupby(["subEventName"])["propAccurate"].rank(pct = True)
        
    return passes


def read_meta_data(path = "Wyscout"):
    """ 
    Parameters:
        path: local path to the Wyscout folder
    Returns:
        players: data frame with meta data over all players (including goalkeepers)
        goalkeepers: data frame with meta data over all goalkeepers
    """
    # Load information of players
    with open(f'{path}/players.json', encoding='utf-8') as f:
        temp = json.load(f)
    
    # Store all players in data frame
    players = pd.DataFrame(temp)
    
    # Fix unicode characters
    players["firstName"] = [i.encode("utf-8").decode('unicode_escape') for i in players["firstName"]]
    players["lastName"]  = [i.encode("utf-8").decode('unicode_escape') for i in players["lastName"]]
    players["shortName"] = [i.encode("utf-8").decode('unicode_escape') for i in players["shortName"]]

    # Get index of all goalkeepers
    gk_index = [i for i, row in players.iterrows() if row["role"]["name"] == "Goalkeeper"]
    
    # Get all goalkeepers
    goalkeepers = players.loc[gk_index].reset_index(drop=True)
        
    return players, goalkeepers


def read_match_data(path = "Wyscout"):
    """ 
    Parameters:
        path: local path to the Wyscout folder
    Returns:
        matches_England: meta data over all matches played in England during 2017-2018 
        matches_France: meta data over all matches played in France during 2017-2018 
        matches_Germany: meta data over all matches played in Germany during 2017-2018 
        matches_Spain: meta data over all matches played in Spain during 2017-2018 
        matches_Italy: meta data over all matches played in Italy during 2017-2018 
    """
    # List of all nations
    nation_list = ["England", "France", "Germany", "Spain", "Italy"]
    match_list = {}
    # Load information of players & matches in a given nation
    for nation in nation_list:           
        with open(f'{path}/matches/matches_{nation}.json', encoding='utf-8') as f:
            match_list[f"{nation}"]  = json.load(f)
  
    # Matches
    matches_England = pd.DataFrame(match_list["England"])
    matches_France = pd.DataFrame(match_list["France"])
    matches_Germany = pd.DataFrame(match_list["Germany"])
    matches_Spain = pd.DataFrame(match_list["Spain"])
    matches_Italy = pd.DataFrame(match_list["Italy"])
    
    del match_list
    
    return matches_England, matches_France, \
        matches_Germany, matches_Spain, matches_Italy


def read_event_data(path = "Wyscout"):
    """ 
    Parameters:
        path: local path to the Wyscout folder
    Returns:
        England: data frame over all events during games in England during 2017-2018 
        France: data frame over all events during games in France during 2017-2018 
        Germany: data frame over all events during games in Germany during 2017-2018 
        Spain: data frame over all events during games in Spain during 2017-2018 
        Italy: data frame over all events during games in Italy during 2017-2018 
    """
    # List of all nations
    nation_list = ["England", "France", "Germany", "Spain", "Italy"]
    data_list = {}
    # Load information of events in a given nation
    # Not very fast mind you...
    for nation in nation_list:
        with open(f'{path}/events/events_{nation}.json', encoding='utf-8') as f:
            data_list[f"{nation}"] = json.load(f)
    
    # Retrieve pbp events for all leagues
    England = pd.DataFrame(data_list["England"])
    France = pd.DataFrame(data_list["France"])
    Germany = pd.DataFrame(data_list["Germany"])
    Spain = pd.DataFrame(data_list["Spain"])
    Italy = pd.DataFrame(data_list["Italy"])
    
    # Remove list of data to save some memory
    del data_list
    
    return England, France, Germany, Spain, Italy


def create_metrics(pbp_df, matches_nation, train_df):
    """ 
    Parameters:
        pbp_df: data frame with all events in a given nation
        matches_nation: data frame with all matches from the given nation
        train_df: data frame with events that will be used for training the models
    Returns:
        gk_rank: information and ranking of goalkeepers for PSxG-GA
        save_percent: success rate for different types of save attempts for goalkeepers
        pass_percent: success rate for different types of passes for goalkeepers
    """
    # Get information over players and goalies
    players, goalkeepers = read_meta_data()
    
    # Add variables used for creating the model (e.g. distance & angle)
    df = add_xg_variables(pbp_df, players)

    # Add tags as variables and custom indicator variables
    shots, headers, freekicks, penalties = xg_model_prep(df)
    
    # Fit the model itself
    shotsXG_gk, headersXG_gk, freekicksXG_gk = fit_postXG_model(df, shots, headers, 
                                                                freekicks, penalties, 
                                                                goalkeepers, players, 
                                                                train_df)
    
    # Calculate minutes playeed in the current nation
    playerDF = played_minutes(matches_nation)
    
    # Get information over own goals
    own_goals_df = own_goals(df, playerDF, goalkeepers)
    
    # Create a data frame with information over postXG and other keeper statistics
    gk_rank = postXG_merge(own_goals_df, df, players, playerDF,
                           shotsXG_gk, headersXG_gk, freekicksXG_gk)
    
    # Information over save percentage
    save_percent = save_percentage(pbp_df, players, playerDF, goalkeepers)
    
    # Information over pass success
    pass_percent = pass_success(pbp_df, goalkeepers)
    
    return gk_rank, save_percent, pass_percent
    
def main():
    """
    Parameters:
        None.
    Returns:
        England_dict: dictionary with keys (gk_rank), save (save_percent), 
                      pass (pass_percent) for English league
        France_dict: dictionary with keys (gk_rank), save (save_percent), 
                     pass (pass_percent) for French league
        Germany_dict: dictionary with keys (gk_rank), save (save_percent), 
                      pass (pass_percent) for German league
        Spain_dict: dictionary with keys (gk_rank), save (save_percent), 
                    pass (pass_percent) for Spanish league
        Italy_dict: dictionary with keys (gk_rank), save (save_percent), 
                    pass (pass_percent) for Italian league
    """
    # Read data over events
    England, France, Germany, Spain, Italy = read_event_data()
    print("Event data read")
    
    # Read data over matches
    matches_England, matches_France, matches_Germany, matches_Spain, matches_Italy = read_match_data()
    print("Match data read")
    
    # Create metrics for all nations
    gk_England, save_percent_England, pass_England = create_metrics(England, matches_England, England)
    print("English data done")
    #del England
    
    gk_France, save_percent_France, pass_France = create_metrics(France, matches_France, England)
    print("French data done")
    #del France
    
    gk_Germany, save_percent_Germany, pass_Germany = create_metrics(Germany, matches_Germany, England)
    print("German data done")
    #del Germany
    
    gk_Spain, save_percent_Spain, pass_Spain = create_metrics(Spain, matches_Spain, England)
    print("Spanish data done")
    #del Spain
    
    gk_Italy, save_percent_Italy, pass_Italy = create_metrics(Italy, matches_Italy, England)
    print("Italian data done")
    #del Italy

    # Store all results in a dictionary
    England_dict = {"gk": gk_England, "save": save_percent_England, "pass": pass_England}
    France_dict = {"gk": gk_France, "save": save_percent_France, "pass": pass_France}
    Germany_dict = {"gk": gk_Germany, "save": save_percent_Germany, "pass": pass_Germany}
    Spain_dict = {"gk": gk_Spain, "save": save_percent_Spain, "pass": pass_Spain}
    Italy_dict = {"gk": gk_Italy, "save": save_percent_Italy, "pass": pass_Italy}
    
    return England_dict, France_dict, Germany_dict, Spain_dict, Italy_dict

if __name__ == '__main__':        
    # Run the program
    England, France, Germany, Spain, Italy = main()
    
    # Combine into one data frame for all nations
    gk_ranks = pd.concat([England["gk"].assign(nation = "England"), 
                          France["gk"].assign(nation = "France"),
                          Germany["gk"].assign(nation = "Germany"),
                          Spain["gk"].assign(nation = "Spain"),
                          Italy["gk"].assign(nation = "Italy")])
    
    save_ranks = pd.concat([England["save"].assign(nation = "England"), 
                            France["save"].assign(nation = "France"),
                            Germany["save"].assign(nation = "Germany"),
                            Spain["save"].assign(nation = "Spain"),
                            Italy["save"].assign(nation = "Italy")])
    
    pass_ranks = pd.concat([England["pass"].assign(nation = "England"), 
                              France["pass"].assign(nation = "France"),
                              Germany["pass"].assign(nation = "Germany"),
                              Spain["pass"].assign(nation = "Spain"),
                              Italy["pass"].assign(nation = "Italy")])
    
    # Add overall percentile rank
    gk_ranks["total_percentile"] = gk_ranks["diff"].rank(pct = True)
    save_ranks["total_percentile"] = save_ranks.groupby(["subEventName"])["propAccurate"].rank(pct = True)
    pass_ranks["total_percentile"] = pass_ranks.groupby(["subEventName"])["propAccurate"].rank(pct = True)
    
    # Write to csv's
    #gk_ranks.to_csv("gkRank.csv", index=False, encoding = "utf-8-sig")
    #pass_ranks.to_csv("pass.csv", index=False, encoding = "utf-8-sig")
    #save_ranks.to_csv("save.csv", index=False, encoding = "utf-8-sig")