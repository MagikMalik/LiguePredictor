import pandas as pd
from understatapi import UnderstatClient
import numpy as np
import pickle


def cumulative_goal_array(i, team_data_dict, team, goals_for, goals_against, xGA, xGF):
    team_data_dict[team]['cGA'][i] = int(goals_against) + int(team_data_dict[team]['cGA'][i - 1]) if i != 0 else int(goals_against)
    team_data_dict[team]['cGF'][i] = int(goals_for) + int(team_data_dict[team]['cGF'][i - 1]) if i !=0 else int(goals_for)

    team_data_dict[team]['cxGA'][i] = float(xGA) + float(team_data_dict[team]['cxGA'][i - 1]) if i != 0 else float(xGA)
    team_data_dict[team]['cxGF'][i] = float(xGF) + float(team_data_dict[team]['cxGF'][i - 1]) if i != 0 else float(xGF)

    return team_data_dict


def generate_data_dict_and_team_ID_dict(CreateNew,league):
    if CreateNew: # Create new prem league team data, and a new empty data dictionary - set CreateNew to True at the start of each season
        print('Creating New Data Objects')
        with UnderstatClient() as understat:
            print('Attempting to Collect API Data...')
            league_team_data = understat.league(league=league).get_team_data(season="2022")
            print('Collected API Data Successfully!')
        data_dict = {}
        prem_team_ids = {}

        for k, v in league_team_data.items():
            team_name = league_team_data[k]['title']

            team_id = k
            prem_team_ids[team_name] = team_id
            data_dict[team_name] = {'id': team_id,
                                     'gamesPlayed': 0,
                                     'opponent': [float('NaN')] * 38,
                                     'GA': [0.0] * 38,
                                     'GF': [0.0] * 38,
                                     'xGA': [0.0] * 38,
                                     'xGF': [0.0] * 38,
                                     'cGA': [0.0] * 38,
                                     'cGF': [0.0] * 38,
                                     'cxGA': [0.0] * 38,
                                     'cxGF': [0.0] * 38,
                                    'played_matches':[float('NaN')] * 38}

        print('User Created New (Empty) Pickle Objects')
        pickle.dump(data_dict, open(league+'DataDict.p', 'wb'))
        pickle.dump(prem_team_ids, open(league+'TeamIDs.p', 'wb'))

    else:
        print('Loading Pickles (Existing Data Objects of Football Stats)')
        prem_team_ids = pickle.load(open(league+"TeamIDs.p", "rb"))
        data_dict = pickle.load(open(league+"DataDict.p", "rb"))

    return data_dict, prem_team_ids


def update_data_dict(team, match, home_name, away_name, data_dict, i):
    if team == home_name:
        GF = match['goals']['h']
        GA = match['goals']['a']
        xGF = match['xG']['h']
        xGA = match['xG']['a']

    if team == away_name:
        GF = match['goals']['a']
        GA = match['goals']['h']
        xGF = match['xG']['a']
        xGA = match['xG']['h']

    data_dict[team]['GA'][i] = int(GA)
    data_dict[team]['GF'][i] = int(GF)
    data_dict[team]['xGA'][i] = float(xGA)
    data_dict[team]['xGF'][i] = float(xGF)
    data_dict[team]['opponent'][i] = home_name

    data_dict = cumulative_goal_array(i, data_dict, team, GF, GA, xGA, xGF)

    data_dict[team]['gamesPlayed'] += 1


def stat_creator(most_recent_update,league):

    data_dict, prem_team_ids = generate_data_dict_and_team_ID_dict(CreateNew=False,league=league) # This loads in the pickle objects. Sequentially update them with just the new GWs data

    # we only want to collect match data from the API once per GW.
    # In a GW, we run the programme multiple times to get predictions for different games
    # So, need a flag to tell us when the last time we ran

    print(most_recent_update)

    if most_recent_update.date() < pd.Timestamp.today().date():
        print('Updating the Data Dictionary With New Stats! Last Update = {}'.format(most_recent_update))

        for team in list(prem_team_ids.keys()): # This gets match data
            with UnderstatClient() as understat:
                print('Attempting to Collect API Data For {}...'.format(team))
                team_match_data = understat.team(team=team).get_match_data(season="2022")
                print('Collected API Data Successfully!')

            i=0
            for match in team_match_data: # Goes through the matches, and builds out cum. arrays for matches that have been played

                if i <= data_dict[team]['gamesPlayed'] - 1:
                    i+=1
                    continue

                if match['isResult']:

                    home_name = match['h']['title']
                    away_name = match['a']['title']

                    # if, when cycling through matches, a match j games ago (ie a previously postponed game) which previously had isResult as False, now has the isResult parameter as True,
                    # it means that we have played this postponed game in the GW just gone. So, we check to see if the ID is less than the ID of the most recent
                    # game before this (as the postponed game was earlier in the season, it will have a lower ID value)
                    # If this is the case, then we need to put in the (x)GA/F stats into the right part of the cumulative arrays, so we define the idx_to_update parameter
                    # as, which will be the last entry of the array.

                    # We only expect to encounter this issue when the game being played this current GW is a game which was previously postponed,
                    # because, if we then go to next week, the array will already have been filled & pickled, which we load in.

                    if int(match['id']) < data_dict[team]['played_matches'][data_dict[team]['gamesPlayed'] - 1]: # ie. if the match that has just been played in the fixtures just gone was prev. a postponed game, then we need to insert the stats at the end of the array rather than a prev entry - the end of the array index will be equal to the number of games played, due to 0 indexing arrays ...
                        idx_to_update = data_dict[team]['gamesPlayed']
                        data_dict[team]['played_matches'][idx_to_update] = int(match['id'])
                        update_data_dict(team, match, home_name, away_name, data_dict, idx_to_update)

                    else:
                        data_dict[team]['played_matches'][i] = int(match['id'])
                        update_data_dict(team, match, home_name, away_name, data_dict, i)

                    i += 1
                    #print(i)

                else: # if match is not result - do not add one to i
                    continue

        print('Generated Updated Stats - Now Saving ...')
        pickle.dump(data_dict, open('TeamDataDict.p', 'wb'))
        print('Pickled & Saved UPDATED Team Data Dict Succesfully')
    else:
        print('No Need To Update Stats Again, as Already Ran The Update Today.')

    return data_dict


def get_weighted_goals(games_lookback, teams, team_data_dict, Use_xG):
    avg_wtd_goal_HomeTeam = 0.0
    avg_wtd_goal_AwayTeam= 0.0

    wtd_goal_series = {teams[0]: [], teams[1]: []}

    for team in teams:
        games_played = team_data_dict[team]['gamesPlayed']
        print('{} games played {}'.format(team, games_played))

        coming_opponent = [teams[0] if opponent != team else teams[1] for opponent in teams][0]

        #print('coming opp = {}'.format(coming_opponent))

        wtd_goal= 0
        if Use_xG == 'True':
            print('we are using xG')
            goal_type = 'xG'
        else:
            goal_type = 'G'

        for i in range(games_played - games_lookback, games_played):
            opponent_game_i = team_data_dict[team]['opponent'][i]
            #print('Opponent Game i = {}'.format(opponent_game_i))

            # Weighting Factor gives you a weight calculated as goals our next opponent have concede up to game i / goals the team we played in game i have conceded up to game i.
            # This weighting then is multiplied by the xG or goals scored by the team of interest in game i.
            # In this way the weighting scales the xG or goals by how much our next opppnent concedes by ie. a proxy for how good the defense is (and also how poor the attack of our team was on the day too)
            # For, if we just used Goals or xG as input to a model then, obviously, scoring 3 Vs Man City for example is different to scoring 3 Vs Bournemouth, so the weighting attempts to scale average goal for opponent difficulty
            # In this example, the 3 scored againt Man City will give you a bigger weighting than the 3 scored Vs bournmeouth  as this is a harder feat
            # and so may reflect a good teams form at that moment in time, likely to bring into the next game

            if team_data_dict[coming_opponent]['cGA'][i-1] == 0 or team_data_dict[opponent_game_i]['cGA'][i-1] == 0:
                weighting_factor = 1
            else:
                weighting_factor = team_data_dict[coming_opponent]['cGA'][i-1]/team_data_dict[opponent_game_i]['cGA'][i-1] # The ratio of the team in questions next opponent (from today) cGA for game i-1 to the opponent in game i cGA in game i-1

                #weighting_factor = team_data_dict[coming_opponent]['cGA'][i] / team_data_dict[opponent_game_i]['cGA'][i]

            # Apply a transformation to make the weightings realistic (eg. a 2.5 multiplier is (most likely) too large)
            transformation = lambda x : (1/(1+np.exp(-4*(x-1))) + 0.5) if (x-1) < 0 else np.tanh(0.8*(x-1))+1
            transformed_weight = transformation(weighting_factor)

            # The weighting factor is based of the prev GW's stats. We then multiply this weighting factor by the goals scored in
            # the current GWs to 'scale' for strength-adjusted goals.

            wtd_goal += transformed_weight * team_data_dict[team]['{}F'.format(goal_type)][i] # multiply the weighting factor by the number of goals scored by the team in question against the opponent in game i
            wtd_goal_series[team].append(wtd_goal)

        if team == teams[0]:
            avg_wtd_goal_HomeTeam = wtd_goal / games_lookback
        else:
            avg_wtd_goal_AwayTeam = wtd_goal / games_lookback
        print(team)
    if avg_wtd_goal_HomeTeam == 0.0 or avg_wtd_goal_AwayTeam == 0.0:
        raise ValueError('The Weighted Average Goals Were Not Computed Correctly For the Teams in Question')

    return avg_wtd_goal_HomeTeam, avg_wtd_goal_AwayTeam, wtd_goal_series


def get_goal_covariance(wtd_goal_series, teams):
    # Use the weighted goal series to get the covariance between the series !
    home_wtd_goal_series = wtd_goal_series[teams[0]]
    away_wtd_goal_series = wtd_goal_series[teams[1]]

    if len(home_wtd_goal_series) == 1:
        l3 = float('NaN')
    else:
        C = np.cov(home_wtd_goal_series, away_wtd_goal_series)
        l3 = C[0][1] # The covariance parameter in the Biv Po distr
    return l3

