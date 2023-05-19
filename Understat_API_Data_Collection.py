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
    if CreateNew:
        print('Création de nouveaux objets de données')
        with UnderstatClient() as understat:
            print('Tentative de collecte des données API...')
            league_team_data = understat.league(league=league).get_team_data(season="2022")
            print('Données API collectées avec succès !')
        data_dict = {}
        prem_team_ids = {}

        for k, v in league_team_data.items():
            team_name = league_team_data[k]['title']

            team_id = k
            prem_team_ids[team_name] = team_id
            data_dict['updatedate'] = pd.Timestamp.today().date()
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

        print('Nouveaux objets Pickle (vides) créés par l\'utilisateur')
        pickle.dump(data_dict, open(league+'DataDict.p', 'wb'))
        pickle.dump(prem_team_ids, open(league+'TeamIDs.p', 'wb'))

    else:
        print('Chargement des Pickles (objets de données existants des statistiques de football)')
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


def stat_creator(league):

    data_dict, prem_team_ids = generate_data_dict_and_team_ID_dict(CreateNew=False,league=league) # Cela charge dans les objets pickle.

    if data_dict['updatedate'] < pd.Timestamp.today().date():
        print('Mise à jour du dictionnaire de données avec de nouvelles statistiques ! Dernière mise à jour = {}'.format(data_dict))
        data_dict['updatedate'] = pd.Timestamp.today().date()
        for team in list(prem_team_ids.keys()): # This gets match data
            with UnderstatClient() as understat:
                print('Tentative de collecte des données d\'API pour {}...'.format(team))
                team_match_data = understat.team(team=team).get_match_data(season="2022")
                print('Données d\'API collectées avec succès')

            i=0
            for match in team_match_data: # tableaux pour les matchs qui ont été joués

                if i <= data_dict[team]['gamesPlayed'] - 1:
                    i+=1
                    continue

                if match['isResult']:

                    home_name = match['h']['title']
                    away_name = match['a']['title']

                    if int(match['id']) < data_dict[team]['played_matches'][data_dict[team]['gamesPlayed'] - 1]: # ie. if the match that has just been played in the fixtures just gone was prev. a postponed game, then we need to insert the stats at the end of the array rather than a prev entry - the end of the array index will be equal to the number of games played, due to 0 indexing arrays ...
                        idx_to_update = data_dict[team]['gamesPlayed']
                        data_dict[team]['played_matches'][idx_to_update] = int(match['id'])
                        update_data_dict(team, match, home_name, away_name, data_dict, idx_to_update)

                    else:
                        data_dict[team]['played_matches'][i] = int(match['id'])
                        update_data_dict(team, match, home_name, away_name, data_dict, i)

                    i += 1
                    #print(i)

                else: # si la correspondance n'est pas un résultat - ne pas incrémenter i
                    continue

        print('Statistiques mises à jour générées - Enregistrement en cours...')
        pickle.dump(data_dict, open(league+'DataDict.p', 'wb'))
    else:
        print('Pas besoin de mettre à jour les statistiques à nouveau, car la mise à jour a déjà été effectuée aujourd\'hui.')

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

            if team_data_dict[coming_opponent]['cGA'][i-1] == 0 or team_data_dict[opponent_game_i]['cGA'][i-1] == 0:
                weighting_factor = 1
            else:
                weighting_factor = team_data_dict[coming_opponent]['cGA'][i-1]/team_data_dict[opponent_game_i]['cGA'][i-1] # The ratio of the team in questions next opponent (from today) cGA for game i-1 to the opponent in game i cGA in game i-1

                #weighting_factor = team_data_dict[coming_opponent]['cGA'][i] / team_data_dict[opponent_game_i]['cGA'][i]

            # Appliquer une transformation pour rendre les pondérations réalistes (par exemple, un multiplicateur de 2,5 est (très probablement) trop grand)
            transformation = lambda x : (1/(1+np.exp(-4*(x-1))) + 0.5) if (x-1) < 0 else np.tanh(0.8*(x-1))+1
            transformed_weight = transformation(weighting_factor)

            wtd_goal += transformed_weight * team_data_dict[team]['{}F'.format(goal_type)][i] # multiply the weighting factor by the number of goals scored by the team in question against the opponent in game i
            wtd_goal_series[team].append(wtd_goal)

        if team == teams[0]:
            avg_wtd_goal_HomeTeam = wtd_goal / games_lookback
        else:
            avg_wtd_goal_AwayTeam = wtd_goal / games_lookback
        print(team)
    if avg_wtd_goal_HomeTeam == 0.0 or avg_wtd_goal_AwayTeam == 0.0:
        raise ValueError('Les buts moyens pondérés n\'ont pas été calculés correctement pour les équipes en question/' + teams[0]+'/'+teams[1]+'/'+str(games_lookback)+'/'+str(games_played)+'/'+str(wtd_goal))

    return avg_wtd_goal_HomeTeam, avg_wtd_goal_AwayTeam, wtd_goal_series


def get_goal_covariance(wtd_goal_series, teams):
    # Utilise la série d'objectifs pondérés pour obtenir la covariance entre les séries
    home_wtd_goal_series = wtd_goal_series[teams[0]]
    away_wtd_goal_series = wtd_goal_series[teams[1]]

    if len(home_wtd_goal_series) == 1:
        l3 = float('NaN')
    else:
        C = np.cov(home_wtd_goal_series, away_wtd_goal_series)
        l3 = C[0][1] # Le paramètre de covariance dans la distribution Poisson Biv
    return l3

