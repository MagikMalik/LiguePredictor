import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Understat_API_Data_Collection import *
from math import comb
from random import choices


def BivariatePoissonProb(x, y, l1, l2, l3):
    summation_term = 0
    for i in range(min(x,y) + 1):
        summation_term += comb(x, i)*comb(y, i)*np.math.factorial(i)*(( l3 / l1*l2 )**i)

    prob = float(np.exp(-(l1+l2+l3)) * (l1**x)/np.math.factorial(x) * (l2**y)/np.math.factorial(y) * summation_term)
    return prob


# function to generate the probability distribution for home_Gf and away_Gf (and l3), then sample from it to get integers
def GenerateProbDistr(l1, l2, l3):
    # x, y can be thought of as a co-ordinate set of all possible score combinations for a match
    x = np.linspace(0,6,7)
    y = np.linspace(0,6,7)

    score_prob_dict = {}
    for i in x:
        for j in y:
            home_score = int(i)
            away_score = int(j)
            score = (home_score, away_score)
            score_prob_dict[score] = BivariatePoissonProb(home_score, away_score, l1, l2, l3)

    return score_prob_dict, x, y


def buildScoreMatrix(MC_score_tracker, teams, x, y):

    home_iterables = [[teams[0]], x]
    away_iterables = [[teams[1]], y]

    home_multidx = pd.MultiIndex.from_product(home_iterables)
    away_multidx = pd.MultiIndex.from_product(away_iterables)

    score_matrix = pd.DataFrame(columns=away_multidx, index=home_multidx)
    for score, frequency in MC_score_tracker.items():
        score_matrix.loc[(teams[0], score[0])][(teams[1], score[1])] = round(float((frequency / np.sum(list(MC_score_tracker.values())))*100), 3)

    print(score_matrix)
    SM = score_matrix.droplevel(level=0, axis=0)
    SM = SM.droplevel(level=0, axis=1)

    return score_matrix

def print_results(func, teams):
    home_win_prob, away_win_prob, draw_prob, MC_score_tracker, x, y = func

    ML_score_dict = {}

    print('\n \n')

    score_matrix = buildScoreMatrix(MC_score_tracker, teams, x, y)

    print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-')
    print('{} Win = {} % \n{} Win = {} % \nDraw = {} %'.format(teams[0], round(home_win_prob, 3),
                                                               teams[1], round(away_win_prob, 3),
                                                               round(draw_prob, 3)))
    print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-')
    print('Scorelines with highest chance:')

    remove_max_score = lambda x, max_likelihood_score: x.pop(max_likelihood_score)
    max_likelihood_score = lambda x: max(x, key=x.get)
    max_likelihood_percent = lambda x: max(x.values())

    ML_score = max_likelihood_score(MC_score_tracker)
    ML_percent = max_likelihood_percent(MC_score_tracker)
    ML_score_dict[ML_score] = ML_percent

    print('1/ {} with {}%'.format(ML_score, ML_percent/10000))

    remove_max_score(MC_score_tracker, ML_score)
    ML_score = max_likelihood_score(MC_score_tracker)
    ML_percent = max_likelihood_percent(MC_score_tracker)
    ML_score_dict[ML_score] = ML_percent

    print('2/ {} with {} %'.format(ML_score, ML_percent/10000))

    remove_max_score(MC_score_tracker, ML_score)
    ML_score = max_likelihood_score(MC_score_tracker)
    ML_percent = max_likelihood_percent(MC_score_tracker)
    ML_score_dict[ML_score] = ML_percent

    print('3/ {} with {} %'.format(ML_score, ML_percent/10000))

    print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-')
    print('Odds:')
    print('{} Win = {}:1 \n{} Win = {}:1 \nDraw = {}:1'.format(teams[0], round(100 / home_win_prob - 1, 2),
                                                               teams[1], round(100 / away_win_prob - 1, 2),
                                                               round(100 / draw_prob - 1, 2)))
    return score_matrix, ML_score_dict

def MonteCarloMatchSim(teams, iterations, GamesLookback, BaseOnxG,league):

    # Loads in the last time we updated the stats.
    most_recent_run = pickle.load(open("MostRecentRun.p", "rb"))

    teams_data_dict = stat_creator(most_recent_run,league)

    avg_weighted_goals_HomeTeam, avg_weighted_goals_AwayTeam, wtd_goal_series = get_weighted_goals(GamesLookback, teams,
                                                                                                   teams_data_dict,
                                                                                                    Use_xG=BaseOnxG)

    goal_covariance = get_goal_covariance(wtd_goal_series, teams)
    l3 = goal_covariance
    print('cov BEFORE condition: {}'.format(l3))

    l3 = 0.1 #if (l3 > 0.3 or l3 == float('NaN')) else l3

    print('\n')
    print('Got Weighted Goals... Now Running Monte Carlo Simulation')
    print('\n')
    print('{} avg weighted goals = {}'.format(teams[0], avg_weighted_goals_HomeTeam))
    print('{} avg weighted goals = {}'.format(teams[1], avg_weighted_goals_AwayTeam))
    print('goal cov = {}'.format(l3))
    print('\n')

    score_prob_dict, x, y = GenerateProbDistr(avg_weighted_goals_HomeTeam, avg_weighted_goals_AwayTeam, l3)
    MC_score_tracker = {k:v for k,v in zip(list(score_prob_dict.keys()), [0]*len(score_prob_dict))}
    MC_win_tracker = {'HW': 0, 'AW': 0, 'D':0}
    for i in range(iterations):
        if ( i / iterations ) % 0.2 == 0:
            print('Iteration: {} / {}. Completion: {}%'.format(i, iterations, (i/iterations) * 100))

        sampled_score = choices(list(score_prob_dict.keys()), weights=list(score_prob_dict.values()))[0]
        MC_score_tracker[sampled_score] += 1
        if sampled_score[0] > sampled_score[1]:
            MC_win_tracker['HW'] += 1
        if sampled_score[0] < sampled_score[1]:
            MC_win_tracker['AW'] += 1
        else:
            MC_win_tracker['D'] += 1
    print('Iteration: {} / {}. Completion: 100%'.format(iterations, iterations))

    home_win_prob = (MC_win_tracker['HW'] / iterations) * 100
    away_win_prob = (MC_win_tracker['AW'] / iterations) * 100
    draw_prob = 100 - home_win_prob - away_win_prob

    # Pickle the date that we ran it.
    most_recent_run = pd.Timestamp.today()
    pickle.dump(most_recent_run, open('MostRecentRun.p', 'wb'))

    return home_win_prob, away_win_prob, draw_prob, MC_score_tracker, x, y, avg_weighted_goals_HomeTeam, avg_weighted_goals_AwayTeam

    #present scores in a nice matrix of probabilities
    #convert to odds

#home_win_prob, away_win_prob, draw_prob = MonteCarloMatchSim(['Manchester City', 'Tottenham'], 100000, GamesLookback=3, BaseOnxG=False)




# Rationale: weighting tan function to reflect linear relationship for weights above (and below 1) for a small period then tail off
# either end, as having a very high weighting is unrealistic.
# And, for weightings larger than 1, this reduces faster/slower than weightings less than 1 because of xyz..

# Had to find an API to extract the data and then store it in the way which I wanted


def master():
    homeTeam = input('Please Enter HOME Team: ')
    awayTeam = input('Please Enter AWAY Team: ')
    teams = [homeTeam, awayTeam]

    glb = input('How Many Games Do You Wish To Look-Back to Calculate Goal Rate?:')
    use_xg = input('Enter True if You Wish To Forecast Using xG, False if Real Goals Scored: ')
    print('\n')

    score_matrix, ML_score = print_results(MonteCarloMatchSim(teams, 1000000, GamesLookback=int(glb), BaseOnxG=use_xg), teams)

    return score_matrix

    print('TODO: Change teh file location of the pickled date object')

# master()


# TODO: Add a Home Advantage term to the goal parameter of each team.
