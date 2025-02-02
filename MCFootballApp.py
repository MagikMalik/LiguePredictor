import streamlit as st
import seaborn as sns
from MC_Score_Predictor import MonteCarloMatchSim, buildScoreMatrix
import matplotlib.pyplot as plt
from understatapi import UnderstatClient
import pickle
import streamlit as st
import os

# Récupérer les valeurs des variables d'environnement
USERNAME = st.secrets["MGKMLKUSER"]
PASSWORD = st.secrets["MGKMLKPASS"]
USERNAME2 = st.secrets["MGKMLKUSER2"]
PASSWORD2 = st.secrets["MGKMLKPASS2"]
print(USERNAME)


def main():
    if 'league' not in st.session_state:
        st.session_state['league'] = ''
        st.session_state['home_team'] = ''
        st.session_state['away_team'] = ''
        st.session_state['list_team'] = ''


    header = st.container()
    league_selector = st.container()
    team_selector = st.container()
    stats_selector = st.container()
    simulation_engine = st.container()

    score_probabilities = st.container()
    team_win_probabilities = st.container()

    predictcote = st.container()
    top_three_scores = st.container()


    def back_grad(df):
        return df.to_frame().style.background_gradient(cmap='viridis').set_properties(**{'font-size': '10px'})


    def ML_scores(score_matrix, MC_Score_tracker):
        ML_score_dict = {}
        remove_max_score = lambda x, max_likelihood_score: x.pop(max_likelihood_score)
        max_likelihood_score = lambda x: max(x, key=x.get)
        max_likelihood_percent = lambda x: max(x.values())

        ML_score = max_likelihood_score(MC_score_tracker)
        for i in range(3):

            if i > 0:
                remove_max_score(MC_score_tracker, ML_score)
                ML_score = max_likelihood_score(MC_score_tracker)
            ML_percent = max_likelihood_percent(MC_score_tracker)
            ML_score_dict[ML_score] = ML_percent/10000

        return ML_score_dict

    def generate_teamids_dict(CreateNew, league):
        if CreateNew: # Create new prem league team data, and a new empty data dictionary - set CreateNew to True at the start of each season
            print('Creating New Data Objects')
            with UnderstatClient() as understat:
                print('Attempting to Collect API Data...')
                league_team_data = understat.league(league=league).get_team_data(season="2022")
                print('Collected API Data Successfully!')
            league_ids = {}

            for k, v in league_team_data.items():
                team_name = league_team_data[k]['title']
                team_id = k
                league_ids[team_name] = team_id
                

            print('User Created New (Empty) League Pickle Objects')
            pickle.dump(league_ids, open(league+"TeamIDs.p", 'wb'))

        else:
            league_ids = pickle.load(open(league+"TeamIDs.p", "rb"))

        return league_ids


    st.markdown(
        """
        <style>
        .main{
        background-color: #F5F5F5;
        }
        <style>
        """,
        unsafe_allow_html=True
    )

    league_List = ['EPL', 'La_Liga', 'Bundesliga', 'Serie_A', 'Ligue_1', 'RFPL']

    list_team = generate_teamids_dict(CreateNew=False,league='La_Liga')

    with header:
        # Obtenir le chemin absolu du répertoire actuel
        image_url = "https://raw.githubusercontent.com/MagikMalik/LiguePredictor/master/MagikMalikAvatar.png"
        rounded_image_html = '<img src="{}" style="width:15%;border-radius: 50%;">'.format(image_url)

        # Afficher le texte avec l'image
        st.markdown("<h1 style='text-align:center;height: 15px;font-size: 35px;'>Magikmalik Match<h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:center;height: 110px;font-size: 65px;'>Predict" + rounded_image_html + "r</h1>", unsafe_allow_html=True)

        # st.title('MagikMalik Match Predictor')

    with league_selector:
        league_col = st.columns(1)[0]
        league = league_col.selectbox('Choisir une ligue:', options=league_List, index=0)

        # Mettre à jour les équipes disponibles lorsque la ligue est modifiée
        if st.session_state.league != league:
            st.session_state.league = league
            list_team = generate_teamids_dict(CreateNew=False,league=league)
            st.session_state.list_team = list_team
            st.session_state.home_team = list(list_team.keys())[0]
            st.session_state.away_team = list(list_team.keys())[1]

    with team_selector:
        home_col, away_col = st.columns(2)
        list_team = generate_teamids_dict(CreateNew=False,league=league)
        if hasattr(st.session_state, 'list_team'):
            home_team = home_col.selectbox('Equipe à domicile:', options=list(list_team.keys()), index=list(list_team.keys()).index(st.session_state.home_team), key='home_team')
            away_team = away_col.selectbox('Equipe à l\'extérieur:', options=list(list_team.keys()), index=list(list_team.keys()).index(st.session_state.away_team), key='away_team')
        else:
            home_team = home_col.selectbox('Equipe à domicile:')
            away_team = away_col.selectbox('Equipe à l\'extérieur:')
        print("dom: " + home_team)
        print("ext: " + away_team)
    teams = [home_team, away_team]

    with stats_selector:
        st.markdown('**Paramétrage:**')

        lookback_col, goal_type_col = st.columns(2)

        games_lookback = lookback_col.slider('Combien de matchs historiques pour le calcul?', min_value=2, max_value=10, value=10)
        goal_type = goal_type_col.selectbox('Calcul basé sur les buts réels(G) ou sur l\'espérance de buts(xG)?', options=['G', 'xG'], index=1)

    glb = games_lookback

    display_button = False

    if goal_type == 'G':
        use_xg = 'False'
        display_button = True
    else:
        use_xg = 'True'
        display_button = True

    col1, simulation_engine, col3 = st.columns(3)

    with simulation_engine:

        run_button = st.empty()
        with run_button.container():

            submit = False
            if st.button("Exécuter la simulation"):
                submit = True

                if submit:
                    run_button.empty()
                home_win_prob, away_win_prob, draw_prob, MC_score_tracker, x, y, HT_GR, AT_GR = MonteCarloMatchSim(teams, 1000000, GamesLookback=int(glb), BaseOnxG=use_xg,league=league)

        if submit:
            sim_end_msg = '<p style="font-family:Arial; color:Red; font-size: 14px;">Simulation terminée. Sélectionnez Nouvelles équipes ou modifiez les paramètres pour recommencer !</p>'
            st.markdown(sim_end_msg, unsafe_allow_html=True)

            with score_probabilities:
                score_matrix = buildScoreMatrix(MC_score_tracker, teams, x, y)

                st.markdown('**Espérance de buts attendus:**')
                ht_param, at_param = st.columns(2)
                ht_param.markdown('**{}** Buts attendus: {}'.format(home_team, round(HT_GR, 3)))
                at_param.markdown('**{}** Buts attendus: {}'.format(away_team, round(AT_GR, 3)))


                st.subheader('Probabilités du nombre de buts: ')

                # fig, ax = plt.subplots()
                # sns.heatmap(fig, ax=ax)
                # st.pyplot(fig)
                #
                # # sns.heatmap(score_matrix, annot=True, linewidth=.5, cmap='OrRd', ax=ax)
                # # st.pyplot(fig)

                st.dataframe(score_matrix)
                # st.dataframe(data=score_matrix.style.background_gradient(cmap ='OrRd'))

                #st.dataframe(score_matrix.apply(back_grad))

                valubet = st.container()
                valubet.markdown('<span style="font-family:Arial;font-weight:bold;font-size:14px;">Prédiction pour le match </span><span style="font-family:Arial; color:Red; font-size: 10px;">Vérifier par rapport a la cote de votre bookmaker si il y a un valuebet (cad: Cote bookmaker plus élevé)</span>', unsafe_allow_html=True)
                with predictcote:
                    home_win_row, draw_row, away_win_row = st.columns(3)

                    home_win_row.markdown('{}/ **{} % (Cote: {})**'.format(home_team, round(home_win_prob, 1), round(100/home_win_prob, 2)), unsafe_allow_html=True)
                    draw_row.markdown('Nul / **{} % (Cote: {})**'.format(round(draw_prob, 1),round(100/draw_prob, 2)), unsafe_allow_html=True)
                    away_win_row.markdown('{} / **{} % (Cote: {})**'.format(away_team, round(away_win_prob, 1), round(100/away_win_prob, 2)), unsafe_allow_html=True)

                with top_three_scores:

                    likelyscore = st.container()
                    most_likely_score, second_likely_score, third_likely_score = st.columns(3)
                    ML_score_dict = ML_scores(score_matrix, MC_score_tracker)

                    likelyscore.markdown('**Scores les plus probable**')
                    most_likely_score.markdown('**1/** Score probable {}: {} %'.format(list(ML_score_dict.keys())[0],
                                                                                       round(list(ML_score_dict.values())[0], 2)))
                    second_likely_score.markdown('**2/** Score probable {}: {} %'.format(list(ML_score_dict.keys())[1],
                                                                                        round(list(ML_score_dict.values())[1], 2)))
                    third_likely_score.markdown('**3/** Score probable {}: {} %'.format(list(ML_score_dict.keys())[2],
                                                                                        round(list(ML_score_dict.values())[2], 2)))

# Ajouter une zone de saisie pour le nom d'utilisateur et le mot de passe
username_input = st.sidebar.text_input('Nom d\'utilisateur')
password_input = st.sidebar.text_input('Mot de passe', type='password')
st.markdown(
    """
    <style>
    div.stActionButton{
    display: none;
    }
    <style>
    """,
    unsafe_allow_html=True
    )
if st.sidebar.button("Se connecter"):
    if (username_input == USERNAME and password_input == PASSWORD) or (username_input == USERNAME2 and password_input == PASSWORD2):
        st.markdown(
        """
        <style>
        section[data-testid="stSidebar"],div.stActionButton{
        display: none;
        }
        <style>
        """,
        unsafe_allow_html=True
        )
        main()
    else:
        # Les informations d'authentification sont incorrectes, afficher un message d'erreur
        if username_input != '' and password_input != '':
            st.sidebar.error('Identifiants invalides. Veuillez réessayer.')
elif (username_input == USERNAME and password_input == PASSWORD) or (username_input == USERNAME2 and password_input == PASSWORD2):
    st.markdown(
    """
    <style>
    section[data-testid="stSidebar"],div.stActionButton{
    display: none;
    }
    <style>
    """,
    unsafe_allow_html=True
    )
    main()
elif username_input != '' and password_input != '':
    st.sidebar.error('Identifiants invalides. Veuillez réessayer.')
