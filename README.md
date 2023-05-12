# Football-StreamLit-Page

This is my project on football score prediction!

The PDF file attached gives a deeper insight behind the methods I have used to predict the scores of football matches, but in summary I used a Bivariate Poisson distribution, where I have tried to take into account a teams 'difficulty to score against' when calculating the average goal rate parameters, inside a Monte Carlo simulation to generate score probabilities.

This project involves:
- Extracting data from an API & subsequent logic to extract the data in the right way 
- Calling and updating a database (stored as pickle objects) in order to avoid un-necessary API calls as the season progresses to update the teams stats
- Implimentation of a Bivariate Poisson distribtion to sample from in a Monte Carlo simulation
- A CLI to interact with the program and display the results
- And finally, an **interactive web app** which anyone can access, via the URL, which was my first time using streamlit!

[Access the web app here](https://zac-tiller-football-streamlit-page-mcfootballapp-f8efjr.streamlit.app/)
