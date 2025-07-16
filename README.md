# Match Predictor for CS

This project is a match predictor that utilizes scklit-learn to predict the match results for a popular esports game called Counter Strike.
It uses scraped data from HLTV, a known reliable site for data, and feeds it to a machine learning model to predict
the outcomes of games. This project uses almost 300 matches to train the model.

# How it works
The match predictor uses past data to train and predicts matches. It then compares the prediction against the real data.
The data the model trains on is only past data. There's two implementations where one uses 9 features to predict while
the other uses 15 features to predict and utilizes the past 10 games to determine the outcome.

Results:
<img width="2427" height="427" alt="image" src="https://github.com/user-attachments/assets/b871eedf-19c3-4d31-a34a-86e8a5626347" />



