import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False)[df.columns.tolist()].apply(lambda x: shift_col(x, col_name))

# test based on previous data to predict future data
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    unique_dates = sorted(data["date"].unique())

    for i in range(start, len(unique_dates), step):
        train_end_date = unique_dates[i]

        train = data[data["date"] < train_end_date]
        test = data[data["date"] == train_end_date]

        if len(train) == 0 or len(test) == 0:
            continue

        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)

    return pd.concat(all_predictions)

def find_team_averages(team):
    # Only calculate rolling means on numeric columns
    numeric_cols = team.select_dtypes(include='number').columns
    rolling = team[numeric_cols].rolling(10, min_periods=1).mean()

    # Add back 'team' and 'date' so we can merge later
    rolling["team"] = team["team"]
    rolling["date"] = team["date"]

    return rolling

# start of code; load data
df = pd.read_csv("data.csv", index_col=0)


df["date"] = pd.to_datetime(df["date"])

# Sort by team and date to ensure chronological order within teams
df = df.sort_values(["team", "date"])


# convert percentages to floats
df['avg_KAST'] = df['avg_KAST'].str.replace('%', '').astype(float) / 100
df['avg_KAST_opp'] = df['avg_KAST_opp'].str.replace('%', '').astype(float) / 100

# deaths should be bad
df['avg_deaths'] = -df['avg_deaths'] 
df['avg_deaths_opp'] = -df['avg_deaths_opp'] 

# target is if they won the next game
def add_target(team):
    team["target"] = team["won"].shift(-1)
    return team

df = df.groupby("team", group_keys=False)[df.columns.tolist()].apply(add_target)

# Drop rows where target is null (last games)
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)



# Remove non-feature columns
removed_columns = ["Index", "tournament", "map", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]

print("Selected features:", list(selected_columns))

scaler = MinMaxScaler()
# convert to values 0-1 (0 is bad 1 is good)
df[selected_columns] = scaler.fit_transform(df[selected_columns])

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)

sfs = SequentialFeatureSelector(rr, n_features_to_select=9, direction="forward", cv=split)
sfs.fit(df[selected_columns], df["target"])

predictors = list(selected_columns[sfs.get_support()])
print("Selected predictors:", predictors)


# Run backtest
predictions = backtest(df, rr, predictors)

# Remove rows with target == 2 if any 
predictions = predictions[predictions["actual"] != 2]

accuracy = accuracy_score(predictions["actual"], predictions["prediction"])

print(f"Backtest Accuracy: {accuracy:.4f}")


# Test data on last 10 games
df_rolling = df[list(selected_columns) + ["won", "team", "date"]]

df_rolling = df_rolling.groupby(["team", "date"], group_keys=False)[df_rolling.columns.tolist()].apply(find_team_averages)
rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols

# combine the two dataframes
df = pd.concat([df, df_rolling], axis=1)
df = df.dropna()



# add data on the next team to face and the next date they'll face them
df["team_next"] = add_col(df, "team") #just the first team
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

# no timestamps just dates
df["date"] = pd.to_datetime(df["date"]).dt.normalize()
df["date_next"] = pd.to_datetime(df["date_next"]).dt.normalize()

# merge the values
full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])

datetime_cols = full.select_dtypes(include=["datetime64"]).columns.tolist()

# Combine with your removed columns. We do not want any datetime objects in there 
removed_cols_merge = list(full.columns[full.dtypes == "object"]) + [
    "tournament", "map", "date", "date_next",
    "team", "team_next", "team_opp", "team_opp_next", "target"
] + datetime_cols

# filter the columns
selected_columns = full.columns[~full.columns.isin(removed_cols_merge)]
if full.empty:
    print("Merge failed: no matching rows in full DataFrame")
    exit()

print()

sfs = SequentialFeatureSelector(rr, n_features_to_select=15, direction="forward", cv=split)

print("Selected Features: ", list(selected_columns))
sfs.fit(full[selected_columns], full["target"])


predictors = list(selected_columns[sfs.get_support()])  # Then get selected features
print("Selected predictors:", predictors)

predictions = backtest(full, rr, predictors)
accuracy = accuracy_score(predictions["actual"], predictions["prediction"])
print(f"Backtest Accuracy with previous 10 games: {accuracy:.4f}")

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False)[df.columns.tolist()].apply(lambda x: shift_col(x, col_name))

# test based on previous data to predict future data
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    unique_dates = sorted(data["date"].unique())

    for i in range(start, len(unique_dates), step):
        train_end_date = unique_dates[i]

        train = data[data["date"] < train_end_date]
        test = data[data["date"] == train_end_date]

        if len(train) == 0 or len(test) == 0:
            continue

        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)

    return pd.concat(all_predictions)

def find_team_averages(team):
    # Only calculate rolling means on numeric columns
    numeric_cols = team.select_dtypes(include='number').columns
    rolling = team[numeric_cols].rolling(10, min_periods=1).mean()

    # Add back 'team' and 'date' so we can merge later
    rolling["team"] = team["team"]
    rolling["date"] = team["date"]

    return rolling

# start of code; load data
df = pd.read_csv("data.csv", index_col=0)


df["date"] = pd.to_datetime(df["date"])

# Sort by team and date to ensure chronological order within teams
df = df.sort_values(["team", "date"])


# convert percentages to floats
df['avg_KAST'] = df['avg_KAST'].str.replace('%', '').astype(float) / 100
df['avg_KAST_opp'] = df['avg_KAST_opp'].str.replace('%', '').astype(float) / 100

# deaths should be bad
df['avg_deaths'] = -df['avg_deaths'] 
df['avg_deaths_opp'] = -df['avg_deaths_opp'] 

# target is if they won the next game
def add_target(team):
    team["target"] = team["won"].shift(-1)
    return team

df = df.groupby("team", group_keys=False)[df.columns.tolist()].apply(add_target)

# Drop rows where target is null (last games)
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)



# Remove non-feature columns
removed_columns = ["Index", "tournament", "map", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]

print("Selected features:", list(selected_columns))

scaler = MinMaxScaler()
# convert to values 0-1 (0 is bad 1 is good)
df[selected_columns] = scaler.fit_transform(df[selected_columns])

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)

sfs = SequentialFeatureSelector(rr, n_features_to_select=9, direction="forward", cv=split)
sfs.fit(df[selected_columns], df["target"])

predictors = list(selected_columns[sfs.get_support()])
print("Selected predictors:", predictors)


# Run backtest
predictions = backtest(df, rr, predictors)

# Remove rows with target == 2 if any 
predictions = predictions[predictions["actual"] != 2]

accuracy = accuracy_score(predictions["actual"], predictions["prediction"])

print(f"Backtest Accuracy: {accuracy:.4f}")


# Test data on last 10 games
df_rolling = df[list(selected_columns) + ["won", "team", "date"]]

df_rolling = df_rolling.groupby(["team", "date"], group_keys=False)[df_rolling.columns.tolist()].apply(find_team_averages)
rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols

# combine the two dataframes
df = pd.concat([df, df_rolling], axis=1)
df = df.dropna()



# add data on the next team to face and the next date they'll face them
df["team_next"] = add_col(df, "team") #just the first team
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

# no timestamps just dates
df["date"] = pd.to_datetime(df["date"]).dt.normalize()
df["date_next"] = pd.to_datetime(df["date_next"]).dt.normalize()

# merge the values
full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])

datetime_cols = full.select_dtypes(include=["datetime64"]).columns.tolist()

# Combine with your removed columns. We do not want any datetime objects in there 
removed_cols_merge = list(full.columns[full.dtypes == "object"]) + [
    "tournament", "map", "date", "date_next",
    "team", "team_next", "team_opp", "team_opp_next", "target"
] + datetime_cols

# filter the columns
selected_columns = full.columns[~full.columns.isin(removed_cols_merge)]
if full.empty:
    print("Merge failed: no matching rows in full DataFrame")
    exit()

print()

sfs = SequentialFeatureSelector(rr, n_features_to_select=15, direction="forward", cv=split)

print("Selected Features: ", list(selected_columns))
sfs.fit(full[selected_columns], full["target"])


predictors = list(selected_columns[sfs.get_support()])  # Then get selected features
print("Selected predictors:", predictors)

predictions = backtest(full, rr, predictors)
accuracy = accuracy_score(predictions["actual"], predictions["prediction"])
print(f"Backtest Accuracy with previous 10 games: {accuracy:.4f}")
