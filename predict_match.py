import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

def shift_col(team, col_name):
    team = team.sort_values("date")
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):  
    df = df.sort_values(["team", "date"])
    # Shift the OPPONENT column for next match
    return df.groupby("team")[col_name].shift(-1)

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


df["map_win_rate"] = df.groupby(["team", "map"])["won"].transform("mean")

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

# 1. Combine dataframes and clean
df = pd.concat([df, df_rolling], axis=1).dropna()

df["date"] = pd.to_datetime(df["date"]).dt.normalize()

# 3. Sort properly (critical!)
df = df.sort_values(['team', 'date', 'map']).reset_index(drop=True)

df["team_next"] = add_col(df, "team_opp")
df["date_next"] =  add_col(df, "date")
df["map_next"] = add_col(df, "map")

# 5. Create match IDs using index
df['match_id'] = df.index
df['next_match_id'] = df.index + 1  # Next row is next match

# 6. Handle edge cases  make them NA
last_matches = df.groupby('team').tail(1).index
df.loc[last_matches, 'next_match_id'] = pd.NA

right_df = df[rolling_cols + ['match_id']].copy()
full = df.merge(
    right_df,
    left_on='next_match_id',
    right_on='match_id',
    how='left',
    suffixes=('', '_next')
)


print("\n=== Check Dates ===")
wrong_dates = full[full["date_next"] < full["date"]]
print(f"Found {len(wrong_dates)} problematic rows where date_next < date")
if len(wrong_dates) > 0:
    print(wrong_dates[["team", "team_next", "date", "date_next"]])

print("\n=== Merge Results ===")
print(f"Total rows: {len(df)}")
print(f"Successful merges: {full['match_id_next'].notna().sum()}")
print(f"Failed merges: {full['match_id_next'].isna().sum()}")
# if full['match_id_next'].isna().sum() > 0:
    # print("\nSample failed matches:")
    # print(full[full['match_id_next'].isna()][['team', 'date', 'team_next', 'date_next']].head())
    
full_clean = full.dropna(subset=['next_match_id'])

print(f"Removed {len(full) - len(full_clean)} rows with missing next matches")
print(f"Final dataset has {len(full_clean)} rows")

# Add a flag column instead of removing
full['has_next_match'] = full['next_match_id'].notna()

# You can then filter later if needed
full = full[full['has_next_match']]


datetime_cols = full.select_dtypes(include=["datetime64"]).columns.tolist()

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
sfs = SequentialFeatureSelector(rr, n_features_to_select=12, direction="forward", cv=split)

print("Selected Features: ", list(selected_columns))
sfs.fit(full[selected_columns], full["target"])


predictors = list(selected_columns[sfs.get_support()])  # Then get selected features
print("Selected predictors:", predictors)

# print(full)
predictions = backtest(full, rr, predictors)
accuracy = accuracy_score(predictions["actual"], predictions["prediction"])
print(f"Backtest Accuracy with previous 10 games: {accuracy:.4f}")

# print(full[['team', 'team_opp', 'date', 'map', 'match_id', 
#            'team_next', 'date_next', 'map_next', 'next_match_id']].head(10).to_string())
