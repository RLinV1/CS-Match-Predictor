import os
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def parse_overview_html(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

    data = {}
    
    
    tournament_name = soup.find("div", class_="menu-header").text.strip()
    
    data["tournament"] = tournament_name

    team_1_score_div = soup.find("div", class_="team-left").find("div", class_=lambda x: x and ("bold won" in x or "bold lost" in x))
    team_1_score = team_1_score_div.text.strip() if team_1_score_div else "N/A"

    team_2_score_div = soup.find("div", class_="team-right").find("div", class_=lambda x: x and ("bold won" in x or "bold lost" in x))
    team_2_score = team_2_score_div.text.strip() if team_2_score_div else "N/A"
    
    
    date_span = soup.find('span', {'data-time-format': 'yyyy-MM-dd HH:mm'})
    date = date_span.text.strip() if date_span else "Date not found"
    
    small_text_div = soup.find('div', class_='small-text')
    map_name = small_text_div.next_sibling.strip() if small_text_div and small_text_div.next_sibling else "Map not found"

    data["map"] = map_name
    data["date"] = date
    
    tables = pd.read_html(StringIO(str(soup)), attrs={"class": "stats-table totalstats"})

    if tables:  # Check if any tables were found
        # print(tables)
        team_1 = parse_table(tables[0])
        data.update(team_1)
        team_2 = parse_table(tables[1], "opp")
        data.update(team_2)

        # 1 is win
    if int(team_1_score) > int(team_2_score):
        data["won"] = 1
    else:
        data["won"] = 0

                
                
    return pd.DataFrame([data])


def parse_table(table_df, is_opp=""):
    player_name = table_df.columns[0]
    desired_columns = [player_name, 'K', 'A', 'D', 'ADR', 'KAST', 'Rating2.1']  # Replace with your actual column names

    if "KAST.1" in table_df.columns:
        table_df = table_df.drop(columns=["KAST.1"])  # Remove column
    
    # Convert relevant table data to dictionary format
    stat_cols = ['K (hs)', 'A (f)', 'D (t)']
    for col in stat_cols:
        if col in table_df.columns:
            # Extract first number and create new lowercase column
            table_df[col.split()[0].upper()] = table_df[col].str.extract(r'^(\d+)').astype(int)
            # Drop the original column if you don't need it
            table_df = table_df.drop(columns=[col])

    
    selected_data = table_df[desired_columns]
    selected_data.columns = [
        'Player', 'Kills', 'Assists', 'Deaths', 'ADR', 'KAST', 'Rating'
    ]
    
    df = pd.DataFrame(selected_data)

    df['KAST'] = df['KAST'].str.replace('%', '').astype(float)
    
    base_columns = ['Player', 'avg_kills', 'avg_assists', 'avg_deaths', 'avg_ADR', 'avg_KAST', 'avg_rating']
    if is_opp:
        df.columns = [col if col == 'Player' else f'{col}_opp' for col in base_columns]
    else:
        df.columns = base_columns
        
    # Calculate averages (excluding 'Player')
    avg_stats = df.drop('Player', axis=1).mean().to_dict()

    # Format KAST back to percentage
    
    if is_opp:
        avg_stats[f'avg_KAST_{is_opp}'] = f"{avg_stats[f'avg_KAST_{is_opp}']:.1f}%"
    else:
        avg_stats[f'avg_KAST'] = f"{avg_stats[f'avg_KAST']:.1f}%"


    selected_data = selected_data.to_dict('records')
    # print(table_df)
    
    rounded_stats = {
        key: round(value, 2) if isinstance(value, float) else value
        for key, value in avg_stats.items()
    }
    
    if is_opp:
        rounded_stats = {
            f"team_{is_opp}": player_name,
            **rounded_stats  # Unpacks the remaining stats
        }
    else:
        rounded_stats = {
            f"team": player_name,
            **rounded_stats  # Unpacks the remaining stats
        }
    return rounded_stats

MAPS_DIR = "hltv_data/maps"
maps = os.listdir(MAPS_DIR)

maps = [os.path.join(MAPS_DIR, f) for f in maps if f.endswith(".html")]

map_df = []
for file in maps:
    data = parse_overview_html(file)
    map_df.append(data)
    
all_data = pd.concat(map_df, ignore_index=True)
sorted_df = all_data.sort_values(by="date", ascending=False)

# Optional: reset index for cleaner display
sorted_df = sorted_df.reset_index(drop=True)
sorted_df.index.name = "Index"

print(sorted_df)

sorted_df.to_csv("data.csv")



