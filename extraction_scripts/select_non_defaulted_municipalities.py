import pandas as pd

# Load and normalize
defaulted_df = pd.read_csv('defaulted_municipalities_with_attraction.csv')
non_defaulted_df = pd.read_csv('non_defaulted_municipalities_with_attraction.csv')

# Standardize column names
defaulted_df.rename(columns={'municipalities': 'Municipality', 'population': 'Population'}, inplace=True)
non_defaulted_df.rename(columns={'municipalities': 'Municipality', 'population': 'Population'}, inplace=True)

# Treat NA or empty in Province as string "NA"
defaulted_df['Province'] = defaulted_df['Province'].fillna('NA').replace('', 'NA')
non_defaulted_df['Province'] = non_defaulted_df['Province'].fillna('NA').replace('', 'NA')

# Define population category
def get_population_category(pop):
    try:
        pop = int(pop)
        if pop < 5000:
            return 'Small'
        elif 5000 <= pop < 20000:
            return 'Medium'
        elif 20000 <= pop < 60000:
            return 'Large'
        else:
            return 'Metropolitan'
    except:
        return None

defaulted_df['PopCategory'] = defaulted_df['Population'].apply(get_population_category)
non_defaulted_df['PopCategory'] = non_defaulted_df['Population'].apply(get_population_category)

selected_rows = []
unmatched = []

for _, row in defaulted_df.iterrows():
    province = row['Province']
    pop_cat = row['PopCategory']
    attraction = row['Attraction']

    # Try strict match first
    matches = non_defaulted_df[
        (non_defaulted_df['Province'] == province) &
        (non_defaulted_df['PopCategory'] == pop_cat) &
        (non_defaulted_df['Attraction'] == attraction)
    ]

    # Relax match if no result
    if matches.empty:
        matches = non_defaulted_df[
            (non_defaulted_df['Province'] == province) &
            (non_defaulted_df['Attraction'] == attraction)
        ]

    if matches.empty:
        unmatched.append(row[['Municipality', 'Province', 'Population', 'Attraction']].to_dict())
        continue

    selected = matches.sample(1)
    selected_rows.append(selected[['Municipality', 'Province', 'Population', 'Attraction']].iloc[0])

# Save matches
selected_df = pd.DataFrame(selected_rows)
selected_df.to_csv('selected.csv', index=False)

# Save unmatched for review
if unmatched:
    unmatched_df = pd.DataFrame(unmatched)
    unmatched_df.to_csv('unmatched.csv', index=False)
    print(f"{len(unmatched)} municipalities could not be matched and were saved in 'unmatched.csv'.")

print(f"{len(selected_df)} municipalities were matched and saved in 'selected.csv'.")
