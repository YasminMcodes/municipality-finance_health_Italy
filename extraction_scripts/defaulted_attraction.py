import pandas as pd
import unicodedata
import re

def normalize_name(name):
    if pd.isna(name):
        return ""
    # Lowercase
    name = name.lower()
    # Replace hyphens with spaces
    name = name.replace("-", " ")
    # Remove apostrophes
    name = name.replace("'", "")
    # Normalize unicode accents to ASCII
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    # Remove extra spaces (multiple spaces to one)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# Load files
inhabitants_df = pd.read_csv("inhabitants_data.csv")
classification_df = pd.read_excel("classification.xlsx", header=2)
classification_df.columns = classification_df.columns.str.strip()

# Apply normalization to the municipalities columns
inhabitants_df['municipalities_norm'] = inhabitants_df['municipalities'].apply(normalize_name)
classification_df['COMUNE_norm'] = classification_df['COMUNE'].apply(normalize_name)

# Merge on normalized columns
merged_df = inhabitants_df.merge(
    classification_df[['COMUNE_norm', 'COD_CAT']],
    left_on='municipalities_norm',
    right_on='COMUNE_norm',
    how='left'
)
# Fill missing Popolazione in inhabitants_df from classification_df based on normalized municipality names
# Fill missing inhabitants from classification population
def fill_population(row):
    if pd.isna(row['inhabitants']) or row['inhabitants'] == '':
        return row['POPOLAZIONE 1/1/2019']
    else:
        return row['inhabitants']

merged_df['inhabitants'] = merged_df.apply(fill_population, axis=1)

# Drop extra columns and rename
merged_df.drop(columns=['COMUNE_norm', 'Municipality_norm', 'POPOLAZIONE 1/1/2019_x', 'POPOLAZIONE 1/1/2019_y'], inplace=True)
merged_df.rename(columns={'COD_CAT': 'Attraction'}, inplace=True)


# Clean up and rename
merged_df.drop(columns=['COMUNE_norm', 'municipalities_norm'], inplace=True)
merged_df.rename(columns={'COD_CAT': 'Attraction'}, inplace=True)
merged_df['Popolazione'] = merged_df.apply(fill_population, axis=1)

# Save to CSV
merged_df.to_csv("inhabitant_data_with_attraction2.csv", index=False)

print("✅ Normalized merge done and saved to 'inhabitant_data_with_attraction2.csv'")
