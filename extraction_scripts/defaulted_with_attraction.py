import pandas as pd
import unicodedata
import re

def normalize_name(name):
    if pd.isna(name):
        return ''
    # Lowercase
    name = name.lower()
    # Replace hyphen with space
    name = name.replace('-', ' ')
    # Remove apostrophes
    name = name.replace("'", "")
    # Remove accents
    name = ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    )
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# Load inhabitants CSV
inhabitants_df = pd.read_csv('inhabitants_data.csv')

# Load classification Excel (header in 3rd row means skiprows=2)
classification_df = pd.read_excel('classification.xlsx', header=2)

# Normalize municipalities names for merging
inhabitants_df['municipalities_norm'] = inhabitants_df['municipalities'].apply(normalize_name)
classification_df['COMUNE_norm'] = classification_df['COMUNE'].apply(normalize_name)

# Merge on normalized municipalities name
merged_df = pd.merge(
    inhabitants_df,
    classification_df[['COMUNE_norm', 'COD_CAT', 'POPOLAZIONE 1/1/2019']],
    left_on='municipalities_norm',
    right_on='COMUNE_norm',
    how='left'
)

# Replace 'inhabitants' with 'POPOLAZIONE 1/1/2019' where available
merged_df['population'] = merged_df['POPOLAZIONE 1/1/2019'].combine_first(merged_df['inhabitants'])

# Rename COD_CAT to Attraction
merged_df.rename(columns={'COD_CAT': 'Attraction'}, inplace=True)

# Drop helper columns used for merging and original inhabitants/population columns
merged_df.drop(columns=['municipalities_norm', 'COMUNE_norm', 'POPOLAZIONE 1/1/2019', 'inhabitants'], inplace=True)
merged_df['Province'] = merged_df['Province'].fillna('NA')
# Save to a new CSV file
merged_df.to_csv('inhabitants_with_attraction2.csv', index=False)

print("Processing complete. Saved to 'inhabitants_with_attraction2.csv'.")
