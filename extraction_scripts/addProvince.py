import pandas as pd

# Load both CSVs
df_non_defaulted = pd.read_csv("non_defaulted_municipalities.csv", encoding="ISO-8859-1", sep=";")
df_target = pd.read_csv("municipality_province_new.csv")

# Merge relevant columns
df_merged = pd.merge(
    df_target,
    df_non_defaulted[["Municipality", "Popolazione", "Attrattività"]],
    on="Municipality",
    how="left"
)

# Rename columns
df_merged = df_merged.rename(columns={
    "Popolazione": "Population",
    "Attrattività": "Attraction"
})

df_merged['Province'] = df_merged['Province'].fillna('NA')


# Save to new CSV
df_merged.to_csv("non_defaulted_municipalities_cleaned.csv", index=False)

print("Done! New file: non_defaulted_municipalities_cleaned.csv")
