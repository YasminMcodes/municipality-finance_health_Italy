import pandas as pd

# File path
file_path = "inhabitants_data.csv"

# List of target provinces (with 'nan' treated as 'NA')
target_provinces = [
    'RC', 'CS', 'ME', 'CE', 'CT', 'NA', 'PA', 'RM', 'VV', 'CZ', 'SR', 'BN', 'CL', 'AV',
    'FR', 'AG', 'CH', 'EN', 'SA', 'CB', 'CN', 'RG', 'FG', 'TP', 'TA', 'VT', 'TR', 'BS',
    'PZ', 'GE', 'KR', 'PU', 'CO', 'AQ', 'MT', 'BG', 'LU', 'RI', 'BA', 'AL', 'PE', 'PV',
    'CR', 'IS', 'LT'
]

def classify_population(pop):
    if pop < 5000:
        return 'Small'
    elif pop < 20000:
        return 'Medium'
    elif pop < 60000:
        return 'Large'
    else:
        return 'Metropolitan'

def analyze_province_inhabitants(file_path):
    try:
        df = pd.read_csv(file_path)

        # Strip whitespace from column names
        df.columns = [col.strip() for col in df.columns]

        # Check required columns
        if 'Province' not in df.columns or 'inhabitants' not in df.columns:
            raise KeyError("Missing 'Province' or 'inhabitants' column.")

        # Treat NaN in Province as 'NA'
        df['Province'] = df['Province'].fillna('NA')

        # Apply classification
        df['SizeCategory'] = df['inhabitants'].apply(classify_population)

        print("\n📊 Municipality Size Counts by Province:\n")
        for province in target_provinces:
            filtered = df[df['Province'] == province]
            if filtered.empty:
                print(f"{province}: No data found.")
            else:
                counts = filtered['SizeCategory'].value_counts()
                print(f"{province}:")
                for category in ['Small', 'Medium', 'Large', 'Metropolitan']:
                    print(f"  {category}: {counts.get(category, 0)}")
                print()

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    analyze_province_inhabitants(file_path)
