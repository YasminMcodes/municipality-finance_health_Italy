import pandas as pd

# File path
file_path = "inhabitants_data.csv"

# Load CSV
try:
    df = pd.read_csv(file_path)

    # Show column names to verify what inhabitants column is called
    print("📋 Available columns:", df.columns.tolist())

    # Ensure column names are stripped of whitespace
    df.columns = [col.strip() for col in df.columns]

    # Change to your column name if it's different
    if 'inhabitants' not in df.columns:
        raise KeyError("❌ Missing 'inhabitants' column. Check column names above.")

    # Define the classification function
    def classify_population(pop):
        if pop < 5000:
            return 'Small'
        elif pop < 20000:
            return 'Medium'
        elif pop < 60000:
            return 'Large'
        else:
            return 'Metropolitan'

    # Apply the classification
    df['SizeCategory'] = df['inhabitants'].apply(classify_population)

    # Preview result
    print("\n✅ Sample classification:")
    print(df[['inhabitants', 'SizeCategory']].head())

    # Optionally save to new CSV
    df.to_csv("inhabitants_classified.csv", index=False)
    print("\n💾 Saved to 'inhabitants_classified.csv'")

except Exception as e:
    print(f"❌ Error: {e}")
