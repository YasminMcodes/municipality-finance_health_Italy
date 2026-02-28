import pandas as pd

# ─── EDIT THIS ────────────────────────────────────────────────────────────
CSV_PATH = "./expense_2016_onwards.csv"
# ──────────────────────────────────────────────────────────────────────────

def count_na_zero_rows(df: pd.DataFrame) -> int:
    """
    Count rows (in df) where every cell is NA or exactly zero.
    """
    # Convert blank strings to NA
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    mask = df.isna() | (df == 0)
    return mask.all(axis=1).sum()

def main():
    try:
        # Read CSV, header=0 treats first line as column names
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Error reading CSV at {CSV_PATH}: {e}")
        return

    # Drop the first data row (index 0 after reading)
    df_data = df.iloc[1:].reset_index(drop=True)

    # Ignore the first three columns entirely for the NA/zero check
    df_to_check = df_data.iloc[:, 3:]

    total_rows = len(df_to_check)
    na_zero_count = count_na_zero_rows(df_to_check)

    print(f"Total rows checked (excluding header & first data row): {total_rows}")
    print(f"Rows where every checked column is NA or zero: {na_zero_count}")

    # ─── NEW SNIPPET: UNIQUE VALUE COUNTS IN SECOND COLUMN ──────────────────
    # Grab the second column (index 1) from the data (after dropping the title row)
    col2 = df_data.iloc[:, 1]
    unique_counts = col2.value_counts(dropna=False)

    print("\nUnique values in the second column and their counts:")
    for value, count in unique_counts.items():
        print(f"{value!r}: {count}")

if __name__ == "__main__":
    main()
