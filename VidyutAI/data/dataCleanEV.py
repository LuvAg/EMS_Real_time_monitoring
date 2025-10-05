import pandas as pd

# Read the dataset
df = pd.read_csv("data/hourlyEVdata.csv")

# Ensure it's sorted properly (important before detecting contiguous blocks)
df = df.sort_values(by=["Date", "Hour"]).reset_index(drop=True)

# Detect where the energy changes
df["same_as_prev"] = df["Energy_kWh"].eq(df["Energy_kWh"].shift())

# Assign group IDs for contiguous same-value regions
df["group_id"] = (~df["same_as_prev"]).cumsum()

# Count how many rows each group has
group_sizes = df.groupby("group_id")["Energy_kWh"].transform("size")

# Drop rows that belong to groups with size >= 7
df_filtered = df[group_sizes < 7].copy()

# Drop helper columns
df_filtered = df_filtered.drop(columns=["same_as_prev", "group_id"])

print(f"Original rows: {len(df)}, Cleaned rows: {len(df_filtered)}")
print(df_filtered.head(10))

df_filtered.to_csv("data/hourlyEVusage_cleaned.csv", index=False)