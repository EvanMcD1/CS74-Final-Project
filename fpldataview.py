import pandas as pd

df = pd.read_csv('data/merged_gw.csv', on_bad_lines='skip')

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"\nColumn names:\n{df.columns.tolist()}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nPoints stats:")
print(df['total_points'].describe())

print(f"\nPoints by position:")
print(df.groupby('position')['total_points'].mean())

print(f"\nPlayers who played (minutes > 0): {len(df[df['minutes'] > 0])}")
print(f"Players who didn't play: {len(df[df['minutes'] == 0])}")