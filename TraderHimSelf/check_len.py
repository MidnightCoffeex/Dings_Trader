import pandas as pd
df = pd.read_parquet('data_processed/aligned_15m.parquet')
print(f"Length: {len(df)}")
print(f"Start: {df.index[0]}")
print(f"End: {df.index[-1]}")
