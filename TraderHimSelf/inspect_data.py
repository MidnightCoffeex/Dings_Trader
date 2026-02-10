import pandas as pd
try:
    print("--- 15m ---")
    df = pd.read_parquet('data_processed/aligned_15m.parquet')
    print(df.columns)
    print(df.head(1))
    
    print("\n--- Funding ---")
    df_f = pd.read_parquet('data_processed/funding.parquet')
    print(df_f.columns)
    print(df_f.head(1))
except Exception as e:
    print(e)
