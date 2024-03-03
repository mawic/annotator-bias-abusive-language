import pandas as pd

def factorize_dataframe(df):
    if "rev_id" in df.columns or "id" in df.columns or "text" in df.columns:
        raise ValueError("Columns should be just the annotators (wide_format)")

    stacked = df.stack()
    codes, uniques = stacked.factorize()
    fact_df = pd.Series(codes, index=stacked.index).unstack()
    print(f"Factorized dataframe with {uniques}")
    return fact_df, uniques
