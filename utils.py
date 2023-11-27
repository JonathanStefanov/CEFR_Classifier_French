import pandas as pd

def merge_clean_datasets(*urls):
    required_columns = {'id', 'sentence', 'difficulty'}
    dataframes = []

    for url in urls:
        df = pd.read_csv(url)

        # Check if the dataframe contains the required columns
        if not required_columns.issubset(df.columns):
            raise ValueError(f"The dataset from {url} does not contain the required columns.")

        dataframes.append(df)

    # Merge the dataframes
    df_merged = pd.concat(dataframes, ignore_index=True)

    # Remove duplicate sentences
    df_merged = df_merged.drop_duplicates(subset='sentence', keep='first')

    return df_merged