import pandas as pd
import os
def merge_clean_datasets(*urls):
    required_columns = {'sentence', 'difficulty'}
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


def get_file_paths(directory):
    """
    Returns a list of file paths for all files in the given directory.
    
    :param directory: The directory to search for files.
    :return: A list of file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    
    return file_paths

def get_full_dataset():
    """
    Returns the full dataset.
    
    :return: The full dataset.
    """
    # Get the file paths for all files in the data directory
    file_paths = get_file_paths('datasets/')
    
    # Merge the datasets
    df = merge_clean_datasets(*file_paths)
    
    return df
