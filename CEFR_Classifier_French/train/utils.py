import pandas as pd
import os

def merge_clean_datasets(*urls):
    """
    Merges multiple datasets from given URLs. Ensures that each dataset contains 
    the required columns ('sentence', 'difficulty') and removes duplicate sentences.

    Parameters:
    *urls : str
        Variable number of URL strings pointing to the datasets to be merged.

    Raises:
    ValueError
        If any dataset does not contain the required columns.

    Returns:
    DataFrame
        The merged and cleaned dataframe containing unique sentences and their 
        corresponding difficulty levels.
    """

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
    
    Parameters:
    directory : str
        The directory to search for files.

    Returns:
    list
        A list of file paths for all files in the specified directory.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    
    return file_paths

def get_full_dataset():
    """
    Retrieves and returns the full dataset by merging and cleaning individual datasets 
    found in the 'datasets/' directory.

    Returns:
    DataFrame
        The full, merged, and cleaned dataset.
    """
    # Get the file paths for all files in the data directory
    file_paths = get_file_paths('datasets/')
    
    # Merge the datasets
    df = merge_clean_datasets(*file_paths)
    
    return df
