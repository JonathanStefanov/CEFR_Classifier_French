import pandas as pd
from phase1 import inference_phase_1

unseen_data_path = 'path_to_unseen_data.csv'  # Replace with your unseen data path
output_file_path = 'predictions.csv'  # The path where you want to save the predictions

csv_file_path = 'path_to_unseen_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Filter the DataFrame for predictions equal to 0
df_A = df[df['predictions'] == 0]
df_B = df[df['predictions'] == 1]
df_C = df[df['predictions'] == 2]
