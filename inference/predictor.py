import pandas as pd
from phase1 import inference_phase_1
from phase2 import inference_phase_2

unseen_data_path = 'path_to_unseen_data.csv'  # Replace with your unseen data path
output_file_path = 'predictions_phase1.csv'  # The path where you want to save the predictions

inference_phase_1(unseen_data_path, output_file_path)

# Read the CSV file into a DataFrame
df = pd.read_csv(output_file_path)

df_A = df[df['predictions'] == 0]
df_B = df[df['predictions'] == 1]
df_C = df[df['predictions'] == 2]

# Phase 2
model_path_A = 'phase_2_A.pth'  
model_path_B = 'phase_2_B.pth' 
model_path_C = 'phase_2_C.pth'  

df_A = df_A.reset_index(drop=True)
df_B = df_B.reset_index(drop=True)
df_C = df_C.reset_index(drop=True)

inference_phase_2('A', model_path_A, df_A)
inference_phase_2('B', model_path_B, df_B)
inference_phase_2('C', model_path_C, df_C)
