import pandas as pd
"""

# Load the data from a CSV file
data_path = 'utils/predictions_phase2_B.csv'  # Replace with the path to your CSV file
df = pd.read_csv(data_path)

# Function to convert predictions
def convert_prediction(prediction):
    return 'B1' if prediction == 0 else 'B2'

# Apply the conversion function to the 'predictions' column
df['difficulty'] = df['predictions'].apply(convert_prediction)

# Drop the original 'predictions' column
df.drop('predictions', axis=1, inplace=True)

# Save the modified dataframe back to a CSV file
output_path = 'utils/p2B.csv'  # Replace with your desired output file path
df.to_csv(output_path, index=False)

print("Data processing complete. File saved as:", output_path)

"""

# Define the file paths
file_A = 'utils/p2A.csv'
file_B = 'utils/p2B.csv'
file_C = 'utils/p2C.csv'

# Columns to keep
columns_to_keep = ['id', 'difficulty']

# Load the data from each file into separate DataFrames with selected columns
df_A = pd.read_csv(file_A)[columns_to_keep]
df_B = pd.read_csv(file_B)[columns_to_keep]
df_C = pd.read_csv(file_C)[columns_to_keep]

# Combine the DataFrames
combined_df = pd.concat([df_A, df_B, df_C])

# Sort by 'id' and reset the index
combined_df = combined_df.sort_values('id').reset_index(drop=True)

# Save the combined dataframe to a new CSV file
output_file = 'submission.csv'
combined_df.to_csv(output_file, index=False)

print("Files combined successfully. Output file:", output_file)
