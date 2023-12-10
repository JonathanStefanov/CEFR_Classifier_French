import streamlit as st
import os
import base64
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.predict import Predictor  # Ensure this import works correctly in your Streamlit environment

def get_table_download_link(df):
    """
    Generates a download link to allow the data in a DataFrame to be downloaded as a CSV.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be converted to CSV and downloaded.

    Returns:
    str: An HTML anchor tag as a string that allows the CSV to be downloaded.
    """    
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="submission.csv">Download CSV file</a>'
    return href

def verify_csv_structure(df):
    """
    Verifies if the given DataFrame has the expected column structure for processing.

    Parameters:
    df (pandas.DataFrame): The DataFrame whose structure is to be verified.

    Returns:
    bool: True if the DataFrame has the expected structure, False otherwise.
    """
    expected_columns = ['id', 'sentence']
    if df.columns.tolist() == expected_columns and len(df.columns) == 2:
        return True
    else:
        return False

def process_file(input_path, output_path, conversion_func):
    """
    Processes a CSV file by reading it, applying a conversion function, and saving the result.

    Parameters:
    input_path (str): Path to the input CSV file.
    output_path (str): Path where the processed CSV file will be saved.
    conversion_func (function): A function to apply to the 'predictions' column.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """
    df = pd.read_csv(input_path)
    df['difficulty'] = df['predictions'].apply(conversion_func)
    df.drop('predictions', axis=1, inplace=True)
    df.to_csv(output_path, index=False)
    return df

def combine_files(file_paths, output_file):
    """
    Combines multiple CSV files into a single file.

    Parameters:
    file_paths (list of str): A list of file paths to the CSV files to be combined.
    output_file (str): Path where the combined CSV file will be saved.

    Returns:
    pandas.DataFrame: The combined DataFrame.
    """
    dfs = [pd.read_csv(file)[['id', 'difficulty']] for file in file_paths]
    combined_df = pd.concat(dfs).sort_values('id').reset_index(drop=True)
    combined_df.to_csv(output_file, index=False)
    return combined_df

def main():
    st.title("CEFR Level Predictor")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if verify_csv_structure(data):
            st.success("The CSV file structure is correct.")
        else:
            st.error("The CSV file does not have the correct structure. It should only contain 'id' and 'sentence' columns.")
            return


        st.write("Uploaded Data:")
        st.write(data.head())  # Display the first few rows of the uploaded file

        # Input for model paths
        model_path_phase1 = st.text_input("Enter the path for Phase 1 Model", 'phase1.pth')
        model_path_A = st.text_input("Enter the path for Phase 2 Model A", 'phase_2_A.pth')
        model_path_B = st.text_input("Enter the path for Phase 2 Model B", 'phase_2_B.pth')
        model_path_C = st.text_input("Enter the path for Phase 2 Model C", 'phase_2_C.pth')

        if st.button("Make Predictions"):
            predictor = Predictor(model_path_phase1=model_path_phase1, model_path_phase2_A=model_path_A, model_path_phase2_B=model_path_B, model_path_phase2_C=model_path_C)
            
            # Phase 1 Inference
            predictor.inference_phase(1, data)

            # Further processing based on Phase 1 predictions
            df_A = data[data['predictions'] == 0].reset_index(drop=True)
            df_B = data[data['predictions'] == 1].reset_index(drop=True)
            df_C = data[data['predictions'] == 2].reset_index(drop=True)

            # Phase 2 Inference
            predictor.inference_phase(2, df_A, 'A')
            predictor.inference_phase(2, df_B, 'B')
            predictor.inference_phase(2, df_C, 'C')
            
            # Process individual files
            process_file('results/predictions_phase2_A.csv', 'results/p2A.csv', lambda x: 'A1' if x == 0 else 'A2')
            process_file('results/predictions_phase2_B.csv', 'results/p2B.csv', lambda x: 'B1' if x == 0 else 'B2')
            process_file('results/predictions_phase2_C.csv', 'results/p2C.csv', lambda x: 'C1' if x == 0 else 'C2')

            # Combine processed files
            combined_df = combine_files(['results/p2A.csv', 'results/p2B.csv', 'results/p2C.csv'], 'submission.csv')

            st.success("Processing complete. Combined file saved as 'submission.csv'.")
            st.write(combined_df.head())  

            # Create a download link for the processed file
            st.markdown(get_table_download_link(combined_df), unsafe_allow_html=True)




if __name__ == "__main__":
    main()
