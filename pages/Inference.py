import streamlit as st
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.predict import Predictor  # Ensure this import works correctly in your Streamlit environment

def process_file(input_path, output_path, conversion_func):
    df = pd.read_csv(input_path)
    df['difficulty'] = df['predictions'].apply(conversion_func)
    df.drop('predictions', axis=1, inplace=True)
    df.to_csv(output_path, index=False)
    return df

def combine_files(file_paths, output_file):
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
        st.write("Uploaded Data:")
        st.write(data.head())  # Display the first few rows of the uploaded file

        # Input for model paths
        model_path_phase1 = st.text_input("Enter the path for Phase 1 Model", 'phase1.pth')
        model_path_A = st.text_input("Enter the path for Phase 2 Model A", 'phase_2_A.pth')
        model_path_B = st.text_input("Enter the path for Phase 2 Model B", 'phase_2_B.pth')
        model_path_C = st.text_input("Enter the path for Phase 2 Model C", 'phase_2_C.pth')

        if st.button("Make Predictions"):
            predictor = Predictor()

            # Phase 1 Inference
            predictions_phase1 = predictor.inference_phase(1, model_path_phase1, data)

            # Further processing based on Phase 1 predictions
            df_A = data[data['predictions'] == 0].reset_index(drop=True)
            df_B = data[data['predictions'] == 1].reset_index(drop=True)
            df_C = data[data['predictions'] == 2].reset_index(drop=True)

            # Phase 2 Inference
            predictor.inference_phase(2, model_path_A, df_A)
            predictor.inference_phase(2, model_path_B, df_B)
            predictor.inference_phase(2, model_path_C, df_C)

            st.success("Predictions completed and saved to inference directory.")

        if st.button("Process and Combine Prediction Results"):
            # Process individual files
            process_file('results/predictions_phase2_A.csv', 'results/p2A.csv', lambda x: 'A1' if x == 0 else 'A2')
            process_file('results/predictions_phase2_B.csv', 'results/p2B.csv', lambda x: 'B1' if x == 0 else 'B2')
            process_file('results/predictions_phase2_C.csv', 'results/p2C.csv', lambda x: 'C1' if x == 0 else 'C2')

            # Combine processed files
            combined_df = combine_files(['results/p2A.csv', 'results/p2B.csv', 'results/p2C.csv'], 'submission.csv')

            st.success("Processing complete. Combined file saved as 'submission.csv'.")
            st.write(combined_df.head())  # Optionally display the first few rows


if __name__ == "__main__":
    main()
