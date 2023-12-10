import streamlit as st
import pandas as pd
from inference.predict import Predictor  # Ensure this import works correctly in your Streamlit environment

def main():
    st.title("French Sentence Difficulty Predictor")
    model_path_phase1 = 'phase1.pth'
    model_path_phase_2_A = 'phase_2_A.pth'
    model_path_phase2_B = 'phase_2_B.pth'
    model_path_phase2_C = 'phase_2_C.pth'

    # Text input for a French sentence
    sentence = st.text_input("Enter your French sentence here:")

    if st.button("Predict"):
        if sentence:
            predictor = Predictor(model_path_phase1=model_path_phase1, model_path_phase2_A=model_path_phase_2_A, model_path_phase2_B=model_path_phase2_B, model_path_phase2_C=model_path_phase2_C)

            # Placeholder for model paths (update these as per your model configuration)
            difficulty = predictor.inference_sentence(sentence)
            st.write("Predicted Difficulty Level: ", difficulty)

if __name__ == "__main__":
    main()