import streamlit as st
import pandas as pd
from inference.predict import Predictor  # Ensure this import works correctly in your Streamlit environment

def main():
    st.title("French Sentence Difficulty Predictor")

    # Text input for a French sentence
    sentence = st.text_input("Enter your French sentence here:")

    if sentence:
        # Placeholder for model paths (update these as per your model configuration)
        model_path_phase1 = 'phase1.pth'
        model_path_phase_2_A = 'phase_2_A.pth'
        model_path_phase2_B = 'phase_2_B.pth'
        model_path_phase2_C = 'phase_2_C.pth'
        letter = ''

        predictor = Predictor(model_path_phase1=model_path_phase1, model_path_phase2_A=model_path_phase_2_A, model_path_phase2_B=model_path_phase2_B, model_path_phase2_C=model_path_phase2_C)

        # Create a DataFrame from the input sentence
        data = pd.DataFrame([sentence], columns=['sentence'])

        # Phase 1 Inference
        predictions_phase1 = predictor.inference_phase(1, data)

        # Assuming phase 1 predicts which phase 2 model to use
        if predictions_phase1[0] == 0:
            letter = 'A'
        elif predictions_phase1[0] == 1:
            letter = 'B'
        else:
            letter = 'C'

        # Phase 2 Inference
        difficulty_level = predictor.inference_phase(2, data, letter)

        st.write("Predicted Difficulty Level: ",letter,  str(difficulty_level[0] + 1))

if __name__ == "__main__":
    main()
