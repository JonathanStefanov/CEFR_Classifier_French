import streamlit as st
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_full_dataset
from train.trainer import Trainer

def save_uploadedfile(uploadedfile):
    with open(os.path.join("datasets", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to datasets/".format(uploadedfile.name))

def main():
    st.title("Model Training Interface")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # To read and display the uploaded file
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        # Save the uploaded file to 'datasets/' directory
        save_uploadedfile(uploaded_file)

    # Advanced Settings Inputs
    lr_exp = st.number_input('Learning Rate Exponent', value=-5)
    batch_size = st.number_input('Batch Size', value=16, min_value=1)
    epochs = st.number_input('Epochs', value=3, min_value=1)

    # Start Training Button
    if st.button('Start Training'):
        # Instantiate Trainer with selected parameters
        trainer = Trainer(lr_phase_1=10**lr_exp, lr_phase_2=10**lr_exp, batch_size=batch_size, epochs_phase_1=epochs, epochs_phase_2=epochs)

        # Load dataset
        dataset = get_full_dataset()


        # Training Process
        st.write("Training Phase 1")
        trainer.train_phase_1(dataset)
        for letter in ["A", "B", "C"]:
            dataset = get_full_dataset()

            st.write(f"Training Phase 2 - {letter}")
            trainer.train_phase_2(letter, dataset)
            st.write(f"Training Phase 2 - {letter} Complete")
        
        st.write("Training Complete")

if __name__ == "__main__":
    main()

