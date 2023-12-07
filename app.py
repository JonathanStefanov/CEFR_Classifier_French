import streamlit as st

st.set_page_config(page_title="CEFR Level Classifier", page_icon="🌍")

st.title("CEFR Level Classifier with Camembert Model 🧀")

# Markdown description with GitHub link and emojis
st.markdown("""
## Welcome to the CEFR Level Classifier Project! 🚀

This Streamlit application 🌐 utilizes a sophisticated 3-phase Camembert model 🧀 to classify texts into the six Common European Framework of Reference for Languages (CEFR) levels: A1, A2, B1, B2, C1, and C2.

### What is CEFR? 📘
The Common European Framework of Reference for Languages (CEFR) is an international standard for describing language ability. It is used around the world to describe learners' language skills.

### About the Model 🤖
The Camembert model, a state-of-the-art language model, is adapted here to understand and classify text based on language complexity and proficiency levels. The model is trained and fine-tuned in three phases to accurately determine the CEFR level of any given text.

### How to Use This App 🖱️
- Navigate through the app using the sidebar.
- Go to the **Training** section 👨‍🏫 to train the model with your dataset.
- Visit the **Inference** section 🔍 to input text and get the CEFR level classification.

### Explore More 🔗
For more details about the project, source code, and documentation, visit our [GitHub Repository](https://github.com/yourusername/CEFR-Level-Classifier) 🌟.

*Happy Exploring and Classifying! 🎉*
""")
