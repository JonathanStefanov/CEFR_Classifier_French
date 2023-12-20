# Welcome to the CEFR Level Classifier Project! 🚀

[![Run in Saturn Cloud](https://saturncloud.io/images/embed/run-in-saturn-cloud.svg)](https://app.community.saturnenterprise.io/dash/o/community/resources?templateId=1eea18712f1c498b81567ea0e854df87)

This application, built using Streamlit 🌐, leverages the advanced capabilities of a 3-phase Camembert AI model 🧀 to classify texts into the six levels of the Common European Framework of Reference for Languages (CEFR): A1, A2, B1, B2, C1, and C2.

## Video Presentation
[![Video Presentation](https://img.youtube.com/vi/3p9YL150QXU/0.jpg)](https://www.youtube.com/watch?v=3p9YL150QXU)


## What is CEFR? 📘
The CEFR is an internationally recognized standard for describing language ability. It's widely used across the globe to assess and describe the language proficiency of learners.
## Installation via pip 📦
Easily install the CEFR Level Classifier package with pip:
```bash
pip install CEFR-Classifier-French
```
## Example Usage 🌟
Here's a quick example of how to use the `CEFR-Classifier-French` package to predict the CEFR level of a French sentence:

```python
from CEFR_Classifier_French.inference.predict import Predictor

predictor = Predictor()

# Predict the CEFR level of a text
text = "Je ne sais pas quoi dire."

level = predictor.inference_sentence(text)

print("Level of the sentence is -> ", level)
```

## How to Run the GUI 🚀
### On Your Own Computer
1. **Clone the Repository**: 
```
git clone git@github.com:JonathanStefanov/CEFR_Classifier_French.git
```
2. **Navigate to the Folder**: 
```
cd CEFR_Classifier_French
```
3. **Install the Requirements**: 
```
pip install -r requirements.txt
```
4. **Run the Streamlit App**: 
```
streamlit run CEFR_Classifier_French/app.py
```
### On Saturn Cloud
- **Why Use Saturn Cloud?**: Ideal if you don't have a GPU. Offers 10 hours for free.
- **Steps**:
1. Click on the "Run in Saturn Cloud" Button at the top of this README.
2. Create the `CEFR_French` Resource and click on Run. All necessary configurations are pre-set.


## About Our Model 🤖
Our application utilizes the Camembert model, a cutting-edge language processing model, structured in a unique three-phase system to accurately assess and classify texts:
1. **Phase 1 - Initial Classification**: This phase classifies texts into broad categories: A, B, or C.
2. **Phase 2 - Detailed Assessment**: 
   - **Phase 2 A**: Distinguishes between A1 and A2 levels for texts classified as 'A' in Phase 1.
   - **Phase 2 B**: Distinguishes between B1 and B2 levels for texts classified as 'A' in Phase 1.
   - **Phase 2 C**: Distinguishes between C1 and C2 levels for texts classified as 'A' in Phase 1.

This multi-phase approach ensures precise and nuanced classification in line with CEFR standards.

## How to Use the App 🖱️
1. **Navigation**: Use the sidebar to easily navigate through the application.
2. **Training the Model**: Head over to the **Training** section 👨‍🏫. Here, you can train the model with your dataset, allowing it to learn and adapt to your specific language use cases.
3. **Text Classification**: Visit the **Inference** section 🔍 to input text. The app will analyze the text and provide you with its CEFR level classification.

## Our Model's Evolution 🤖
### Initial Attempts
- **Logistic Regression Approach**: We began by analyzing sentence structure - counting length, verbs, punctuation, and checking for passive sentences. Despite these efforts, a logistic regression model yielded unsatisfactory results.

### Transition to Camembert Model
- **First Camembert Trial**: Shifting gears, we implemented a Camembert language model. Although it improved accuracy to 58%, the model's size and training speed were concerning.

### Final, Optimized Model
- **Two-Phase Camembert System**: Our breakthrough came with a refined version of the Camembert model, structured in two phases for precise, efficient classification. This significantly accelerated training times without compromising accuracy. It even increased it to 60,2% with the same dataset.

## Explore More 🔗
Interested in learning more about this project? Looking for source code or detailed documentation? Visit our [GitHub Repository](https://github.com/JonathanStefanov/CEFR_Classifier_French) 🌟 for all the resources you need.

*We hope you enjoy exploring and using our CEFR Level Classifier! Happy Classifying! 🎉*

## Feedback and Contributions
Your feedback is valuable to us! If you have suggestions or want to contribute to this project, please feel free to open an issue or submit a pull request on our GitHub repository. Let's make language learning and classification better, together!

## License
This project is licensed under the GNU General Public License (GPL). This license ensures users have the freedom to share and change all versions of a program to make sure it remains free software for all its users. For more details, see the LICENSE file in the repository.
