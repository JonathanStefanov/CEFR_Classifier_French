import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from stqdm import stqdm
import os
import requests
import wget

# Define a class for the Predictor
class Predictor:

    def __init__(self, model_path_phase1='phase1.pth', model_path_phase2_A='phase_2_A.pth', model_path_phase2_B='phase_2_B.pth', model_path_phase2_C='phase_2_C.pth'):
        self.device = torch.device("cuda") if torch.cuda.is_available() else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        self.MAX_LEN = 387
        self.BATCH_SIZE = 16
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.model_path_phase1 = self._check_and_download_model(model_path_phase1, phase=1)
        self.model_path_phase2_A = self._check_and_download_model(model_path_phase2_A, phase=2, letter='A')
        self.model_path_phase2_B = self._check_and_download_model(model_path_phase2_B, phase=2, letter='B')
        self.model_path_phase2_C = self._check_and_download_model(model_path_phase2_C, phase=2, letter='C')


    class FrenchSentencesDataset(Dataset):
        def __init__(self, sentences, tokenizer, max_len):
            self.sentences = sentences
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, item):
            sentence = str(self.sentences[item])
            encoding = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            return {
                'sentence_text': sentence,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
        
    def _check_and_download_model(self, model_path, phase, letter=None):
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Downloading...")
            self._download_model(phase, letter)
        return model_path
    
    def _download_model(self, phase, letter=None):
        # Base URL structure
        base_url = "https://github.com/JonathanStefanov/CEFR_Classifier_French/releases/download/Weights/phase"

        # Validate inputs
        if phase not in [1, 2]:
            raise ValueError("Invalid phase")
        if phase == 2 and letter not in ['A', 'B', 'C']:
            raise ValueError("Invalid letter for phase 2")

        # Construct the URL based on the phase and letter
        url = f"{base_url}{phase}" if phase == 1 else f"{base_url}_{phase}" 
        url += f"_{letter}" if letter else ""
        url += ".pth"

        # Download the model
        print(f"Downloading model from {url}...")
        model_path = f"phase_{phase}_{letter}.pth" if letter else f"phase{phase}.pth"
        
        wget.download(url)
        
        print(f"Model downloaded and saved to {model_path}")



    def _predict(self, model, data_loader):
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in stqdm(data_loader, desc="Predicting...", backend=True, frontend=True):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.tolist())

        return predictions
    def inference_phase(self, phase, data, letter=None):
        # Load model
        print(f'Using device: {self.device}')
        print('Loading model...')
        num_labels = 3 if phase == 1 else 2
        model_path = self.model_path_phase1 if phase == 1 else (self.model_path_phase2_A if letter == 'A' else (self.model_path_phase2_B if letter == 'B' else self.model_path_phase2_C))
        model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)

        # Prepare the dataset and data loader
        print('Preparing dataset and data loader...')
        DatasetClass = self.FrenchSentencesDataset
        unseen_dataset = DatasetClass(data['sentence'].reset_index(drop=True), self.tokenizer, self.MAX_LEN)
        unseen_data_loader = DataLoader(unseen_dataset, batch_size=self.BATCH_SIZE)

        # Make predictions
        predictions = self._predict(model, unseen_data_loader)

        # Save predictions to a CSV file
        data['predictions'] = predictions
        # Check if the model path ends with A.pth, B.pth or C.pth
        if phase == 1:
            output_file_path = 'results/predictions_phase1.csv'
        else:
            output_file_path = f'results/predictions_phase2_{model_path[-5]}.csv'

        data.to_csv(output_file_path, index=False)

        return predictions
    
    def inference_sentence(self, sentence):
        data = pd.DataFrame([sentence], columns=['sentence'])
        letter = ''

        # Phase 1 Inference
        predictions_phase1 = self.inference_phase(1, data)

        # Assuming phase 1 predicts which phase 2 model to use
        if predictions_phase1[0] == 0:
            letter = 'A'
        elif predictions_phase1[0] == 1:
            letter = 'B'
        else:
            letter = 'C'

        # Phase 2 Inference
        difficulty_level = self.inference_phase(2, data, letter)

        difficulty = letter + str(difficulty_level[0] + 1)

        return difficulty
