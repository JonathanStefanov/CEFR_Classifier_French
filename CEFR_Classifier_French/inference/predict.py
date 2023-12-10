import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from stqdm import stqdm
import os
import requests
import wget

class FrenchSentencesDataset(Dataset):
    """
    A PyTorch Dataset class for handling French sentences. It tokenizes the sentences using
    a specified tokenizer and prepares them for model input.

    Attributes:
    sentences (list): A list of sentences to be tokenized and processed.
    tokenizer (CamembertTokenizer): The tokenizer used for tokenizing the sentences.
    max_len (int): The maximum length of tokens for a single sentence.
    """
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
    
# Define a class for the Predictor
class Predictor:
    """
    Predictor class for determining the difficulty level of French sentences.

    This class handles the loading of pre-trained models and predicts the difficulty level
    of given sentences in two phases.

    Attributes:
    device (torch.device): Device to run the model (CPU, CUDA, MPS).
    MAX_LEN (int): Maximum length of tokens for a sentence.
    BATCH_SIZE (int): Batch size for model prediction.
    tokenizer (CamembertTokenizer): Tokenizer for processing the text.
    model_path_phase1, model_path_phase2_A, model_path_phase2_B, model_path_phase2_C (str): Paths to the model weights.
    """
    def __init__(self, model_path_phase1='phase1.pth', model_path_phase2_A='phase_2_A.pth', model_path_phase2_B='phase_2_B.pth', model_path_phase2_C='phase_2_C.pth'):
        self.device = torch.device("cuda") if torch.cuda.is_available() else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        self.MAX_LEN = 387
        self.BATCH_SIZE = 16
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.model_path_phase1 = self._check_and_download_model(model_path_phase1, phase=1)
        self.model_path_phase2_A = self._check_and_download_model(model_path_phase2_A, phase=2, letter='A')
        self.model_path_phase2_B = self._check_and_download_model(model_path_phase2_B, phase=2, letter='B')
        self.model_path_phase2_C = self._check_and_download_model(model_path_phase2_C, phase=2, letter='C')


        
    def _check_and_download_model(self, model_path, phase, letter=None):
        """
        Checks if the model file exists locally, and if not, initiates a download.

        Args:
        model_path (str): The path where the model is expected to be found or saved.
        phase (int): The phase of the model (1 or 2).
        letter (str, optional): The letter specifying the submodel in phase 2 (A, B, C).
        """
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Downloading...")
            self._download_model(phase, letter)
        return model_path
    
    def _download_model(self, phase, letter=None):
        """
        Downloads the model from a predefined URL.

        Args:
        phase (int): The phase of the model (1 or 2).
        letter (str, optional): The letter specifying the submodel in phase 2 (A, B, C).

        Raises:
        ValueError: If an invalid phase or letter is provided.
        """
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
        """
        Makes predictions on the provided dataset using the specified model.

        Args:
        model (CamembertForSequenceClassification): The loaded model used for predictions.
        data_loader (DataLoader): DataLoader containing the dataset for prediction.

        Returns:
        list: A list of predicted labels for the input data.
        """
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
        """
        Conducts the inference in the specified phase (phase 1 or phase 2 with submodels A, B, C).

        Args:
        phase (int): The phase of the model (1 or 2).
        data (pd.DataFrame): DataFrame containing the sentences for inference.
        letter (str, optional): The letter specifying the submodel in phase 2 (A, B, C).

        Returns:
        list: Predicted difficulty levels of the sentences.
        """
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
        DatasetClass = FrenchSentencesDataset
        unseen_dataset = DatasetClass(data['sentence'].reset_index(drop=True), self.tokenizer, self.MAX_LEN)
        unseen_data_loader = DataLoader(unseen_dataset, batch_size=self.BATCH_SIZE)

        # Make predictions
        predictions = self._predict(model, unseen_data_loader)

        # Save predictions to a CSV file
        data['predictions'] = predictions
        # Check if the model path ends with A.pth, B.pth or C.pth

        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
            
        if phase == 1:
            output_file_path = 'results/predictions_phase1.csv'
        else:
            output_file_path = f'results/predictions_phase2_{model_path[-5]}.csv'

        data.to_csv(output_file_path, index=False)

        return predictions
    
    def inference_sentence(self, sentence):
        """
        Infers the difficulty level of a single French sentence.

        This method first uses the phase 1 model to determine which phase 2 model to use,
        then predicts the difficulty using the chosen phase 2 model.

        Args:
        sentence (str): The French sentence for which the difficulty level is to be predicted.

        Returns:
        str: The predicted difficulty level of the sentence.
        """
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
