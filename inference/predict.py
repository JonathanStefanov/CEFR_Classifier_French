import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Define a class for the Predictor
class Predictor:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.MAX_LEN = 387
        self.BATCH_SIZE = 16
        self.tokenizer = AutoTokenizer.from_pretrained("almanach/camemberta-base")

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

    def _predict(self, model, data_loader):
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.tolist())

        return predictions

    def inference_phase(self, phase, model_path, data):
        # Load model
        num_labels = 3 if phase == 1 else 2
        model = AutoModelForSequenceClassification.from_pretrained("almanach/camemberta-base", num_labels=num_labels)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)

        # Prepare the dataset and data loader
        DatasetClass = self.FrenchSentencesDataset
        unseen_dataset = DatasetClass(data['sentence'].reset_index(drop=True), self.tokenizer, self.MAX_LEN)
        unseen_data_loader = DataLoader(unseen_dataset, batch_size=self.BATCH_SIZE)

        # Make predictions
        predictions = self._predict(model, unseen_data_loader)

        # Save predictions to a CSV file
        data['predictions'] = predictions
        # Check if the model path ends with A.pth, B.pth or C.pth
        if phase == 1:
            output_file_path = 'inference/predictions_phase1.csv'
        else:
            output_file_path = f'inference/predictions_phase2_{model_path[-5]}.csv'

        data.to_csv(output_file_path, index=False)

        return predictions

# Example usage
predictor = Predictor()
unseen_data_path = 'kaggle/unlabelled_test_data.csv'  # Replace with your unseen data path
model_path_phase1 = 'phase1.pth'  # Replace with your Phase 1 model path

# Phase 1 Inference
df = pd.read_csv(unseen_data_path)
predictions_phase1 = predictor.inference_phase(1, model_path_phase1, df)

# Further processing based on Phase 1 predictions
df_A = df[df['predictions'] == 0].reset_index(drop=True)
df_B = df[df['predictions'] == 1].reset_index(drop=True)
df_C = df[df['predictions'] == 2].reset_index(drop=True)

# Phase 2 Inference
model_path_A = 'phase_2_A.pth'  # Replace with your Phase 2 model path
model_path_B = 'phase_2_B.pth' 
model_path_C = 'phase_2_C.pth'  

predictor.inference_phase(2, model_path_A, df_A)
predictor.inference_phase(2, model_path_B, df_B)
predictor.inference_phase(2, model_path_C, df_C)

print("Predictions saved to inference directory.")
