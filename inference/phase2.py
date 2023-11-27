import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class FrenchSentencesInferenceDataset(Dataset):
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

def _predict(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.tolist())

    return predictions

def inference_phase_2(model_path, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Load tokenizer and model
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    MAX_LEN = 387
    BATCH_SIZE = 8

    # Load unseen data
    unseen_sentences = data['sentence'].reset_index(drop=True)

    # Prepare the dataset and data loader
    unseen_dataset = FrenchSentencesInferenceDataset(unseen_sentences, tokenizer, MAX_LEN)
    unseen_data_loader = DataLoader(unseen_dataset, batch_size=BATCH_SIZE)

    # Make predictions
    predictions = _predict(model, unseen_data_loader, device)

    # Save predictions to a CSV file
    data['predictions'] = predictions
    data.to_csv('phase_2_predictions.csv', index=False)

    return predictions

# Example usage
model_path = 'phase_2_A.pth'  # Replace with your Phase 2 model path
unseen_data_path = 'path_to_unseen_data.csv'  # Replace with your unseen data path
# Map the predictions to the original labels


predictions = inference_phase_2(model_path, unseen_data_path)
print("Predictions saved to phase_2_predictions.csv")
