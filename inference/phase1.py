import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

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

def predict(model, data_loader, device):
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


def inference_phase_1(unseen_data_path, output_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Load tokenizer and model
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=3)
    model.load_state_dict(torch.load("phase1.pth", map_location=device))
    model.to(device)

    MAX_LEN = 387
    BATCH_SIZE = 8

    # Load unseen data
    df_unseen = pd.read_csv(unseen_data_path)
    unseen_sentences = df_unseen['sentence'].reset_index(drop=True)

    # Prepare the dataset and data loader
    unseen_dataset = FrenchSentencesDataset(unseen_sentences, tokenizer, MAX_LEN)
    unseen_data_loader = DataLoader(unseen_dataset, batch_size=BATCH_SIZE)

    # Make predictions
    predictions = predict(model, unseen_data_loader, device)

    # Save predictions to a CSV file
    df_unseen['predictions'] = predictions
    df_unseen.to_csv(output_file_path, index=False)



