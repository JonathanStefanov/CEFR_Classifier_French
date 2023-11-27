import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

class FrenchSentencesDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_data_loader, optimizer, loss_fn, epochs, device, letter):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_data_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_data_loader)
        print(f'Epoch {epoch + 1}/{epochs} - Training loss: {avg_train_loss:.2f}')

    print("Training complete. Saving the model.")
    torch.save(model.state_dict(), "phase_2_" + str(letter) + ".pth")

def train_phase_2(letter):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f'Using device: {device}')

    MAX_LEN = 387
    BATCH_SIZE = 8
    EPOCHS = 3

    # Load the dataset
    url = 'https://storage.googleapis.com/kagglesdsdata/competitions/64188/7030891/training_data.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1701357839&Signature=NG20u%2FBUt4YRLG7S%2BXcYIrHpWLtQboLz%2B1HEfvGLcGail9adb6C2Yt%2F%2Bf%2BctAv3lq8hPwO8RWAAcTunj1hElUR9AO3B%2BygmbJbcJw3P%2FRbO1JcJjRPg4MgUhSnSjUNI2xQWjUklkSTl82ORrqVargteW0N%2FxLcqh%2BRR35wJbPgAec6RyJjHCFOAIFqneVtMqGUnmOniHWoF0wwqWUQfez6eAZWE7SlyulpnV5XNDevBwOGEMNvg6%2FhkI9VsooMHqNH9k92DwqOsa9oxnDtPWBLNzdoiOe4nr0ossnhHIWXv5RrHsTPCt3ERu3RH8wVfLcSARgRuXUD31VXoYxWPKiQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dtraining_data.csv'
    df = pd.read_csv(url)

    wanted_letters = [str(letter) + '1', str(letter) + '2']

    # Filter the dataset to only include wanted levels
    df = df[df['difficulty'].isin(wanted_letters)]

    # Update difficulty mapping for binary classification
    difficulty_mapping = {wanted_letters[0]: 0, wanted_letters[1]: 1}
    df['difficulty_label'] = df['difficulty'].map(difficulty_mapping)

    # Load Camembert tokenizer and model
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
    model.to(device)

    # Use the entire dataset for training
    train_sentences = df['sentence'].reset_index(drop=True)
    train_labels = df['difficulty_label'].reset_index(drop=True)

    train_dataset = FrenchSentencesDataset(train_sentences, train_labels, tokenizer, MAX_LEN)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_data_loader, optimizer, loss_fn, EPOCHS, device, letter)


if __name__ == '__main__':
    train_phase_2('A')
