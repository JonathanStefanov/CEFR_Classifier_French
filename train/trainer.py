import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import get_full_dataset

# Define the FrenchSentencesDataset class
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

# Define the Trainer class
class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.MAX_LEN = 387
        self.BATCH_SIZE = 16
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    def _train_model(self, model, train_data_loader, optimizer, loss_fn, epochs, phase_letter=None):
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch in tqdm(train_data_loader, desc="Training"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_data_loader)
            print(f'Epoch {epoch + 1}/{epochs} - Training loss: {avg_train_loss:.2f}')

        model_filename = "phase1.pth" if phase_letter is None else f"phase_2_{phase_letter}.pth"
        print(f"Training complete. Saving the model to {model_filename}.")
        torch.save(model.state_dict(), model_filename)

    def train_phase_1(self, dataset):
        print(f'Using device: {self.device}')

        EPOCHS = 3
        df = dataset
        df['difficulty'] = df['difficulty'].str[0]
        difficulty_mapping = {'A': 0, 'B': 1, 'C': 2}
        df['difficulty_label'] = df['difficulty'].map(difficulty_mapping)

        train_sentences = df['sentence'].reset_index(drop=True)
        train_labels = df['difficulty_label'].reset_index(drop=True)

        train_dataset = FrenchSentencesDataset(train_sentences, train_labels, self.tokenizer, self.MAX_LEN)
        train_data_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=3)
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()

        self._train_model(model, train_data_loader, optimizer, loss_fn, EPOCHS)

    def train_phase_2(self, letter, dataset):
        print(f'Using device: {self.device}')

        EPOCHS = 2
        df = dataset
        wanted_letters = [str(letter) + '1', str(letter) + '2']
        df = df[df['difficulty'].isin(wanted_letters)]
        difficulty_mapping = {wanted_letters[0]: 0, wanted_letters[1]: 1}
        df['difficulty_label'] = df['difficulty'].map(difficulty_mapping)

        train_sentences = df['sentence'].reset_index(drop=True)
        train_labels = df['difficulty_label'].reset_index(drop=True)

        train_dataset = FrenchSentencesDataset(train_sentences, train_labels, self.tokenizer, self.MAX_LEN)
        train_data_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()

        self._train_model(model, train_data_loader, optimizer, loss_fn, EPOCHS, letter)


# Example usage
trainer = Trainer()
dataset = get_full_dataset()  # Replace with your dataset loading method

# Train Phase 1
print("Training Phase 1")
#trainer.train_phase_1(dataset)

# Train Phase 2
for letter in ["A", "B", "C"]:
    print(f"Training Phase 2 - {letter}")
    trainer.train_phase_2(letter, dataset)
