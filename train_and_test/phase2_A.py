import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

MAX_LEN = 387
BATCH_SIZE = 8
EPOCHS = 3

# Load the dataset
url = 'https://storage.googleapis.com/kagglesdsdata/competitions/64188/7030891/training_data.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1700944126&Signature=fWtTy0APW1%2B0d9ad9px0qP4Y%2BUY2%2BYO0LRnD%2F8SHZTXCtFnLDWkDLePTmWL%2BOVlyeHrfIxR6vas2dSmxrmfEzJl1r0zrTudfdzI3vFaAxm25l%2BG5WOJyYKEPYNNoAPLsvGRg6cY3wnVQ844M7vrXJ7ryAS13iji9TII3BCBbFoFOFAQ15kG7BPdKxtI1basBFZmSnK9lbAKjfFB9uA6iWdUGvtAb3PE2J0M2rXjlVcaGqp6Yvu04bOaSryt0c66WZLa9FZv70pNd3RLnn7g7LWxi%2BGtpNDUXE71WHAFOBSnctum4TH%2Fb5Z0aCU1xlvGUhz8byJA%2FP8kvPCX3Cb7eTQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dtraining_data.csv'
df = pd.read_csv(url)

# Filter the dataset to only include A1 and A2 levels
df = df[df['difficulty'].isin(['A1', 'A2'])]

# Update difficulty mapping for binary classification
difficulty_mapping = {'A1': 0, 'A2': 1}
df['difficulty_label'] = df['difficulty'].map(difficulty_mapping)

# Update the model (still using Camembert, but with 2 output labels)
model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
model.to(device)

# Update the dataset splitting to use the filtered dataframe
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    df['sentence'], df['difficulty_label'], test_size=0.2, random_state=42)

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

# Load Camembert tokenizer and model
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
model.to(device)
# Train and evaluate the updated model
train_sentences = train_sentences.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_sentences = val_sentences.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

train_dataset = FrenchSentencesDataset(train_sentences, train_labels, tokenizer, MAX_LEN)
val_dataset = FrenchSentencesDataset(val_sentences, val_labels, tokenizer, MAX_LEN)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training and evaluation function
def train_and_evaluate(model, train_data_loader, val_data_loader, optimizer, loss_fn, epochs, device):
    best_val_accuracy = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Training loop
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

        # Validation loop
        model.eval()
        total_val_accuracy = 0
        for batch in tqdm(val_data_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)

            total_val_accuracy += torch.sum(preds == labels).item()

        avg_val_accuracy = total_val_accuracy / len(val_dataset)
        print(f'Epoch {epoch + 1}/{epochs} - Validation Accuracy: {avg_val_accuracy:.2f}')

        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy

    print(f'Best Validation Accuracy: {best_val_accuracy:.2f}')
    print("Saving the model")

    torch.save(model.state_dict(), "phase2_A_model.pth")

# Train and evaluate the model
train_and_evaluate(model, train_data_loader, val_data_loader, optimizer, loss_fn, EPOCHS, device)
