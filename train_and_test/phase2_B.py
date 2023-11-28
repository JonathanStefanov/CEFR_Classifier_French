import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import os 
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

MAX_LEN = 387
BATCH_SIZE = 16
EPOCHS = 5

# Load the dataset
def merge_clean_datasets(*urls):
    required_columns = {'sentence', 'difficulty'}
    dataframes = []

    for url in urls:
        df = pd.read_csv(url)

        # Check if the dataframe contains the required columns
        if not required_columns.issubset(df.columns):
            raise ValueError(f"The dataset from {url} does not contain the required columns.")

        dataframes.append(df)

    # Merge the dataframes
    df_merged = pd.concat(dataframes, ignore_index=True)

    # Remove duplicate sentences
    df_merged = df_merged.drop_duplicates(subset='sentence', keep='first')

    return df_merged


def get_file_paths(directory):
    """
    Returns a list of file paths for all files in the given directory.
    
    :param directory: The directory to search for files.
    :return: A list of file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    
    return file_paths

def get_full_dataset():
    """
    Returns the full dataset.
    
    :return: The full dataset.
    """
    # Get the file paths for all files in the data directory
    file_paths = get_file_paths('datasets/')
    
    # Merge the datasets
    df = merge_clean_datasets(*file_paths)
    
    return df

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f'Using device: {device}')

MAX_LEN = 387
BATCH_SIZE = 16
EPOCHS = 5

# Load the dataset
df = get_full_dataset()

# Filter the dataset to only include A1 and A2 levels
df = df[df['difficulty'].isin(['B1', 'B2'])]

# Update difficulty mapping for binary classification
difficulty_mapping = {'B1': 0, 'B2': 1}
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

    torch.save(model.state_dict(), "phase2_B_model.pth")

# Train and evaluate the model
train_and_evaluate(model, train_data_loader, val_data_loader, optimizer, loss_fn, EPOCHS, device)
