import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import re
import json
import urllib.parse
import base64
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertLSTMClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size, output_size, num_layers, bidirectional=True):
        super(BertLSTMClassifier, self).__init__()
        self.bert_model = bert_model
        self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(outputs.last_hidden_state)
        lstm_output = self.dropout(lstm_output)
        logits = self.fc(lstm_output[:, -1, :])  
        return logits


# Define the path to your saved model
model_path = 'bert_lstm_model.pkl'

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Set the device to CPU or GPU based on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Now you can use the model for inference on new data
# Here's a simple example with dummy data
dummy_test_texts = [
    "waitfor delay '00:00:05'",
    "waitfor delay '00:00:05'--",
    "benchmark(50000000,MD5(1))",
    "ORDER BY 1,SLEEP(5),BENCHMARK(1000000,MD5('A')),4",
    "ORDER BY 1,SLEEP(5),BENCHMARK(1000000,MD5('A')),4,5,6,7,8,9,10,11,12,13,14",
    " UNION SELECT @@VERSION,SLEEP(5),USER(),BENCHMARK(1000000,MD5('A')),5,6,7,8,9#",
    "AND 5650=CONVERT(INT,(UNION ALL SELECTCHAR(88)+CHAR(88)))",

]

# Data preprocessing functions
def decode_sql(encoded_string):
    try:
        decoded_string = bytes.fromhex(encoded_string).decode('ascii')
    except:
        pass

    try:
        decoded_string = bytes.fromhex(encoded_string).decode('unicode_escape')
    except:
        pass

    try:
        decoded_string = json.loads(encoded_string)
    except:
        pass

    try:
        decoded_string = urllib.parse.unquote(encoded_string)
    except:
        pass

    try:
        decoded_string = base64.b64decode(encoded_string).decode('utf-8')
    except:
        pass

    return decoded_string

def lowercase_sql(query):
    return query.lower()

def generalize_sql(query):
    generalized_query = re.sub(r'\d+', '0', query)
    return generalized_query

def tokenize_sql(query):
    query = re.sub(r'([<>!=])', r' \1 ', query)
    tokens = query.split()
    return ' '.join(tokens)

# Preprocess the dummy test data similarly to how you preprocessed your training data
def preprocess_text(text):
    text = decode_sql(text)
    text = lowercase_sql(text)
    text = generalize_sql(text)
    text = tokenize_sql(text)
    return text

# Preprocess the dummy test texts
preprocessed_texts = [preprocess_text(text) for text in dummy_test_texts]

# Tokenize the preprocessed texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128
tokenized_texts = [tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt') for text in preprocessed_texts]

# Convert tokenized texts to tensors and move to the appropriate device
input_ids = torch.cat([text['input_ids'] for text in tokenized_texts], dim=0).to(device)
attention_masks = torch.cat([text['attention_mask'] for text in tokenized_texts], dim=0).to(device)

# Perform inference
with torch.no_grad():
    logits = model(input_ids, attention_masks)
    _, predicted = torch.max(logits, dim=1)

# Convert predictions to a list
predictions = predicted.cpu().tolist()

# Print the predictions
for text, prediction in zip(dummy_test_texts, predictions):
    print(f"Text: {text}")
    print(f"Predicted Label: {prediction}")
    print()




