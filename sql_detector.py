import torch
import pickle
import re
import json
import urllib.parse
import base64
from transformers import BertTokenizer
import torch.nn as nn

# Define your BertLSTMClassifier here
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

def preprocess_text(text):
    text = decode_sql(text)
    text = lowercase_sql(text)
    text = generalize_sql(text)
    text = tokenize_sql(text)
    return text

def detect_sql_injection(input_text, model):
    # Preprocess input text
    preprocessed_text = preprocess_text(input_text)

    # Tokenize the preprocessed text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128
    tokenized_text = tokenizer(preprocessed_text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert tokenized text to tensor and move to device
    input_ids = tokenized_text['input_ids'].to(device)
    attention_mask = tokenized_text['attention_mask'].to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        _, predicted = torch.max(logits, dim=1)

    # Convert prediction to 0 or 1
    prediction = predicted.item()

    return prediction
