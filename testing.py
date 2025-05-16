import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
from bs4 import BeautifulSoup

df_trump = pd.read_json('./data/tweets.json')

model_path = "bert-finetuned-hate-speech"

# Load tokenizer and model
trained_model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def clean_html(raw_html):
    # Parse HTML
    soup = BeautifulSoup(raw_html, "html.parser")
    # Get plain text
    return soup.get_text(separator=" ", strip=True)

df_trump['isRetweet'] = df_trump['isRetweet'].astype(bool)
df_trump['text'] = df_trump['text'].apply(lambda x: clean_html(x))
# remove all rows with empty text
df_trump = df_trump[df_trump['text'].str.strip() != '']

print(df_trump.head(10))

dataset_trump = DatasetDict({
    "trump": Dataset.from_pandas(df_trump)
})
# dataset_trump.set_format("torch")
# dataset_trump

def process(examples):
    # take a batch of texts
    text = examples["text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    return encoding

dataset_trump = dataset_trump.map(process, batched=True, remove_columns=dataset_trump["trump"].column_names)
dataset_trump.set_format("torch")

# Put model in eval mode
trained_model.eval()

# Create a DataLoader for the test dataset
trump_loader = DataLoader(dataset_trump['trump'], batch_size=8)

trump_results = pd.DataFrame(columns=['Text', 'Predicted_Values'])

print("Starting prediction...")

# Iterate through the test dataset and make predictions
for batch in trump_loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

    with torch.no_grad():
        outputs = trained_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.sigmoid(logits) > 0.5
        # Convert predictions to integers (1 or 0)
        predictions = predictions.int()
        # predictions = torch.sigmoid(logits)

        batch_results = []
        for text, pred in zip(decoded_texts, predictions.numpy()):
            batch_results.append({'Text': text, 'Predicted_Values': list(pred)})

        # Convert the list of results to a DataFrame
        batch_df = pd.DataFrame(batch_results)
        print(f"Processed {len(trump_results) + len(batch_df)} rows so far.")

        # Concatenate the batch DataFrame with the main results DataFrame
        trump_results = pd.concat([trump_results, batch_df], ignore_index=True)

print(trump_results.head(10))
    
# Save the results to a CSV file
trump_results.to_csv('trump_results.csv', index=False)