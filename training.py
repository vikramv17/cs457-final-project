import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch

df = pd.read_parquet('./data/train-data.parquet')
df_trump = pd.read_json('./data/tweets.json')

# map all columns with true/false labels to 1 for true and 0 for false
def map_labels(df):
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        elif np.issubdtype(df[col].dtype, np.floating):
            df[col] = (df[col] > 2).astype(int)
    return df

# drop all columns with annotator in the name
df = df.loc[:, ~df.columns.str.contains('annotator')]
df = df.drop(columns=["infitms", "outfitms", "std_err", "hypothesis", "platform", "hate_speech_score"])
df = map_labels(df)
# train test split
# df_train, df_test = train_test_split(df, test_size=0.4, random_state=42)
# df_test, df_dev = train_test_split(df_test, test_size=0.5, random_state=42)
df_extra, df_train = train_test_split(df, test_size=0.01, random_state=42)
df_train, df_test = train_test_split(df_train, test_size=0.4, random_state=42)
df_test, df_dev = train_test_split(df_test, test_size=0.5, random_state=42)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_dev = df_dev.reset_index(drop=True)

dataset = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "test": Dataset.from_pandas(df_test),
    "dev": Dataset.from_pandas(df_dev)
})

labels = [label for label in dataset["train"].features.keys() if label not in ['comment_id', 'text']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

X_train = df_train["text"].reset_index()
y_train = df_train.drop(columns=["text"])

max_length = X_train["text"].str.len().max()
print(f"Maximum length of text: {max_length}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset["train"].column_names)

encoded_dataset.set_format("torch")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

batch_size = 8
metric_name = "f1"

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"bert-finetuned-hate-speech",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["dev"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# trainer.train()

# save the weights
# trainer.save_model("bert-finetuned-hate-speech")