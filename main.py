#Rajeev Bhavin Botadra
#---------------------------Finetuning Language Models Project---------------------------#
#Milestone-3: Fine-Tuning a Language Model
#This code loads and finetunes a pretrained roBERTa model for toxicity analysis of tweets

#Imports
#We use the transformers library to instantiate pretrained language models and tune them on the data
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

#Download and load model and tokenizer
#Define Model name, we will use the roberta model with six labels
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
#Note that we donwnload a model with a differnt number of classes
#Consequently the loader will destroy the head layer and create a new randomly instantiated head
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=6, ignore_mismatched_sizes=True)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    if isinstance(text, str):
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    else: #Handle case where text is list
        print(text)
        return [preprocess(t) for t in text]


#Import training data and convert to pytorch dataset
#We define a class to map the dataframes into iterable pytorch datasets
class ToxicityDataset(Dataset):
    def __init__(self, x, y, tokenizer, max_len):
        self.x_data = x
        self.y_data = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        input = self.x_data[idx]
        labels = self.y_data[idx]

        # Preprocess input
        input = preprocess(input)

        # Tokenize input using AutoTokenizer, use padding+truncation so all inputs are equal length
        encoding = self.tokenizer.encode_plus(input, return_tensors='pt', max_length=self.max_len, 
                                              padding='max_length', truncation=True)

        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


training_args = TrainingArguments(output_dir = "./train_checkpoints", evaluation_strategy="epoch")

df = pd.read_csv("train.csv")

#Train set
x_train = df["comment_text"]
x_train = x_train.values.tolist()
x_train_subset = x_train[0:239]

y_train = df.drop(["id","comment_text"], axis=1)
y_train = y_train.values.tolist()
y_train_subset = y_train[0:239]

#Test set
x_test = pd.read_csv("test.csv")
x_test = x_test["comment_text"]
x_test = x_test.values.tolist()
x_test_subset = x_test[0:29]

y_test = pd.read_csv("test_labels.csv")
y_test = y_test.drop(["id"], axis=1)
y_test = y_test.values.tolist()
y_test_subset = y_test[0:29]

train_dataset = ToxicityDataset(x_train_subset, y_train_subset, tokenizer, max_len=512)
test_dataset = ToxicityDataset(x_test_subset, y_test_subset, tokenizer, max_len=512)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, drop_last=True)

#Define training batch
#batch = tokenizer(x_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
#batch_subset = tokenizer(x_train_subset, padding=True, truncation=True, max_length=512, return_tensors="pt")


trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset)

trainer.train()

#Save model after training
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)
