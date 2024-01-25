import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

import evaluate
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

import random
from accelerate import Accelerator
import math
# from mlm_cla import BertForMaskedLM
# from transformers import BertForMaskedLM
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from torch.nn import Sigmoid, CosineSimilarity
import pickle


## parameters
batch_size = 1
model_name_or_path = "textattack/bert-base-uncased-WNLI"
task = "wnli"
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 1


if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

## prepare dataset
datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)

## prepare tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    if task in ['sst2', 'cola']:
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
    else:
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence"] if task in ['sst2', 'cola'] else ["idx", "sentence1", "sentence2"],
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

train_datasets = tokenized_datasets["train"]


## prepare dataloader
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

train_dataloader = DataLoader(
    train_datasets, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, 
    return_dict=True,
    output_hidden_states=True,
    )


# initialize accelerator for training
accelerator = Accelerator()
model,  train_dataloader = accelerator.prepare(model, train_dataloader)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_epochs * num_update_steps_per_epoch
print("num_update_steps_per_epoch: ", num_update_steps_per_epoch)

## training
progress_bar = tqdm(range(num_training_steps))
embeddings = []
# with torch.no_grad():
model.eval()
i = 0
for batch in train_dataloader:
    # print(batch)
    with torch.no_grad():
        outputs = model(**batch)
        hidden_states = outputs.hidden_states
        cls_embedding = hidden_states[-1][:, 0, :]
        embeddings.append(cls_embedding)
        progress_bar.update(1)
        i += 1
        # if i == 1000:
        #     break

# print(embeddings, len(embeddings), embeddings[0].shape)
embeddings = torch.cat(embeddings, 0)
print(embeddings.shape)

mean = torch.mean(embeddings, axis=0).reshape(1, -1).to('cuda')
cov = torch.cov(embeddings.T).to('cuda')
print(mean.shape, cov.shape)
with open('./MD/'+str(task).upper()+'_bert_base_'+str(task)+'_mean.pickle', 'wb') as handle:
    pickle.dump(mean, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./MD/'+str(task).upper()+'_bert_base_'+str(task)+'_cov.pickle', 'wb') as handle:
    pickle.dump(cov, handle, protocol=pickle.HIGHEST_PROTOCOL)

