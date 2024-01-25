import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from pl_peft_bert import PEFT_BERT_ATTACK
from mlm_cla import BertForMaskedLM

from sklearn.metrics import classification_report

from datasets import load_dataset
import random
import pickle

# Load the pre-trained BERT model and tokenizer
# model_name = "textattack/bert-base-uncased-SST-2"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, output_hidden_states=True).to('cuda:1')
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-CoLA").to('cuda:1')
# model = PEFT_BERT_ATTACK(pretrained_model_name="textattack/bert-base-uncased-SST-2").to('cuda:1')
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-CoLA", padding_side="right")

task = 'cola'

df = load_dataset("glue", task)["train"]
total_samples = len(df)
selected_samples = df

# # Tokenize and preprocess the sentences
# # Encode sentences and labels
# if task in ['sst2', 'cola']:
#     sentences = selected_samples["sentence"]
#     inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
# else:
#     sentence1 = selected_samples["sentence1"]
#     sentence2 = selected_samples["sentence2"]
#     inputs = tokenizer(sentence1, sentence2, padding=True, truncation=True, return_tensors="pt")
labels = selected_samples["label"]
labels = torch.tensor(labels)

# # Create a DataLoader for batch processing
# dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, labels)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Put the model in evaluation mode
model.eval()

# Lists to store predictions and true labels
predictions = []
true_labels = []

# Inference loop
hidden_states = []
msp = []
for example in selected_samples:
    if task in ['sst2', 'cola']:
        inputs = tokenizer(example["sentence"], padding=True, truncation=True, return_tensors="pt").to('cuda:1')
    else:
        inputs = tokenizer(example["sentence1"], example["sentence2"], padding=True, truncation=True, return_tensors="pt").to('cuda:1')
    # input_ids, attention_mask, batch_labels = batch
    with torch.no_grad():
        # outputs = model(input_ids=input_ids.to('cuda:1'), attention_mask=attention_mask.to('cuda:1'), output_hidden_states=True)
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        hidden = outputs.hidden_states[-1][:, 0, :].detach().to('cpu').view([768])
        hidden_states.append(hidden.tolist())
        # print(hidden.shape)

        logits = F.softmax(logits, dim=-1)
        msp.append(torch.max(logits.detach().cpu()).tolist())
        batch_predictions = logits.detach().cpu().argmax(axis=-1).item()
    
    predictions.append(batch_predictions)
    # true_labels.extend(batch_labels.tolist())

# Calculate accuracy or any other evaluation metrics
correct_predictions = sum(p == t for p, t in zip(predictions, labels))
accuracy = correct_predictions / len(labels)
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(labels, predictions))
# print(msp)
# print(len(hidden_states), hidden_states[0], len(msp), max(msp), min(msp))


with open('./Threshold/'+str(task)+'_bert_base_msp.pickle', 'wb') as handle:
    pickle.dump(msp, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./Threshold/'+str(task)+'_bert_base_md.pickle', 'wb') as handle:
    pickle.dump(hidden_states, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df['sentence_prediction'] = predictions
# df['adv_sentence_sst2_msp_md_enable_32_3e-5_4_prediction'] = predictions
# df.to_csv("./sst_adv.csv", index=False)

