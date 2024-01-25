import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from pl_peft_bert import PEFT_BERT_ATTACK
from mlm_cla import BertForMaskedLM

from sklearn.metrics import classification_report

# Load the pre-trained BERT model and tokenizer
# model_name = "textattack/bert-base-uncased-SST-2"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, output_hidden_states=True).to('cuda:1')
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", output_hidden_states=True).to('cuda:1')
# model = PEFT_BERT_ATTACK(pretrained_model_name="textattack/bert-base-uncased-CoLA").to('cuda:1')
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2", padding_side="right")


# Load your CSV dataset
task = 'sst2'
csv_file = "./data/"+str(task)+"/peft/log_011_128_1e4_20_30_0.csv"
# csv_file = "cola_adv.csv"
df = pd.read_csv(csv_file)

# Tokenize and preprocess the sentences
# sentences = df["original_text"].tolist()
sentences = df["perturbed_text"].tolist()
labels = df["ground_truth_output"].tolist()

# # Encode sentences and labels
# inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
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
for example in sentences:
    if task in ['sst2', 'cola']:
        inputs = tokenizer(example, padding=True, truncation=True, return_tensors="pt").to('cuda:1')
    else:
        example_split = example.split("<SPLIT>")
        sent1 = example_split[0].replace("Sentence1:", "").strip()
        sent2 = example_split[1].replace("Sentence2:", "").strip()
        inputs = tokenizer(sent1, sent2, padding=True, truncation=True, return_tensors="pt").to('cuda:1')
    with torch.no_grad():
        # output_embeds = model(input_ids, attention_mask=attention_mask).hidden_states[-1]
        # outputs = model(inputs_embeds=output_embeds, attention_mask=attention_mask)

        # input_embeds = torch.stack([model.bert.get_input_embeddings()(torch.tensor(id).to(torch.device('cuda:1'))) for id in input_ids])
        # outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask.to('cuda:1'))
        # logits = outputs.logits

        outputs = model(**inputs)
        logits = outputs.logits
        hidden = outputs.hidden_states[-1].detach().cpu().view([-1, 768]).tolist()
        hidden_states.append(hidden)
        # print(hidden)
        # break

        # # MLM generated text output embeddings
        # output_embeds = model(input_ids=input_ids, attention_mask=attention_mask, 
        #                 ).hidden_states[-1]

        # pooled_output = output_embeds[:, 0, :]
        # pooled_output = model.dropout(pooled_output)
        # logits = model.classifier(pooled_output)
    logits = F.softmax(logits, dim=-1)
    batch_predictions = logits.detach().cpu().argmax(axis=-1)
    
    predictions.extend(batch_predictions)
    # true_labels.extend(batch_labels.tolist())

print(len(hidden_states))
torch.save(hidden_states, "./data/"+str(task)+"/peft/all_last_hidden_states_perturbed.pt")
# torch.save(hidden_states, "./data/"+str(task)+"/peft/all_last_hidden_states_original.pt")
# Calculate accuracy or any other evaluation metrics
correct_predictions = sum(p == t for p, t in zip(predictions, labels))
accuracy = correct_predictions / len(labels)
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(labels, predictions))

# df['sentence_prediction'] = predictions
# df['adv_sentence_sst2_msp_md_enable_32_3e-5_4_prediction'] = predictions
# df.to_csv("./sst_adv.csv", index=False)

