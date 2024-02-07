import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
# from my_datacollactor import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

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

from mlm_cla import BertForMaskedLM
# from transformers import BertForMaskedLM
import pickle
from typing import List, Optional, Tuple, Union
from sklearn.metrics import classification_report
import math
from accelerate import Accelerator
import argparse
import torch.nn.functional as F
from scipy.spatial import distance
import numpy as np

from torch.nn import Sigmoid, CosineSimilarity, CrossEntropyLoss
import os
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
accelerator = Accelerator()

## define new loss
class DHLLoss(nn.Module):
    def __init__(self):
        super(DHLLoss, self).__init__()

    def MSP(self, P_adv, P_ori):
        sig = Sigmoid()
        msp = P_adv-P_ori
        msp = 1-sig(msp)
        return msp.mean()

    def mahalanobis(self, u, v, VI):
        delta = u-v
        m = torch.matmul(torch.matmul(delta, VI), delta.T)
        return torch.sqrt(m)[0]


    def MD(self, mean, VI, adv_embed):
        # print(mean, cov, adv_embed)
        md = torch.Tensor([0]).to('cuda')
        for i in range(adv_embed.shape[0]):
            embed = adv_embed[i]
            md += self.mahalanobis(embed.reshape(1, -1), mean, VI)
        # print(cov)
        # print(torch.linalg.pinv(cov))
        # dist = distance.cdist(adv_embed, mean, 'mahalanobis', VI=torch.linalg.pinv(cov))
        # print(dist)
        # md = dist.sum()
        # print(md)
        return torch.log(md/adv_embed.shape[0])

    def SIM(self, adv_embed, ori_embed):
        cos = CosineSimilarity(dim=1, eps=1e-6)
        sim = cos(adv_embed, ori_embed)
        return -sim.reshape(-1, 1).mean()
    
    def forward(self, P_adv, P_ori, mean, VI, adv_embed, ori_embed):
        msp = self.MSP(P_adv, P_ori)
        # print(P_adv, P_ori)
        # print("MSP: ", msp, msp.shape)

        md = self.MD(mean, VI, adv_embed)
        # md = torch.Tensor(md).reshape(-1).to('cuda')
        # print("MD: ", md, md.shape)

        sim = self.SIM(adv_embed, ori_embed)
        # print("SIM: ", sim, sim.shape)

        # dhl = -torch.log((msp+md+sim)/3)
        # dhl = -torch.log((msp+md)/2)
        # dhl = 3*msp+md
        # print("DHL: ", dhl)
        return [msp, md, sim]





class PEFT_BERT_ATTACK(pl.LightningModule):
    def __init__(self, pretrained_model_name, learning_rate=2e-5, **kwas):
        super(PEFT_BERT_ATTACK, self).__init__()
        self.save_hyperparameters()
        self.peft_config = LoraConfig(
            # task_type="SEQ_CLS", 
            # inference_mode=False, 
            r=8,
            lora_alpha=32, 
            lora_dropout=0.1,
            modules_to_save=["base_model.model.transformer.h.3.mlp"],
            )
        self.model = BertForMaskedLM.from_pretrained(
            pretrained_model_name, 
            return_dict=True,
            output_hidden_states=True,
            num_labels=2,
            )
        # self.model.enable_input_require_grads()
        self.model = get_peft_model(
            self.model, 
            self.peft_config
            ).to('cuda')
        # self.model.enable_adapter_layers()

        self.tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2", padding_side="right")
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.learning_rate = learning_rate
        self.validation_step_outputs = []
        ## prepare for new loss
        self.loss_fn = DHLLoss()
        # self.token_embeddings = torch.stack([self.model.bert.get_input_embeddings()(torch.tensor(id).to(torch.device('cuda'))) for token, id in self.tokenizer.get_vocab().items()])

        with open('./MD/SST2_bert_base_sst2_mean.pickle', 'rb') as handle:
            self.mean = pickle.load(handle)
        with open('./MD/SST2_bert_base_sst2_cov.pickle', 'rb') as handle:
            self.cov = pickle.load(handle)
            self.VI = torch.linalg.pinv(self.cov)
        self.train_loss = []
        self.val_loss = []

    def forward(
            self, 
            mlm: Optional[bool] = None,
            input_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            ):
        return self.model(mlm, input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        cls_labels = batch['cls_labels']
        outputs = self(mlm=True, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], 
                        token_type_ids=batch['token_type_ids'], labels=batch['labels'])
        # outputs = self(mlm=True, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], 
        #                 token_type_ids=batch['token_type_ids'])

        # # MLM generated text adv_input_ids
        # input_ids = batch["input_ids"].clone()
        # for i in range(input_ids.shape[0]):
        #     mask_token_index = (input_ids == self.tokenizer.mask_token_id)[i].nonzero(as_tuple=True)[0]
        #     predicted_token_id = outputs.logits[i, mask_token_index].argmax(axis=-1)
        #     input_ids[i, mask_token_index] = predicted_token_id

        # # # MLM generated text adv_input_embeds
        token_embeddings = torch.stack([self.model.bert.get_input_embeddings()(torch.tensor(id).to(torch.device('cuda'))) for id in torch.arange(outputs.logits.shape[-1])])
        # predicted_token_id_rand = F.gumbel_softmax(outputs.logits, hard=True) # gumbel softmax with random
        # predicted_token_id_onehot = F.one_hot(outputs.logits.argmax(axis=-1), outputs.logits.shape[-1])
        # predicted_token_id = (predicted_token_id_onehot-outputs.logits).detach()+outputs.logits # without random
        # adv_input_embeds = torch.matmul(predicted_token_id, token_embeddings)

        adv_input_embeds = torch.stack([self.model.bert.get_input_embeddings()(torch.tensor(id).to(torch.device('cuda'))) for id in batch["input_ids"]])
        for i in range(batch["input_ids"].shape[0]):
            mask_token_index = (batch["input_ids"] == self.tokenizer.mask_token_id)[i].nonzero(as_tuple=True)[0]
            # predicted_token_id = F.gumbel_softmax(outputs.logits[i, mask_token_index], hard=True)
            predicted_token_id_onehot = F.one_hot(outputs.logits[i, mask_token_index].argmax(axis=-1), outputs.logits.shape[-1])
            predicted_token_id = (predicted_token_id_onehot-outputs.logits[i, mask_token_index]).detach()+outputs.logits[i, mask_token_index] # without random
            adv_input_embeds[i, mask_token_index] = torch.matmul(predicted_token_id, token_embeddings)


        # MLM generated text output embeddings
        self.model.disable_adapter_layers()
        cls_outputs = self(mlm=False, inputs_embeds=adv_input_embeds, attention_mask=batch['attention_mask'], 
                        token_type_ids=batch['token_type_ids'], labels=cls_labels)
        output_embeds = cls_outputs.hidden_states[-1]
        logits = cls_outputs.logits
        # model2_outputs = self.model2(input_ids=input_ids)
        # output_embeds = model2_outputs.hidden_states[-1]
        # logits = model2_outputs.logits

        # # self.model.disable_adapter_layers()
        # # cls_outputs = self(False, inputs_embeds=output_embeds)
        # pooled_output = output_embeds[:, 0, :]
        # pooled_output = self.model.dropout(pooled_output)
        # logits = self.model.classifier(pooled_output)

        logits = F.softmax(logits, dim=-1)
        P_adv = logits[torch.arange(len(cls_labels)), (1-cls_labels).long()]
        P_ori = logits[torch.arange(len(cls_labels)), cls_labels.long()]
        adv_embed = output_embeds[:, 0, :]
        # loss = self.compute_loss(P_adv, P_ori, self.mean, self.VI, adv_embed, adv_embed)
        [msp, md, sim] = self.compute_loss(P_adv, P_ori, self.mean, self.VI, adv_embed, adv_embed)
        loss = (md+msp+sim).mean()
        # self.train_loss.append([msp, md, sim])
        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, 2), 1-cls_labels.view(-1))

        self.log('train_loss', loss)

        self.model.enable_adapter_layers()
        # self.model.enable_adapter()
        # self.model.print_trainable_parameters()
        return loss

    def validation_step(self, batch, batch_idx):
        # print("batch['input_ids'].shape: ", batch['input_ids'].shape)
        batch_size = batch['input_ids'].shape[0]
        cls_labels = batch['cls_labels']
        outputs = self(mlm=True, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], 
                        token_type_ids=batch['token_type_ids'], labels=batch['labels'])
        # outputs = self(mlm=True, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], 
        #                 token_type_ids=batch['token_type_ids'])

        # # MLM generated text input_ids and adv_input_embeds
        token_embeddings = torch.stack([self.model.bert.get_input_embeddings()(torch.tensor(id).to(torch.device('cuda'))) for id in torch.arange(outputs.logits.shape[-1])])
        # input_ids = batch["input_ids"].clone()
        adv_input_embeds = torch.stack([self.model.bert.get_input_embeddings()(torch.tensor(id).to(torch.device('cuda'))) for id in batch["input_ids"]])
        for i in range(batch["input_ids"].shape[0]):
            mask_token_index = (batch["input_ids"] == self.tokenizer.mask_token_id)[i].nonzero(as_tuple=True)[0]
            # predicted_token_id = F.gumbel_softmax(outputs.logits[i, mask_token_index], hard=True)
            predicted_token_id_onehot = F.one_hot(outputs.logits[i, mask_token_index].argmax(axis=-1), outputs.logits.shape[-1])
            predicted_token_id = (predicted_token_id_onehot-outputs.logits[i, mask_token_index]).detach()+outputs.logits[i, mask_token_index] # without random
            adv_input_embeds[i, mask_token_index] = torch.matmul(predicted_token_id, token_embeddings)
            # input_ids[i, mask_token_index] = predicted_token_id.argmax(axis=-1)


        # MLM generated text output embeddings
        self.model.disable_adapter_layers()
        # decoded_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        cls_outputs = self(mlm=False, inputs_embeds=adv_input_embeds, attention_mask=batch['attention_mask'], 
                        token_type_ids=batch['token_type_ids'], labels=cls_labels)
        output_embeds = cls_outputs.hidden_states[-1]
        logits = cls_outputs.logits

        prediction = logits.detach().cpu().argmax(axis=-1)

        logits = F.softmax(logits, dim=-1)
        P_adv = logits[torch.arange(len(cls_labels)), (1-cls_labels).long()]
        P_ori = logits[torch.arange(len(cls_labels)), cls_labels.long()]
        adv_embed = output_embeds[:, 0, :]
        # loss = self.compute_loss(P_adv, P_ori, self.mean, self.VI, adv_embed, adv_embed)
        [msp, md, sim] = self.compute_loss(P_adv, P_ori, self.mean, self.VI, adv_embed, adv_embed)
        loss = (md+msp+sim).mean()
        # self.val_loss.append([msp, md, sim])


        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, 2), 1-cls_labels.view(-1))

        # loss += loss_cls

        
        self.log("val_loss", loss)
        self.validation_step_outputs.append([accelerator.gather(loss.repeat(batch_size)), cls_labels, prediction])
        
        self.model.enable_adapter_layers()
        # return loss, cls_labels, prediction

    def on_validation_epoch_end(self):
        losses = torch.cat([output[0] for output in self.validation_step_outputs])
        labels = torch.cat([output[1] for output in self.validation_step_outputs]).detach().cpu().numpy()
        preds = torch.cat([output[2] for output in self.validation_step_outputs])
        # adv_sentences = []
        # for output in self.validation_step_outputs:
        #     adv_sentences.extend(output[3])

        # datasets = load_dataset("glue", args.task)["validation"]
        # df = pd.DataFrame(datasets)
        # if len(adv_sentences) == len(df):
        #     print("\n\n=============\n saving adv_sentences: \n")
        #     df["adv_sentences"] = adv_sentences
        #     df.to_csv("./sst_adv_msp.csv", index=False)
        
        # with open('./sst2_train_loss.pickle', 'wb') as handle:
        #     pickle.dump(self.train_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('./sst2_val_loss.pickle', 'wb') as handle:
        #     pickle.dump(self.val_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # losses = losses[: len(val_datasets)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        report = classification_report(labels, preds)
        print(f">>> Perplexity: {perplexity}")
        print(f">>> Classification Report: {report}")
        self.validation_step_outputs.clear()


    def compute_loss(self, P_adv, P_ori, mean, cov, adv_embed, ori_embed):
        msp, md, sim = self.loss_fn(P_adv, P_ori, mean, cov, adv_embed, ori_embed)
        # loss = md+msp+sim
        # # print("loss: ", loss, loss.shape)
        # # print("msp: ", msp)
        # # print("md: ", md)
        # return loss.mean()
        return [msp, md, sim]
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)





# Initialize and train the model with DistributedDataParallel
def main(args):
    pl.seed_everything(42)

    if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
        

    ## prepare dataset
    datasets = load_dataset("glue", args.task)
    metric = evaluate.load("glue", args.task)


    ## prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        if args.task in ['sst2', 'cola']:
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["sentence"], truncation=True, max_length=512)
        else:
            outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=512)
        outputs['cls_labels'] = examples['label']
        # print(type(examples['label'][0]))
        # print(len(examples["sentence"]))
        return outputs

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"] if args.task in ['sst2', 'cola'] else ["idx", "sentence1", "sentence2"],
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


    ## random mask
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability = args.mask/100)

    train_datasets = tokenized_datasets["train"]
    val_datasets = tokenized_datasets["validation"]

    print(len(train_datasets), len(val_datasets))

    train_dataloader = DataLoader(
        train_datasets, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(
        val_datasets, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size
    )


    model = PEFT_BERT_ATTACK(pretrained_model_name=args.model_name_or_path, learning_rate=args.lr)

    # model.print_trainable_parameters()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.dirpath,
        filename="best_model",
        monitor="train_loss",
        mode="min",
        save_top_k=5,
    )
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=args.log_interval)
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.num_epochs,
        gpus=1,
        # accelerator='cuda',
        callbacks=[checkpoint_callback, progress_bar],
        # callbacks=[progress_bar],
    )
    trainer.fit(model, train_dataloader, eval_dataloader)

    # # Evaluate the model on the test set
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dirpath", type=str, default="./models/")
    parser.add_argument("--num_epochs", type=int, default=1)  # Maximum number of training epochs
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--mask", type=int, default=15)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    print(args)

    main(args)
