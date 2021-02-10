import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, BertTokenizer
import seaborn as sns
from multiprocessing import Pool
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import multiprocessing
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import torch.nn.functional as F
import sys


if len(sys.argv) != 4:
    print("Usage: python simulate.py lambda_fair formulation GPU_index")
    exit(1)
lam_fair = float(sys.argv[1])
form = sys.argv[2]
gpui = int(sys.argv[3])


class Config:
    def __init__(self):
        self.PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
        self.NUM_CLASSES = 5
        self.MAX_LENGTH = 512
        self.NUM_CLASSES = 5
        self.RANDOM_SEED=42


config = Config()


df = pd.read_json("yelp_curated_reviews.json", lines=True)


df['stars'] = (df['stars'] >= 4) * 1.0

NUM_CLASSES = 2
tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def count_tokens(df):
    df['text_len'] = df['text'].apply(lambda t : len(tokenizer.encode(t, max_length=config.MAX_LENGTH, truncation=True)))
    return df

df = parallelize_dataframe(df, count_tokens, 10)

genders = pd.read_csv("yelp_users_gender.csv")
gender_dict = {}
for _, row in genders.iterrows():
    gender_dict[row['user_id']] = row['ga_gender']
    
def get_gender(uid):
    gen = gender_dict[uid]
    if gen == 'female':
        return 1
    if gen == 'male':
        return 2
    return 0

class YelpDataset(Dataset):
    def __init__(self, reviews, targets, uids, tokenizer, max_len):
        self.reviews, self.targets,  self.tokenizer, self.max_len = reviews.to_numpy(), targets.to_numpy(), tokenizer, max_len
        self.genders = uids.map(get_gender).to_numpy()

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = self.reviews[item]
        tokens = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'review': review,
            'input_ids': tokens['input_ids'].flatten(),
            'attention_mask': tokens['attention_mask'].flatten(),
            'target': torch.tensor(self.targets[item]).long(),
            'gender': torch.tensor(self.genders[item]).long()
        }

RANDOM_SEED = 42

train_df, test_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED, 
                                     stratify=df.stars.values)
valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=RANDOM_SEED, stratify=test_df.stars.values)
train_df.shape, test_df.shape, valid_df.shape


gen_ints = train_df['uid'].map(get_gender)
(Pmale, Pfem) = ((gen_ints == 2).sum(), (gen_ints == 1).sum())
(Pmale, Pfem) = (Pmale/(Pmale + Pfem), Pfem/(Pmale + Pfem))
assert Pmale > 0
assert Pfem > 0
ploss = nn.BCEWithLogitsLoss()
if form == "logistic":
    def floss(outputs, sens_attr):
        return -lam_fair/outputs.shape[0] * (F.logsigmoid(outputs[sens_attr==2]).sum()/Pmale + F.logsigmoid(-outputs[sens_attr==1]).sum()/Pfem)
elif form == "hinge":
    baseline = torch.tensor(0.).to(device)
    def floss(outputs, sens_attr):
        return lam_fair/outputs.shape[0] * (torch.max(baseline,1-outputs[sens_attr==2]).sum()/Pmale + torch.max(baseline,1+outputs[sens_attr==1]).sum()/Pfem)
else:
    def floss(outputs, sens_attr):
        return lam_fair/outputs.shape[0] * (-outputs[sens_attr==2].sum()/Pmale + outputs[sens_attr==1].sum()/Pfem)


def create_data_loader(df, tokenizer, max_length, batch_size):
    ds = YelpDataset(df['text'], df['stars'], df['uid'], tokenizer, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count() - 1
    )

BATCH_SIZE=8

train_dl = create_data_loader(train_df, tokenizer, config.MAX_LENGTH, BATCH_SIZE)
test_dl = create_data_loader(test_df, tokenizer, config.MAX_LENGTH, BATCH_SIZE)
valid_dl = create_data_loader(valid_df, tokenizer, config.MAX_LENGTH, BATCH_SIZE)

device = torch.device("cuda:" + str(gpui) if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(config.PRE_TRAINED_MODEL_NAME, num_labels = NUM_CLASSES,
                                                      output_attentions = False, output_hidden_states = False)
model = model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-3},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

len([p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]), len([p for n, p in param_optimizer if any(nd in n for nd in no_decay)])

optimizer = AdamW(optimizer_parameters, 
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                 )

EPOCHS = 3

total_steps = len(train_dl) * EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import os
import shutil
import pickle

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

def save_model(model, tokenizer, output_dir = 'model_save'):

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    time.sleep(1)

    print(f"Saving model to {output_dir}")

    tokenizer.save_pretrained(output_dir)
    torch.save(model, f'{output_dir}/model')


def grab_batch_data(batch):
    input_ids = batch['input_ids'].to(device)
    input_mask = batch['attention_mask'].to(device)
    labels = batch['target'].to(device)
    genders = batch['gender'].to(device)
    return input_ids, input_mask, labels, genders

def eval_fn():
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in tqdm(valid_dl, total=len(valid_dl)):
        input_ids, input_mask, labels, genders = grab_batch_data(batch)
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            output = model(input_ids,
                                   token_type_ids=None,
                                   attention_mask=input_mask,
                                   labels=labels)
            _, logits = output[0], output[1]
            flat_logits = logits[:,1] - logits[:,0]
            loss = ploss(flat_logits, labels.float()) + floss(flat_logits, genders)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(valid_dl)
    avg_val_loss = total_eval_loss / len(valid_dl)
    print(f"  Accuracy: {avg_val_accuracy:.2f}")
    print(f"  Validation Loss: {avg_val_loss:.2f}")
    return {
        'Accuracy': avg_val_accuracy,
        'Validation Loss': avg_val_loss
    }

training_stats = []
best_accuracy = 0
def train():
    training_stats = []
    best_accuracy = 0
    for epoch in tqdm(range(0, EPOCHS), total=EPOCHS):
        total_train_loss = 0
        model.train()
        for step, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        # for step, batch in enumerate(train_dl):
            input_ids, input_mask, labels, genders = grab_batch_data(batch)
            model.zero_grad()
            output = model(input_ids,
                             token_type_ids=None,
                             attention_mask=input_mask,
                             labels=labels)
            _, logits = output[0], output[1]
            flat_logits = logits[:,1] - logits[:,0]
            loss = ploss(flat_logits, labels.float()) + floss(flat_logits, genders)
            total_train_loss += loss.item()
            if step % 2000 == 0:
                print(f'{step}: Loss: {total_train_loss / (step + 1)}\r')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dl)
        print("")
        print(f"  Average training loss: {avg_train_loss}")
        stats_info = eval_fn()
        stats_info['epoch'], stats_info['Average training loss'] = epoch, avg_train_loss
        training_stats.append(stats_info)
        save_model(model, tokenizer, output_dir=f'{form}{lam_fair}/output_dir_{epoch}') #TODO check this
        if stats_info['Accuracy'] > best_accuracy:
            save_model(model, tokenizer, output_dir=f'{form}{lam_fair}/output_dir_best')
            best_accuracy = stats_info['Accuracy']

train()

torch.cuda.empty_cache()

def eval_fn_test():
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
   
def flat_ddp(preds, gender):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return torch.tensor([pred_flat[gender==1].sum(), pred_flat[gender==1].shape[0],
                         pred_flat[gender==2].sum(), pred_flat[gender==2].shape[0]]) #msmiling, m, fsmiling, f



def eval_fn_test():
    total_eval_accuracy = 0
    total_eval_loss = 0
    total_eval_ddp = 0
    nb_eval_steps = 0
    
    # Evaluate data for one epoch
    for batch in tqdm(test_dl, total=len(test_dl)):
        input_ids, input_mask, labels, genders = grab_batch_data(batch) #TODO add _
        with torch.no_grad():        
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            output = model(input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=input_mask,
                                   labels=labels)
            (loss, logits) = (output[0], output[1])
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            gender_ids = genders.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            total_eval_ddp = total_eval_ddp + flat_ddp(logits, gender_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_dl)
    avg_val_loss = total_eval_loss / len(test_dl)
    ddp = total_eval_ddp[2]/total_eval_ddp[3] - total_eval_ddp[0]/total_eval_ddp[1]
    print(f"  Accuracy: {avg_val_accuracy:.2f}")
    print(f"  Validation Loss: {avg_val_loss:.2f}")
    print(f"  DDP: {ddp:.2f}")

    return {
        'Accuracy': [avg_val_accuracy],
        'Validation Loss': [avg_val_loss],
        'DDP': [total_eval_ddp]
    }

test_res = eval_fn_test()
print(test_res)
res_dict = pd.DataFrame.from_dict(test_res)
res_dict.to_csv(f'{form}{lam_fair}/summary.csv')
