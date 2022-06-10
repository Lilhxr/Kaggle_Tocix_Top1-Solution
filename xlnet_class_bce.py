import os
import gc
import copy
import time
import random
import string

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Utils
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

# For Transformer Models
from transformers import AutoTokenizer, AutoModel, AdamW
from transformers import  XLNetTokenizer, XLNetModel, TFXLNetModel, XLNetLMHeadModel, XLNetConfig, XLNetForSequenceClassification
# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import wandb

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def id_generator(size=12, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

HASH_NAME = 'length_128_class_bce_xlnet'
print(HASH_NAME)
HASH_NAME = '1202_' + HASH_NAME

CONFIG = {"seed": 2021,
          "epochs": 2,
          "model_name": "../input/xlnet-base-cased",
          "train_batch_size": 32,
          "valid_batch_size": 64,
          "max_length": 128,
          "learning_rate": 2e-5,
          "scheduler": 'OneCycleLR',
          "max_lr": 7e-5,
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 2,
          "num_classes": 1,
          "margin": 0.5,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          "hash_name": HASH_NAME,
          'pct_start': 0.1,               # OneCycleLR
          'anneal_strategy': 'cos',       # OneCycleLR
          'div_factor': 1e2,             # OneCycleLR
          'final_div_factor': 1e2,        # OneCycleLR
          'no_decay': True,
          'frac_1': 0.7,
          'frac_1_factor': 1.5
          }

# CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['model_name'])
# CONFIG['group'] = f'{HASH_NAME}-Baseline'

CONFIG['tokenizer'] = XLNetTokenizer.from_pretrained('../input/xlnetbasecased/xlnet_cased_L-12_H-768_A-12')
config = XLNetConfig.from_pretrained('../input/xlnet-base-cased')
CONFIG['group'] = f'{HASH_NAME}-Baseline'

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CONFIG['seed'])

class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
#         self.more_toxic = df['more_toxic'].values
#         self.less_toxic = df['less_toxic'].values
        self.text = df['text'].values
        self.y = df['y'].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
#         more_toxic = self.more_toxic[index]
#         less_toxic = self.less_toxic[index]
        text = self.text[index]
#         inputs_more_toxic = self.tokenizer.encode_plus(
#                                 more_toxic,
#                                 truncation=True,
#                                 add_special_tokens=True,
#                                 max_length=self.max_len,
#                                 padding='max_length'
#                             )
#         inputs_less_toxic = self.tokenizer.encode_plus(
#                                 less_toxic,
#                                 truncation=True,
#                                 add_special_tokens=True,
#                                 max_length=self.max_len,
#                                 padding='max_length'
#                             )
        inputs_toxic = self.tokenizer.encode_plus(
                                text,
                                truncation=True,
                                add_special_tokens=True,
                                max_length=self.max_len,
                                padding='max_length'
                            )
    
        target = self.y[index]
        
#         more_toxic_ids = inputs_more_toxic['input_ids']
#         more_toxic_mask = inputs_more_toxic['attention_mask']
        
#         less_toxic_ids = inputs_less_toxic['input_ids']
#         less_toxic_mask = inputs_less_toxic['attention_mask']
        toxic_ids = inputs_toxic['input_ids']
        toxic_mask = inputs_toxic['attention_mask']
        toxic_token_type_ids = inputs_toxic['token_type_ids']
        
        return {
            'toxic_ids': torch.tensor(toxic_ids, dtype=torch.long),
            'toxic_mask': torch.tensor(toxic_mask, dtype=torch.long),
            'toxic_token_type_ids': torch.tensor(toxic_token_type_ids, dtype=torch.long),
#             'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
#             'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float32)
        }

class ValDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.more_toxic = df['more_toxic'].values
        self.less_toxic = df['less_toxic'].values
#         self.text = df['text'].values
#         self.y = df['y'].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        more_toxic = self.more_toxic[index]
        less_toxic = self.less_toxic[index]
#         text = self.text[index]
        inputs_more_toxic = self.tokenizer.encode_plus(
                                more_toxic,
                                truncation=True,
                                add_special_tokens=True,
                                max_length=self.max_len,
                                padding='max_length'
                            )
        inputs_less_toxic = self.tokenizer.encode_plus(
                                less_toxic,
                                truncation=True,
                                add_special_tokens=True,
                                max_length=self.max_len,
                                padding='max_length'
                            )
#         inputs_toxic = self.tokenizer.encode_plus(
#                                 text,
#                                 truncation=True,
#                                 add_special_tokens=True,
#                                 max_length=self.max_len,
#                                 padding='max_length'
#                             )
    
#         target = self.y[index]
        
        more_toxic_ids = inputs_more_toxic['input_ids']
        more_toxic_mask = inputs_more_toxic['attention_mask']
        more_toxic_token_type_ids = inputs_more_toxic['token_type_ids']
        
        less_toxic_ids = inputs_less_toxic['input_ids']
        less_toxic_mask = inputs_less_toxic['attention_mask']
        less_toxic_token_type_ids = inputs_less_toxic['token_type_ids']
#         toxic_ids = inputs_toxic['input_ids']
#         toxic_mask = inputs_less_toxic['attention_mask']
        
        return {
            'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
            'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
            'less_toxic_token_type_ids':torch.tensor(less_toxic_token_type_ids,dtype= torch.long),
            'more_toxic_ids': torch.tensor(more_toxic_ids, dtype=torch.long),
            'more_toxic_mask': torch.tensor(more_toxic_mask, dtype=torch.long),
            'more_toxic_token_type_ids':torch.tensor(more_toxic_token_type_ids,dtype=torch.long)
        }
    
class JigsawModel(nn.Module):
    def __init__(self, checkpoint=CONFIG['model_name']):
        super(JigsawModel, self).__init__()
        self.checkpoint = checkpoint
        self.xlnet = XLNetModel.from_pretrained(checkpoint, return_dict=False)
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids,token_type_ids, attention_mask):
        last_hidden_state = self.xlnet(input_ids=input_ids,token_type_ids = token_type_ids, attention_mask=attention_mask)
        pooled_output = self.pool_hidden_state(last_hidden_state)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        preds = self.dense(pooled_output)
        preds = torch.nn.Sigmoid()(preds)
        return preds
    
    def pool_hidden_state(self,last_hidden_state):
        '''
        pool the last_hidden_state into a mean hidden_state
        '''
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

def criterion(outputs1, outputs2, targets):
    return nn.MarginRankingLoss(margin=CONFIG['margin'])(outputs1, outputs2, targets)

def rmse_criterion(preds, targets):
    return nn.MSELoss()(preds, targets)

def BCE_criterion(preds, targets):
    return nn.BCELoss()(preds, targets)

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
#         more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
#         more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
#         less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
#         less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)

        toxic_ids = data['toxic_ids'].to(device, dtype=torch.long)
        toxic_mask = data['toxic_mask'].to(device, dtype=torch.long)
        toxic_token_type_ids = data['toxic_token_type_ids'].to(device, dtype=torch.long)
        
        targets = data['target'].to(device, dtype=torch.float32)
        
        batch_size = toxic_ids.size(0)

#         more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
#         less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        toxic_outputs = model(toxic_ids, toxic_token_type_ids, toxic_mask)
#         print(toxic_outputs.shape)
#         print(targets.shape)
        loss = BCE_criterion(toxic_outputs.squeeze(), targets)
#         loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)
        loss = loss / CONFIG['n_accumulate']
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
        
        wandb.log({'step_loss': epoch_loss, 'lr': optimizer.param_groups[0]['lr']})
    gc.collect()
    
    return epoch_loss


# @torch.no_grad()
# def valid_one_epoch(model, dataloader, device, epoch):
#     model.eval()
    
#     dataset_size = 0
#     running_loss = 0.0
    
#     bar = tqdm(enumerate(dataloader), total=len(dataloader))
#     for step, data in bar:        
#         toxic_ids = data['toxic_ids'].to(device, dtype=torch.long)
#         toxic_mask = data['toxic_mask'].to(device, dtype=torch.long)
#         toxic_token_type_ids = data['toxic_token_type_ids'].to(device, dtype=torch.long)
# #         more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
# #         more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
# #         less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
# #         less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
#         targets = data['target'].to(device, dtype=torch.float32)
        
#         batch_size = toxic_ids.size(0)

# #         more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
# #         less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        
#         toxic_outputs = model(toxic_ids, toxic_token_type_idstoxic_mask)
        
# #         loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)
#         loss = rmse_criterion(toxic_outputs.squeeze(), targets)
    
#         running_loss += (loss.item() * batch_size)
#         dataset_size += batch_size
        
#         epoch_loss = running_loss / dataset_size
        
#         bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
#                         LR=optimizer.param_groups[0]['lr'])   
        
#     gc.collect()
    
#     return epoch_loss

@torch.no_grad()
def calculate_cv(model, dataloader, device, fold):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    more_list = []
    less_list = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
        more_token_type_ids = data['more_toxic_token_type_ids'].to(device, dtype = torch.long)
        
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
#         targets = data['target'].to(device, dtype=torch.long)
        less_token_type_ids = data['less_toxic_token_type_ids'].to(device, dtype = torch.long)
    
        batch_size = more_toxic_ids.size(0)

        more_toxic_outputs = model(more_toxic_ids, more_token_type_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_token_type_ids, less_toxic_mask)
    
        more_list.extend(list(more_toxic_outputs.cpu().reshape(-1)))
        less_list.extend(list(less_toxic_outputs.cpu().reshape(-1)))
    
    more_list = np.array(more_list)
    less_list = np.array(less_list)
    cv_score = np.mean(less_list < more_list)
    
    print('cv: ', cv_score)
#     wandb.log({'cv_score': cv_score})
    run.summary['cv_score'] = cv_score
    gc.collect()
    
    return cv_score, less_list, more_list

def create_train (df):
    toxic = 1.0
    severe_toxic = 2.0
    obscene = 1.0
    threat = 1.0
    insult = 1.0
    identity_hate = 1.0
    df['y'] = df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].max(axis=1)
    df['y'] = df["y"]+df['severe_toxic']*severe_toxic
    df['y'] = df["y"]+df['obscene']*obscene
    df['y'] = df["y"]+df['threat']*threat
    df['y'] = df["y"]+df['insult']*insult
    df['y'] = df["y"]+df['identity_hate']*identity_hate
    
    df['y'] = df['y']/df['y'].max()
    
    df = df[['comment_text', 'y', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].rename(columns={'comment_text': 'text'})

    #undersample non toxic comments  on Toxic Comment Classification Challenge
    min_len = (df['y'] > 0).sum()
    df_y0_undersample = df[df['y'] == 0].sample(n=int(min_len*1.3),random_state=201)
    df = pd.concat([df[df['y'] > 0], df_y0_undersample])
                                                
    return df

def run_training(model, optimizer, scheduler, device, num_epochs, fold):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_cv = -np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
#         val_epoch_loss = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
#                                          epoch=epoch)
        
        
        cv_epoch, _, _ = calculate_cv(model, cv_loader, device=CONFIG['device'], 
                                         fold=fold)
        history['Train Loss'].append(train_epoch_loss)
        history['Epoch CV'].append(cv_epoch)
        
        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
#         wandb.log({"Valid Loss": val_epoch_loss})
        wandb.log({"Epoch CV": cv_epoch})
        
        # deep copy the model
        if cv_epoch >= best_epoch_cv:
            print(f"{b_}CV Improved ({best_epoch_cv} ---> {cv_epoch})")
            best_epoch_cv = cv_epoch
            run.summary["Best CV"] = best_epoch_cv
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"Loss-Fold-{fold}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best CV: {:.4f}".format(best_epoch_cv))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def prepare_loaders(fold):
#     df_train = df[df.kfold != fold].reset_index(drop=True)
#     df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    df_train = pd.read_csv(f'class_data/df_class_fld{fold}.csv')
#     len_df = len(tmp_df)
    df_valid = df_train.sample(n=100) # not use 
    
    train_dataset = JigsawDataset(df_train, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])
#     valid_dataset = JigsawDataset(df_valid, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=8, shuffle=True, pin_memory=True, drop_last=False)
#     valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
#                               num_workers=8, shuffle=False, pin_memory=True)
    
    df_len = len(df_train)
    return train_loader, df_len

def fetch_scheduler(optimizer, df_len):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CONFIG['max_lr'],
            steps_per_epoch=int(df_len /  (CONFIG['train_batch_size'] * CONFIG['n_accumulate']) ) + 1,
            epochs=CONFIG['epochs'],
            pct_start=CONFIG['pct_start'],
            anneal_strategy=CONFIG['anneal_strategy'],
            div_factor=CONFIG['div_factor'],
            final_div_factor=CONFIG['final_div_factor'],
        )
        
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

def clean(data, col):

    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' \n ')
    # Remove ip address
    data[col] = data[col].str.replace(r'(([0-9]+\.){2,}[0-9]+)',' ')
    
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')
    # patterns with repeating characters 
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()   
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)',r' \1 ')    
    
    return data

def create_folds():
    n_folds = CONFIG['n_fold']
    frac_1 = CONFIG['frac_1']
    frac_1_factor = CONFIG['frac_1_factor']
    
    for fld in range(n_folds):
        print(f'Fold: {fld}')
        tmp_df = pd.concat([df[df.y>0].sample(frac=frac_1, random_state = 10*(fld+1)) , 
                            df[df.y==0].sample(n=int(len(df[df.y>0])*frac_1*frac_1_factor) , 
                                                random_state = 10*(fld+1))], axis=0).sample(frac=1, random_state = 10*(fld+1))
#         tmp_df = df.sample(frac=frac_1, random_state=10*(fld+1))
        tmp_df.to_csv(f'class_data/df_class_fld{fld}.csv', index=False)
        print(tmp_df.shape)
        print(tmp_df['y'].value_counts())
# main

df_test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")
df_test_l = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv").replace(-1,0)
df_test = pd.merge(df_test, df_test_l, how="left", on = "id")

df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
# print(df.shape)
df = pd.concat([df, df_test])
# print(df.shape)/
del df_test


df['severe_toxic'] = df.severe_toxic * 2
df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
df['y'] = df['y']/df['y'].max()

df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

# df_o = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

# df_cv = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")

df_val = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")
print(df_val.shape)

# Find cases already present in toxic data

# df = create_train(df_o)

# df = df.reset_index(drop=True)

df_val = pd.merge(df_val, df.loc[:,['text']], 
                  left_on = 'less_toxic', 
                  right_on = 'text', how='left')

df_val = pd.merge(df_val, df.loc[:,['text']], 
                  left_on = 'more_toxic', 
                  right_on = 'text', how='left')

# Removing those cases
df_val = df_val[(~df_val.text_x.isna()) | (~df_val.text_y.isna())][['worker', 'less_toxic', 'more_toxic']]

print('val shape: ', df_val.shape)

print(df.shape)

## clean data
df_cv = df_val.copy()

df[df['y'] > 0]['y'] = 1
df[df['y'] == 0]['y'] = 0
# df = pd.read_csv('../input/bias-pseudo-label/bias_aug.csv')
# df = clean(df, 'text')
# df = df.rename(columns={'score': 'y'})
# df_cv = clean(df_cv, 'less_toxic')
# df_cv = clean(df_cv, 'more_toxic')

create_folds()

# skf = StratifiedKFold(n_splits=CONFIG['n_fold'], shuffle=True, random_state=CONFIG['seed'])

# for fold, ( _, val_) in enumerate(skf.split(X=df, y=np.around(df.y.values))):
#     df.loc[val_ , "kfold"] = int(fold)
    
# df["kfold"] = df["kfold"].astype(int)
# df.head()

cv_list = []

p1 = np.zeros((df_val.shape[0], CONFIG['n_fold']))
p2 = np.zeros((df_val.shape[0], CONFIG['n_fold']))

for fold in range(0, CONFIG['n_fold']):
    print(f"{y_}====== Fold: {fold} ======{sr_}")
    run = wandb.init(project='Jigsaw', 
                     config=CONFIG,
                     job_type='Train',
                     group=CONFIG['group'],
                     tags=['xlnet-base-case', f'{HASH_NAME}', 'BCE_loss'],
                     name=f'{HASH_NAME}-fold-{fold}',
                     anonymous='must')
    
    # Create Dataloaders
    train_loader, df_len = prepare_loaders(fold=fold)
    
    cv_dataset = ValDataset(df_cv, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])
    
    cv_loader = DataLoader(cv_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=8, shuffle=True, pin_memory=True, drop_last=False)
    
    model = JigsawModel(CONFIG['model_name'])
    model.to(CONFIG['device'])
    
    # Define Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer, df_len)
    
    model, history = run_training(model, optimizer, scheduler,
                                  device=CONFIG['device'],
                                  num_epochs=CONFIG['epochs'],
                                  fold=fold)
    
    cv, p1_tmp, p2_tmp = calculate_cv(model, cv_loader, device=CONFIG['device'], 
                                         fold=fold)
    
    cv_list.append(cv)
    p1[:, fold] = p1_tmp
    p2[:, fold] = p2_tmp
    
    if fold == CONFIG['n_fold'] - 1:
        p1 = np.mean(p1, axis=1)
        p2 = np.mean(p2, axis=1)
        ensemble_cv = np.mean(p1 < p2)
        run.summary['Ensemble CV'] = ensemble_cv
        
    run.finish()
    
    del model, history, train_loader
    _ = gc.collect()
    print()