import pandas as pd
import os
import torch

import datasets
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import StratifiedShuffleSplit


# LABEL2ID = {"entailment" : 0, "contradiction" : 1, "neutral" : 2}
LABEL2ID = {"real" : 0, "fake" : 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_aeda_data(train_df, path):
    print("######### Adding aeda data #########")
    aeda_df = pd.read_excel(os.path.join(path,'validation_aeda.csv'), engine='openpyxl')
    aeda_df['Label'] = aeda_df['Label'].map(LABEL2ID)
    print(f"#### Aeda dataset length : {len(aeda_df)} ####")
    if isinstance(train_df, pd.DataFrame):
        train_df = pd.concat([train_df, aeda_df]).reset_index(drop=True)
        train_df['index'] = train_df.index
    elif isinstance(train_df, Dataset):
        aeda_dataset = Dataset.from_pandas(aeda_df)
        train_df = concatenate_datasets([train_df, aeda_dataset]).shuffle(seed=42)
    return train_df

def load_train_rtt_data(train_df, path):
    print("######### Adding Train RTT data #########")
    rtt_df = pd.read_excel(os.path.join(path,'train_rtt.csv'), engine='openpyxl')
    rtt_df['Label'] = rtt_df['Label'].map(LABEL2ID)
    print(f"#### Train RTT dataset length : {len(rtt_df)} ####")
    if isinstance(train_df, pd.DataFrame):
        train_df = pd.concat([train_df, rtt_df]).reset_index(drop=True)
        train_df['index'] = train_df.index
    elif isinstance(train_df, Dataset):
        rtt_dataset = Dataset.from_pandas(rtt_df)
        train_df = concatenate_datasets([train_df, rtt_dataset]).shuffle(seed=42)
    return train_df

def load_valid_rtt_data(train_df, path):
    print("######### Adding Valid RTT data #########")
    rtt_df = pd.read_excel(os.path.join(path,'validation_rtt.csv'), engine='openpyxl')
    rtt_df['Label'] = rtt_df['Label'].map(LABEL2ID)
    print(f"#### RTT dataset length : {len(rtt_df)} ####")
    if isinstance(train_df, pd.DataFrame):
        train_df = pd.concat([train_df, rtt_df]).reset_index(drop=True)
        train_df['index'] = train_df.index
    elif isinstance(train_df, Dataset):
        rtt_dataset = Dataset.from_pandas(rtt_df)
        train_df = concatenate_datasets([train_df, rtt_dataset]).shuffle(seed=42)
    return train_df

def load_train_data(args, path, train_dataset_path, valid_dataset_path):
    if 'xlsx' in train_dataset_path:
        train_df = pd.read_excel(os.path.join(path, train_dataset_path), engine='openpyxl')
    elif 'csv' in train_dataset_path:    
        train_df = pd.read_csv(os.path.join(path, train_dataset_path))
    if 'xlsx' in valid_dataset_path:
        valid_df = pd.read_excel(os.path.join(path, valid_dataset_path), engine='openpyxl')
    elif 'csv' in valid_dataset_path:    
        valid_df = pd.read_csv(os.path.join(path, valid_dataset_path))
    train_df['Label'] = train_df['Label'].map(LABEL2ID)
    valid_df['Label'] = valid_df['Label'].map(LABEL2ID)
            
    # if len(train_df.columns) > 3:
    #    train_df = train_df[['Title', 'Content', 'Label']]
    #if len(valid_df.columns) > 3:
    #     valid_df = valid_df[['Title', 'Content', 'Label']]
    
    if args.k_fold == 0 :
        if args.aeda :
            train_df = load_aeda_data(train_df, path)
        
    return Dataset.from_pandas(train_df), Dataset.from_pandas(valid_df)
    
    # else :
    #     print("######### Loading full train dataset #########")
    #     train_df = pd.concat([train_df, valid_df])
        
    #     print(f"#### Train dataset length : {len(train_df)} ####")
    #     train_df = train_df.reset_index(drop=True)
    #     train_df['index'] = train_df.index
    #     return Dataset.from_pandas(train_df), None
    
def load_test_data(path, test_dataset_path):
    print("######### Loading test dataset #########")
    if 'xlsx' in test_dataset_path:
        test_df = pd.read_excel(os.path.join(path, test_dataset_path), engine='openpyxl')
    elif 'csv' in test_dataset_path:
        test_df = pd.read_csv(os.path.join(path, test_dataset_path))
        
    test_df['Label'] = 0
    # if len(test_df.columns) > 3:
    #     test_df = test_df[['Title', 'Content', 'Label']]      

    return test_df, Dataset.from_pandas(test_df)


def preprocess_function(examples : datasets, tokenizer, args) -> datasets:
    premise = examples['Title']
    hypothesis = examples['Content']
    hypothesis2 = examples['Fake Content']
    label = examples['Label']
    
    hypothesis2 = [" " if hyp2 == "not fake" else hyp2 for hyp2 in hypothesis2]

    if args.use_SIC:
        input_ids = tokenizer(premise, hypothesis, truncation=True, return_token_type_ids = False)['input_ids']
        length = [len(one_input) for one_input in input_ids]
        model_inputs = {'input_ids':input_ids, 'labels':label, 'length':length}
    else :
        model_inputs = tokenizer(premise, hypothesis, hypothesis2, truncation=True, padding=True, return_token_type_ids = False)
        model_inputs['labels'] = label

    return model_inputs