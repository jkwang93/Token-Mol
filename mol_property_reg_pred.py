# -*- coding:utf-8 -*-
# @Author: meijing
# @Time: 2023/11/28 18:06
# loss only in property
import os
import time
import pandas as pd
import torch
import argparse
import numpy as np

# from rouge import Rouge
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from transformers import get_scheduler, GPT2Config
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from early_stop.pytorchtools import EarlyStopping

from bert_tokenizer import ExpressionBertTokenizer
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import re

import numpy as np
from sklearn import metrics
from scipy.stats import spearmanr

from search import BeamSearch

pp_range = {"esol":[-13.1719,2.137682], "caco2":[-7.76,-3.51],"ld50":[0.291,10.207], "lipo":[-1.5,4.5], "freesolv":[-25.47,3.43],
            "hf":[0.065,69],"ppbr":[10.09,99.95],"vdss":[0.01,60],"ch":[3,150],"cm":[3,150]}

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="final_model_path", type=str, help='')
    parser.add_argument('--vocab_path', default="./data/torsion_version/torsion_voc_property.csv", type=str, help='')
    parser.add_argument('--save_model_path', default="save_model", type=str, help='')
    parser.add_argument('--final_model_path', default="final_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.csv', type=str, help='')
    parser.add_argument('--valid_raw_path', default='valid_raw_data.csv', type=str, help='')
    parser.add_argument('--test_raw_path', default='test_raw_data.csv', type=str, help='')
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=20000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    parser.add_argument('--property_name', default='caco2', type=str, required=False,
                        help='property need to be predicted')
    parser.add_argument('--min', default=-7.76, type=float, required=False,
                        )
    parser.add_argument('--max', default=-3.51, type=float, required=False,
                        )
    return parser.parse_args()


def collate_fn(batch):
    input_ids = []
    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([0] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def predict(model, DataLoader, args):
    beam_search = BeamSearch(
        temperature = 1.0,
        beam_width = 1,
        top_tokens = 5,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    pred_label = []
    max_length = 6
    with torch.no_grad():
        for i,data in enumerate(tqdm(DataLoader)):
            input_ids = data[0].to(device)
            response = []
            for _ in range(max_length):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                logits /= 1.2

                next_token_logits = logits[-1, :]
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                _,next_token = F.softmax(next_token_logits).max(dim=0)
                if next_token == tokenizer.sep_token_id:
                    break
                if len(input_ids) == 200:
                    print("over length,num {}".format(i+1),tokenizer.decode(input_ids))
                    break
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=0)
            text = tokenizer.convert_ids_to_tokens(response)
            pred_label.append(normalize_to_real(text ,args))
        #print(pred_label)
        return pred_label

def normalize_to_real(norm_label,args):
    try:
        norm_label=norm_label[0]
        norm_label=float(norm_label.split('p_')[1])
        real_label=(norm_label) *(args.max - args.min)+ args.min
    except ValueError:
        real_label = -1
    except IndexError:
        real_label = -1
    return real_label

def data_to_list(tokenizer, data_all):
    none = tokenizer.bos_token_id
    tokenizer.bos_token_id = tokenizer.cls_token_id
    tokenizer.eos_token_id = tokenizer.sep_token_id
    tokenizer.sep_token_id = tokenizer.vocab['<|endofmask|>']

    data_list = []

    true_label = []
    for data_i in tqdm(data_all):
        property_name = args.property_name
        label = str("{:.3f}".format(float(data_i.split('labels')[-1])))
        smiles = data_i.split('GEO')[0]
        data_j = data_i.split('GEO')[1]
        torsion = data_j.split('labels')[0]
        data_i = '<|beginoftext|> ' + smiles + 'GEO' + torsion + '<' + property_name + '>' + ' <|mask:0|>'
        data = tokenizer.encode(data_i, truncation=True, max_length=200, return_special_tokens_mask=True,
                                add_special_tokens=False)

        start_value = tokenizer.vocab['<' + property_name + '>']
        end_value = tokenizer.vocab['<|endofmask|>']
        if start_value not in data :
            print("error data:", data_i, "decode:", tokenizer.decode(data))
            continue
        data_list.append(data)
        true_label.append(float(label))
    return data_list, true_label

def evaluation(true_label, pred_label):
    rmse = np.sqrt(metrics.mean_squared_error(true_label, pred_label))
    mae = metrics.mean_absolute_error(true_label, pred_label)
    spearman = spearmanr(true_label, pred_label)[0]
    return rmse, mae, spearman


if __name__ == '__main__':
    start = time.perf_counter()
    args = setup_args()
    print(args.model_path)
    column = 'smiles_label_torsion_all'
    args.min = pp_range[args.property_name][0]
    args.max = pp_range[args.property_name][1]
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)

    test_data = pd.read_csv(args.test_raw_path, usecols=[column]).values.flatten().tolist()
    test_data_list, true_label = data_to_list(tokenizer, test_data)
    test_dataset = MyDataset(test_data_list)


    # testing
    testDataLoader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                                 pin_memory=False, collate_fn=collate_fn)

    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    pred_label = predict(model, testDataLoader, args)
    print(true_label, '\n',pred_label)
    rmse, mae, spearman = evaluation(true_label, pred_label)
    print(args.property_name)
    print("val_rmse: {},".format(rmse),
          "val_mae: {}.".format(mae),
          "val_spearman: {}.".format(spearman))
    end = time.perf_counter()

    runTime = end - start
    runTime_ms = runTime * 1000

    print("runTime：", runTime, "s")
    print("runTime：", runTime_ms, "ms")