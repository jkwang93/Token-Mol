# -*- coding:utf-8 -*-
# @Author: meijing
# @Time: 2023/11/28 18:06
# loss only in property
import os
import time
from typing import Tuple
import pandas as pd

import torch
import argparse
import numpy as np
import copy

# from rouge import Rouge
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from transformers import get_scheduler, GPT2Config
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss, MSELoss

from early_stop.pytorchtools import EarlyStopping

from bert_tokenizer import ExpressionBertTokenizer
from torch import distributions
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import re

import numpy as np
from sklearn import metrics
from scipy.stats import spearmanr


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)

pp_range = {"esol":[-13.1719,2.137682], "caco2":[-7.76,-3.51],"ld50":[0.291,10.207], "lipo":[-1.5,4.5], "freesolv":[-25.47,3.43],
            "hf":[0.065,69],"ppbr":[10.09,99.95],"vdss":[0.01,60],"ch":[3,150],"cm":[3,150]}

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='RTM_torsion_countinue_v2_epoch7/', type=str, help='')
    parser.add_argument('--vocab_path', default="./data/torsion_version/torsion_voc_property.csv", type=str, help='')
    parser.add_argument('--save_model_path', default="save_model", type=str, help='')
    parser.add_argument('--final_model_path', default="final_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.csv', type=str, help='')
    parser.add_argument('--valid_raw_path', default='valid_raw_data.csv', type=str, help='')
    parser.add_argument('--test_raw_path', default='test_raw_data.csv', type=str, help='')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=20000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    parser.add_argument('--property_name', default='caco2', type=str, required=False,
                        help='property need to be predicted')

    return parser.parse_args()

def get_all_normal_dis_pdf(voc_len=836, label_num=101):
    means = torch.arange(1, label_num + 1)
    std_dev = 2.0
    normal_dist_list = [distributions.Normal(mean.float(), std_dev) for mean in means]

    pdf_list = []
    for i in range(633):
        zero_pdf = torch.zeros(voc_len)
        zero_pdf[i] = 1
        pdf_list.append(zero_pdf)
    for idx, normal_dist in enumerate(normal_dist_list):
        pdf = torch.zeros(voc_len)

        pdf[633:label_num + 633] = normal_dist.log_prob(means.float()).exp().float()  # 计算 PDF
        pdf[idx + 633] = pdf[idx + 633] * 2
        normalized_pdf = pdf / pdf.sum()
        pdf_list.append(normalized_pdf)
    return pdf_list


def calculate_loss_and_accuracy_label(outputs, labels, device, pdf_array):
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous().to(device)
    shift_labels = labels[..., 1:].contiguous().to(device)
    conditinal_label = torch.zeros_like(shift_labels).to(device)
    conditinal_label_noend = copy.deepcopy(conditinal_label)

    conditinal_label = torch.where((shift_labels >= 630) & (shift_labels <= 735)| (shift_labels == 829), torch.tensor(1,device=device), conditinal_label)
    conditinal_label_noend = torch.where((shift_labels >= 630) & (shift_labels <= 735), torch.tensor(1,device=device), conditinal_label_noend)
    mask=(conditinal_label==1)
    mask_noend = (conditinal_label_noend == 1)
    select_label = torch.where(mask,shift_labels,torch.tensor(0,device=device))
    select_label_noend = torch.where(mask_noend, shift_labels, torch.tensor(0, device=device))

    one_hot = F.one_hot(select_label, num_classes=836).float()

    non_zero_indices = torch.nonzero(select_label_noend)

    for i in non_zero_indices:
        row = i[0]
        li_index = i[1]
        poisson_one_hot = pdf_array[select_label[row][li_index].cpu()]

        one_hot[row][li_index] = poisson_one_hot

    logsoftmax = F.log_softmax(shift_logits, dim=-1)
    not_ignore = select_label.ne(0)
    one_hot = not_ignore.unsqueeze(-1) * one_hot
    loss = -torch.sum(one_hot * logsoftmax)

    _, preds = shift_logits.max(dim=-1)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()
    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy, preds

def collate_fn(batch):
    input_ids = []
    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([0] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def train(train_dataset, eval_dataset, args):
    '''rank, world_size = setup(rank, world_size)
    torch.cuda.set_device(rank)
    save_model_path: the model last path to save
    print(rank)'''

    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)

    tokenizer.bos_token_id = tokenizer.cls_token_id
    tokenizer.eos_token_id = tokenizer.sep_token_id
    tokenizer.sep_token_id = tokenizer.vocab['<|endofmask|>']

    model = GPT2LMHeadModel.from_pretrained(args.model_path, ignore_mismatched_sizes=True)


    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  pin_memory=True, collate_fn=collate_fn)

    evalDataLoader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                                 pin_memory=False, collate_fn=collate_fn)

    num_training_steps = args.epochs * len(trainDataLoader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    # model.train()
    batch_steps = 0
    early_stopping = EarlyStopping(patience=10, verbose=False)
    pdf_array = get_all_normal_dis_pdf()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_list = []

        total_loss = 0
        for batch in trainDataLoader:
            model.train()
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}

            outputs = model(**inputs, labels=batch.to(device))
            loss, acc, _ = calculate_loss_and_accuracy_label(outputs, batch.to(device), device, pdf_array)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc,
                ))
            lr_scheduler.step()
            optimizer.zero_grad()
        print('average epoch loss:', total_loss / len(trainDataLoader))
        eval_loss, eval_acc, _ = evaluate(model, evalDataLoader, args=args)
        early_stopping(eval_loss, model, args.save_model_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.save_model_path)

def evaluate(model, dataloader, args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()
    loss_list, acc_list = [], []
    batch_steps = 0
    test_predict_all = []
    pdf_array = get_all_normal_dis_pdf()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            loss, acc, _ = calculate_loss_and_accuracy_label(outputs, batch.to(device), device, pdf_array)
            loss_list.append(float(loss))
            acc_list.append(float(acc))

    epoch_loss = np.mean(loss_list)
    model.train()
    epoch_accuracy = np.mean(acc_list)
    print("val_loss: {},".format(np.mean(loss_list)),
          "val_accuracy: {}.\n".format(np.mean(acc_list)))
    return epoch_loss, epoch_accuracy, outputs


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def normalize_to_range(values,current_min,current_max, new_min, new_max):
    normalized_values = (values - current_min) / (current_max - current_min) * (new_max - new_min) + new_min
    normalized_values = round(normalized_values, 2)
    return normalized_values

def data_to_list(tokenizer, data_all, args):
    none = tokenizer.bos_token_id
    tokenizer.bos_token_id = tokenizer.cls_token_id
    tokenizer.eos_token_id = tokenizer.sep_token_id

    data_list = []
    for data_i in tqdm(data_all):
        property_name = args.property_name
        label = str("{:.3f}".format(float(data_i.split('labels')[-1])))
        label = float(label)
        normalized_values = normalize_to_range(label, args.min, args.max, 0, 1)
        normalized_values = str("{:.2f}".format(normalized_values))
        normalized_values = "p_" + normalized_values

        smiles = data_i.split('GEO')[0]
        data_j = data_i.split('GEO')[1]
        torsion = data_j.split('labels')[0]

        data_i = '<|beginoftext|> ' + smiles + 'GEO' + torsion + '<' + property_name + '>'+ ' <|mask:0|> ' + normalized_values + ' <|endofmask|>'
        data = tokenizer.encode(data_i, truncation=True, max_length=200, return_special_tokens_mask=True,
                                add_special_tokens=False)
        start_value = tokenizer.vocab['<|mask:0|>']
        end_value = tokenizer.vocab['<|endofmask|>']
        if end_value not in data:
            print("error data:", data_i, "decode:", tokenizer.decode(data))
            continue
        data_list.append(data)
    return data_list

if __name__ == '__main__':
    args = setup_args()

    column = 'smiles_label_torsion_all'
    property_name = args.property_name
    args.min = pp_range[property_name][0]
    args.max = pp_range[property_name][1]

    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)
    train_data = pd.read_csv(args.train_raw_path, usecols=[column]).values.flatten().tolist()
    valid_data = pd.read_csv(args.valid_raw_path, usecols=[column]).values.flatten().tolist()

    train_data_list = data_to_list(tokenizer, train_data, args)

    train_dataset = MyDataset(train_data_list)
    eval_data_list = data_to_list(tokenizer, valid_data, args)
    eval_dataset = MyDataset(eval_data_list)
    print("data splited! trian,valid,test:", len(train_dataset), len(eval_dataset))

    train(args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
