# -*- coding:utf-8 -*-
import os
import time
import argparse
import numpy as np
import pandas as pd
import time
import torch
# from rouge import Rouge
# from torchinfo import summary
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from bert_tokenizer import ExpressionBertTokenizer
from ada_model import Token3D
from pocket_fine_tuning_rmse import Ada_config
from pocket_fine_tuning_rmse import read_data

#from early_stop.pytorchtools import EarlyStopping


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
    parser.add_argument('--model_path', default='./Trained_model/pocket_generation.pt', type=str, help='')
    parser.add_argument('--vocab_path', default="./data/torsion_version/torsion_voc_pocket.csv", type=str, help='')
    parser.add_argument('--protein_path', default='./example/ARA2A.pkl', type=str, help='')
    parser.add_argument('--output_path', default='output.csv', type=str, help='')
    parser.add_argument('--batch_size', default=25, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=4, type=int, required=False, help='epochs')
    return parser.parse_args()


def decode(matrix):
    chars = []
    for i in matrix:
        if i == '<|endofmask|>': break
        chars.append(i)
    seq = " ".join(chars)
    return seq

@torch.no_grad()
def predict(model, tokenizer, batch_size, single_pocket,
            text="<|beginoftext|> <|mask:0|> <|mask:0|>"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, _ = load_model(args.save_model_path, args.vocab_path)
    # text = ""
    protein_batch = single_pocket
    model.to(device)
    model.eval()
    #time1 = time.time()
    max_length = 195
    input_ids = []
    input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    input_length = len(input_ids)

    input_tensor = torch.zeros(batch_size, input_length).long()
    input_tensor[:] = torch.tensor(input_ids)

    Seq_list = []

    finished = torch.zeros(batch_size, 1).byte().to(device)

    protein_batch = torch.tensor(protein_batch, dtype=torch.float32)
    protein_batch = protein_batch.to(device)
    protein_batch = protein_batch.repeat(batch_size, 1, 1)
    for i in range(max_length):
        inputs = input_tensor.to(device)
        outputs = model(inputs, protein_batch)
        logits = outputs.logits
        logits = F.softmax(logits[:, -1, :])

        last_token_id = torch.multinomial(logits, 1)
        #last_token_id = torch.argmax(logits,1).view(-1,1)

        EOS_sampled = (last_token_id == tokenizer.encode('<|endofmask|>', add_special_tokens=False))
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            print('End')
            break

        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)

        Seq_list.append(last_token)
    Seq_list = np.array(Seq_list).T

    return Seq_list

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    model_path, protein_path = args.model_path, args.protein_path

    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)
    model = Token3D(pretrain_path='./Pretrained_model', config=Ada_config)

    param_dict = {key.replace("module.", ""): value for key, value in torch.load(model_path, map_location='cuda').items()}

    model.load_state_dict(param_dict)
    eval_data_protein = read_data(protein_path)
    
    all_output = []

    start_time = time.time()
    # Total number = range * batch size
    for pocket in tqdm(eval_data_protein):
        one_output = []
        Seq_all = []
        for i in range(args.epochs):
            Seq_list = predict(model, tokenizer, single_pocket=pocket,batch_size=25)
            Seq_all.extend(Seq_list)
        for j in Seq_all:
            one_output.append(decode(j))
        all_output.append(one_output)
    time_elapsed = (time.time() - start_time)
    aver_time = time_elapsed / len(eval_data_protein) / 100
    print(f'Time elapsed: {time_elapsed}s; Average time: {aver_time}s')
    output = pd.DataFrame(all_output)

    output.to_csv(args.output_path, index=False, header=False, mode='a')
