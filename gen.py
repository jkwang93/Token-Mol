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
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
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
    parser.add_argument('--model_path', default="gpt2通用中文模型", type=str, help='')
    parser.add_argument('--vocab_path', default="gpt2通用中文模型/vocab.txt", type=str, help='')
    parser.add_argument('--save_model_path', default="save_model", type=str, help='')
    parser.add_argument('--final_model_path', default="final_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=128, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1e-3, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=100, type=int, required=False, help='print log steps')
    return parser.parse_args()




def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(tokenizer.pad_token_id)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    # rouge_score = rouge(not_ignore, shift_labels, preds)
    return loss, accuracy


def collate_fn(batch):
    input_ids = []
    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def data_loader(args, train_data_path, tokenizer, shuffle):
    data_list = []
    # with open(train_data_path, 'rb') as f:
    #     data = f.read().decode("utf-8")
    #     train_data = data.split("\n")
    #     print("数据总行数:{}".format(len(train_data)))
    train_data = pd.read_csv(train_data_path, header=None).values.flatten().tolist()
    print("数据总行数:{}".format(len(train_data)))

    for data_i in tqdm(train_data):
        data_list.append(tokenizer.encode(data_i, padding="max_length", truncation=True, max_length=34,
                                          return_special_tokens_mask=True, ))

    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader


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
    time1 = time.time()
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
    args.model_path, args.vocab_path = '', './data/torsion_version/torsion_voc_pocket.csv'

    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)

    model = Token3D(pretrain_path='./RTM_torsion_countinue_v2_epoch20_final_model2/',
                    config=Ada_config,
                    from_scratch=True)

    param_dict = {key.replace("module.", ""): value for key, value in torch.load('Trained_model/scrath_100epo_every.pt', map_location='cuda').items()}

    model.load_state_dict(param_dict)
    eval_data_protein = read_data('./data/val_protein_represent.pkl')
    
    all_output = []

    start_time = time.time()
    # 对每个口袋，生成1000个分子；可以根据自己的需求修改
    for pocket in tqdm(eval_data_protein):
        one_output = []
        Seq_all = []
        for i in range(4):
            Seq_list = predict(model, tokenizer, single_pocket=pocket,batch_size=25)
            Seq_all.extend(Seq_list)
        for j in Seq_all:
            one_output.append(decode(j))
        all_output.append(one_output)
    time_elapsed = (time.time() - start_time)
    aver_time = time_elapsed / len(eval_data_protein) / 100
    print(f'Time elapsed: {time_elapsed}s; Average time: {aver_time}s')
    output = pd.DataFrame(all_output)

    output.to_csv('scrath_100epo_gen.csv', index=False, header=False, mode='a')
