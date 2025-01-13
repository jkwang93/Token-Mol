# -*- coding:utf-8 -*-
# @Author: jikewang
# @Time: 2023/10/8 17:55
# @File: pocket_fine_tuning.py
import pickle
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from transformers import get_scheduler, GPT2Config
from torch.utils.data import Dataset, DataLoader
from early_stop.pytorchtools import EarlyStopping
from bert_tokenizer import ExpressionBertTokenizer
from ada_model import Token3D
from utils.utils import cal_loss_and_accuracy, gce_loss_and_accuracy

# cross-attention+self-attention
Ada_config = GPT2Config(
    architectures=["GPT2LMHeadModel"],
    model_type="GPT2LMHeadModel",
    vocab_size=836,
    n_positions=380,
    n_ctx=380,  # max length
    n_embd=768,
    n_layer=12,
    n_head=8,

    task_specific_params={
        "text-generation": {
            "do_sample": True,
            "max_length": 380
        }
    }
)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="Pretrained_model", type=str, help='')
    parser.add_argument('--vocab_path', default="./data/torsion_version/torsion_voc_pocket.csv", type=str, help='')
    parser.add_argument('--every_step_save_path', default="Trained_model/pocket_generation", type=str, help='')
    parser.add_argument('--early_stop_path', default="Trained_model/pocket_generation", type=str, help='')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=20000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=5e-3, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    return parser.parse_args()


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def collate_fn(mix_batch):
    batch, protein_batch = list(zip(*mix_batch))
    input_ids = []

    input_lens_list = [len(w) for w in batch]
    input_protein_len_list = [len(ww) for ww in protein_batch]

    max_input_len = max(input_lens_list)

    max_protein_len = max(input_protein_len_list)

    # create a zero array for padding protein batch
    protein_ids = np.zeros((len(protein_batch), max_protein_len, len(protein_batch[0][0])),
                           dtype=protein_batch[0][0].dtype)

    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - input_len))

        # padding protein
        protein_ids[btc_idx, :len(protein_batch[btc_idx]), :] = protein_batch[btc_idx]

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(protein_ids, dtype=torch.float32)


def data_loader(args, train_data, matrix_protein, tokenizer, shuffle):
    data_list = []
    for ind, data_i in tqdm(enumerate(train_data)):
        # data_i = data_i.replace('GEO', '')
        data_i = '<|beginoftext|> <|mask:0|> <|mask:0|> ' + data_i + ' <|endofmask|>'
        mol_ = [tokenizer.encode(data_i, truncation=False, max_length=200, return_special_tokens_mask=True,
                                 add_special_tokens=False)]

        mol_.append(matrix_protein[ind])
        data_list.append(mol_)

    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader


def train(args, model, dataloader, eval_dataloader):
    num_training_steps = args.epochs * len(dataloader)
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

    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_loss_list = []
        print("\n")
        print("***********")
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print("***********")
        print("\n")
        for mix_batch in dataloader:
            batch, protein_batch = mix_batch
            batch_steps += 1
            batch = batch.to(device)
            protein_batch = protein_batch.to(device)
            outputs = model(batch, protein_batch)

            loss_conf, acc_conf = gce_loss_and_accuracy(outputs, batch.to(device), device)

            loss_smiles, acc_smiles = cal_loss_and_accuracy(outputs, batch.to(device), device)

            # Weight of conf and smiles loss can be adjusted
            loss = (loss_conf + loss_smiles) / 2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc_conf, acc_smiles
                ))

        every_save_path = f"{args.every_step_save_path}"
        torch.save(model.state_dict(), every_save_path + '.pt')

        evaluate(model, eval_dataloader, args=args)

    # torch.save(model, os.path.join(args.final_model_path, 'gpt2_WenAn.pth'))


def evaluate(model, dataloader, args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = GPT2LMHeadModel.from_pretrained(args.save_model_path)
    # model.load_state_dict(torch.load('final_model_early_stop.pt'))

    model.to(device)
    model.eval()
    loss_list, acc_list = [], []
    batch_steps = 0
    early_stopping = EarlyStopping(patience=20, verbose=False)

    with torch.no_grad():
        for mix_batch in dataloader:
            batch, protein_batch = mix_batch
            batch_steps += 1
            batch = batch.to(device)
            protein_batch = protein_batch.to(device)
            outputs = model(batch, protein_batch)

            loss, acc = gce_loss_and_accuracy(outputs, batch.to(device), device)
            loss_list.append(float(loss))
            acc_list.append(float(acc))

    epoch_loss = np.mean(loss_list)
    early_stopping(epoch_loss, model, args.early_stop_path)

    print("loss: {},".format(np.mean(loss_list)),
          "accuracy: {}.".format(np.mean(acc_list)))


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def read_data(path):
    data = []
    with open(path, 'rb') as f:
        while True:
            try:
                aa = pickle.load(f)
                data.extend(aa)
            except EOFError:
                break
    return data


if __name__ == '__main__':
    args = setup_args()
    args.model_path = './Pretrained_model'
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)

    save_path = Path(args.every_step_save_path).parent.mkdir(exist_ok=True)

    protein_matrix = read_data('./data/train_protein_represent.pkl')
    mol_data = read_data('./data/mol_input.pkl')
    eval_protein = read_data('./data/val_protein_represent.pkl')
    eval_mol = read_data('./data/val_mol_input.pkl')

    model = Token3D(pretrain_path=args.model_path, config=Ada_config)

    train_dataloader = data_loader(args, mol_data, protein_matrix, tokenizer=tokenizer, shuffle=True)
    eval_dataloader = data_loader(args, eval_mol, eval_protein, tokenizer=tokenizer, shuffle=True)

    train(args, model, train_dataloader, eval_dataloader)
