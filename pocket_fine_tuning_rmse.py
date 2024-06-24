# -*- coding:utf-8 -*-
# @Author: jikewang
# @Time: 2023/10/8 17:55
# @File: pocket_fine_tuning.py
import copy
import os
import pickle
import time
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from transformers import get_scheduler, GPT2Config
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch import distributions
import torch.nn.functional as F
from early_stop.pytorchtools import EarlyStopping
from bert_tokenizer import ExpressionBertTokenizer
from ada_model import Token3D

# cross-attention+self-attention
Ada_config = GPT2Config(
    architectures=["GPT2LMHeadModel"],  # pretrain的时候用来预加载模型
    model_type="GPT2LMHeadModel",  # 定义模型类型，导出给`AutoConfig`用，如果要上传到hub请必填
    # tokenizer_class="BertTokenizer",  # 定义tokenizer类型，导出给`AutoTokenizer`用，如果要上传到hub请必填
    vocab_size=836,
    n_positions=380,
    n_ctx=380,  # 词最大长度
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


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="gpt2通用中文模型", type=str, help='')
    parser.add_argument('--vocab_path', default="gpt2通用中文模型/vocab.txt", type=str, help='')
    parser.add_argument('--every_step_save_path', default="every_step_model", type=str, help='')
    parser.add_argument('--early_stop_path', default="early_stop_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=20000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=5e-3, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    return parser.parse_args()


def get_all_normal_dis_pdf(voc_len=836, confs_num=629):
    means = torch.arange(1, confs_num + 1)  # create x of normal distribution for conf num
    std_dev = 2.0
    normal_dist_list = [distributions.Normal(mean.float(), std_dev) for mean in means]

    # 对数概率密度函数(log PDF)
    pdf_list = []
    zero_pdf = torch.zeros(voc_len)
    zero_pdf[0] = 1
    pdf_list.append(zero_pdf)
    for idx, normal_dist in enumerate(normal_dist_list):
        # if not confs num, make it as 0
        pdf = torch.zeros(voc_len)

        pdf[1:confs_num + 1] = normal_dist.log_prob(means.float()).exp().float()  # 计算 PDF
        # rate of ground truth 50% default is set to 4
        pdf[idx + 1] = pdf[idx + 1] * 2
        # normalized pdf
        normalized_pdf = pdf / pdf.sum()
        # print(normalized_pdf[idx+1])
        pdf_list.append(normalized_pdf)

    return np.array(pdf_list)


def calculate_loss_and_accuracy_confs(outputs, labels, device, pdf_array=get_all_normal_dis_pdf()):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    # need to ignore mask:i, mask 0-5 id: 830 smiles
    # shift_labels
    # covert_maskid_to_padid
    shift_labels_copy = copy.deepcopy(shift_labels)
    shift_labels_copy = shift_labels_copy.masked_fill((shift_labels_copy != 829), 0)

    shift_labels = shift_labels.masked_fill((shift_labels >= 630), 0)
    shift_labels_copy_copy = copy.deepcopy(shift_labels)

    shift_labels = shift_labels + shift_labels_copy

    one_hot = F.one_hot(shift_labels, num_classes=836).float()  # 对标签进行one_hot编码

    non_zero_indices = torch.nonzero(shift_labels_copy_copy)

    # todo speed up this part
    for i in non_zero_indices:
        row = i[0]
        li_index = i[1]
        poisson_one_hot = pdf_array[shift_labels[row][li_index].cpu()]

        one_hot[row][li_index] = poisson_one_hot

    # softmax = torch.exp(shift_logits) / torch.sum(torch.exp(shift_logits), dim=1).reshape(-1, 1)
    # logsoftmax = torch.log(softmax)

    # custom cross entropy loss
    logsoftmax = F.log_softmax(shift_logits, dim=-1)

    # mask 0
    not_ignore = shift_labels.ne(0)
    one_hot = not_ignore.unsqueeze(-1) * one_hot

    # / shift_labels.shape[0]
    loss = -torch.sum(one_hot * logsoftmax)

    # loss_fct = CrossEntropyLoss(ignore_index=0, reduction='sum')
    # loss2 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #
    # print(loss,loss2)

    _, preds = shift_logits.max(dim=-1)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    # rouge_score = rouge(not_ignore, shift_labels, preds)
    return loss, accuracy


def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    # need to ignore mask:i, mask 0-5 id: 830
    # shift_labels
    # covert_maskid_to_padid
    shift_labels = shift_labels.masked_fill((shift_labels < 630), 0)

    loss_fct = CrossEntropyLoss(ignore_index=0, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(0)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    # rouge_score = rouge(not_ignore, shift_labels, preds)
    return loss, accuracy


def data_loader(args, train_data, matrix_protein, tokenizer, shuffle):
    data_list = []
    # with open(train_data_path, 'rb') as f:
    #     data = f.read().decode("utf-8")
    #     train_data = data.split("\n")
    #     print("数据总行数:{}".format(len(train_data)))
    # train_data = pd.read_csv(train_data_path, header=None).values.flatten().tolist()
    # print("数据总行数:{}".format(len(train_data)))

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

            loss_conf, acc_conf = calculate_loss_and_accuracy_confs(outputs, batch.to(device), device)

            loss_smiles, acc_smiles = calculate_loss_and_accuracy(outputs, batch.to(device), device)

            # 这个地方可以按照需求跟conf与smiles加权重
            loss = (loss_conf + loss_smiles) / 2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}, confs_accuracy {}, smi_accuracy {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc_conf, acc_smiles
                ))

        every_save_path = f"{args.every_step_save_path}"
        torch.save(model.state_dict(), every_save_path + '_every.pt')

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

            loss, acc = calculate_loss_and_accuracy_confs(outputs, batch.to(device), device)
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
    args.model_path, args.vocab_path = './RTM_torsion_countinue_v2_epoch20_final_model2', './data/torsion_version/torsion_voc.csv'
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)

    # model = GPT2LMHeadModel.from_pretrained(args.model_path)
    # CrossSelfAttention = CrossSelfAttention(config=Ada_config)
    # print(model)
    # from torchsummary import summary
    # summary(model=model, input_size=(1,y,z), batch_size=32, device="cpu") # 分别是输入数据的三个维度

    protein_matrix = read_data('./data/add_Hs_version/train_protein_represent_addHs_fixed.pkl')
    mol_data = read_data('./data/add_Hs_version/train_mol_input_addHs_fixed.pkl')
    eval_protein = read_data('./data/add_Hs_version/val_protein_represent_addHs.pkl')
    eval_mol = read_data('./data/add_Hs_version/val_mol_input_addHs.pkl')
    # while True:
    #     try:
    #         aa = pickle.load(f)
    #         print(len(aa))
    #     except EOFError:
    #         break

    model = Token3D(pretrain_path=args.model_path, config=Ada_config)
    # print(model)

    train_dataloader = data_loader(args, mol_data, protein_matrix, tokenizer=tokenizer, shuffle=True)
    eval_dataloader = data_loader(args, eval_mol, eval_protein, tokenizer=tokenizer, shuffle=True)

    train(args, model, train_dataloader, eval_dataloader)
