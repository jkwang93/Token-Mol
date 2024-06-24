# -*- coding:utf-8 -*-
# @Author: jikewang
# @Time: 2024/2/19 17:57
# @File: reinforce_multigpu.py

# !/usr/bin/env python
import argparse
import warnings
import numpy as np
import pandas as pd
import time
import os
import glob
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from ada_model import Token3D
from torch.nn.parallel import DistributedDataParallel as DDP
from bert_tokenizer import ExpressionBertTokenizer
from transformers import GPT2Config
from smi_torsion_2_molobj import construct_molobj
from reward_score import scoring
from MCMG_utils.data_structs import Experience
from utils import Variable, unique, read_data, decode

warnings.filterwarnings("ignore")

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# If TensorBoard is available
# from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda") if torch.cuda.is_available() else None
if not device:
    raise TimeoutError('No GPU detected.')

Ada_config = GPT2Config(
    architectures=["GPT2LMHeadModel"],  # pretrain的时候用来预加载模型
    model_type="GPT2LMHeadModel",  # 定义模型类型，导出给`AutoConfig`用，如果要上传到hub请必填
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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # dist.init_process_group("nccl",rank=-1,world_size=-1)
    rank = dist.get_rank()
    ws = dist.get_world_size()
    return rank, ws


def nll_loss(inputs, targets):
    target_expanded = torch.zeros(inputs.size()).to(device)
    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).detach(), 1.0)
    loss = torch.sum(target_expanded * inputs, 1)
    return loss


def likelihood(model, tokenizer, target, single_protein):
    batch_size, seq_length = target.size()
    start_token = Variable(torch.zeros(batch_size, 1).long())
    start_token[:] = tokenizer.encode("<|beginoftext|> <|mask:0|> <|mask:0|>", add_special_tokens=False)[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()

    x = torch.cat((start_token, target[:, :-1]), 1).to(device)
    protein_batch = torch.tensor(np.array(single_protein), dtype=torch.float32)
    protein_batch = protein_batch.repeat(batch_size, 1, 1).to(device)

    log_probs = Variable(torch.zeros(batch_size))

    for step in range(seq_length):
        inputs = x[:, step].unsqueeze(1).to(device)
        outputs = model(inputs, protein_batch)
        logits = outputs.logits
        log_prob = F.log_softmax(logits[:, -1, :], dim=1)
        log_probs += nll_loss(log_prob, target[:, step])

    return log_probs


def sample(model, tokenizer, single_protein: list, batch_size: int, max_length: int,
           text="<|beginoftext|> <|mask:0|> <|mask:0|>"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()

    input_ids = []
    input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    input_length = len(input_ids)

    input_tensor = torch.zeros(batch_size, input_length).long()
    input_tensor[:] = torch.tensor(input_ids)
    Seq_list = []
    log_probs = torch.zeros(batch_size).to(device)
    finished = torch.zeros(batch_size, 1).byte().to(device)

    protein_batch = torch.tensor(np.array(single_protein), dtype=torch.float32)
    protein_batch = protein_batch.to(device)
    protein_batch = protein_batch.repeat(batch_size, 1, 1)

    for i in range(max_length):
        inputs = input_tensor.to(device)

        outputs = model(inputs, protein_batch)

        logits = outputs.logits
        prob = F.softmax(logits[:, -1, :])
        log_prob = F.log_softmax(logits[:, -1, :], dim=1)

        last_token_id = torch.multinomial(prob, 1)
        # last_token_id = torch.argmax(logits,1).view(-1,1)
        log_probs += nll_loss(log_prob, last_token_id)

        EOS_sampled = (last_token_id == tokenizer.encode('<|endofmask|>', add_special_tokens=False))
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            break

        # last_token = tokenizer.convert_ids_to_tokens(last_token_id)

        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)

        Seq_list.append(last_token_id.view(-1, 1))

    Seq_list = torch.cat(Seq_list, 1)

    return Seq_list, log_probs


def train_agent(rank, world_size, n_steps, batch_size, max_length, sigma, experience_replay, agent_save, protein_dir,
                Prior, Agent):
    rank, world_size = setup(rank, world_size)
    torch.cuda.set_device(rank)

    #tokenizer = ExpressionBertTokenizer('../pocket_generate/data/torsion_version/torsion_voc_pocket.csv')
    tokenizer = ExpressionBertTokenizer('../data/torsion_version/torsion_voc_pocket.csv')
    start_time = time.time()

    Prior.to(rank)
    Agent.to(rank)

    # We dont need gradients with respect to Prior
    for param in Prior.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(Agent.parameters(), lr=1e-5)  # default lr=1e-4
    experience = Experience(tokenizer)

    protein_emb = next(Path(protein_dir).glob('*.pkl'))
    single_protein = read_data(protein_emb)
    
    # sync batch normalization
    Agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Agent)
    # run model on the rank pid
    Agent = DDP(Agent, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    print("Model initialized, starting training...")

    for step in tqdm(range(n_steps)):

        # Sample from Agent
        # with torch.no_grad():
        seqs, agent_likelihood = sample(model=Agent,
                                        tokenizer=tokenizer,
                                        single_protein=single_protein,
                                        batch_size=batch_size,
                                        max_length=max_length)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]

        # Get prior likelihood and score the molecules
        prior_likelihood = likelihood(Prior, tokenizer, Variable(seqs), single_protein)
        smiles_torsions = []
        for seq in seqs:
            seq = tokenizer.convert_ids_to_tokens(seq)
            smiles_torsions.append(decode(seq))

        # todo insert your scoring function
        pred_file = construct_molobj(smiles_torsions)
        score, terms, poses = scoring(pred_file, protein_dir, rank=rank)

        # Save molecules in every steps
        saved_moles = {f'Steps{step+1}': poses}
        with open('every_steps_saved.pkl', 'ab') as f:
            pickle.dump(saved_moles, f)

        # Write the reward terms into csv file
        record_s = pd.DataFrame({'Steps': step + 1}, index=[0])
        terms = pd.concat([record_s, terms], axis=1)

        terms.to_csv('reward_terms' + str(rank) + '.csv', mode='a', header=False, index=False)

        # Calculate augmented likelihood
        expanded_score = torch.Tensor(score)
        augmented_likelihood = prior_likelihood + sigma * Variable(expanded_score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood = likelihood(Agent, tokenizer, exp_seqs.long(), single_protein)
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(pred_file, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights

        # loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dist.barrier()
        optimizer.zero_grad()

        if step % 5 == 0 and step != 0:
            torch.save(Agent.state_dict(), agent_save)

        # Print some information for this step
        if rank == 0:
            time_elapsed = (time.time() - start_time) / 3600
            time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
            print("\n   Step {}   Mean score: {:6.2f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
                step, np.mean(score), time_elapsed, time_left), flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--world-size', action='store', dest='world_size', type=int,
                        default=1)
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=1000)
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=2,
                        help='Batch size in a single device. Remember that total batch size = batch-size * world-size')
    parser.add_argument('--max-length', action='store', dest='max_length', type=int,
                        default=50)
    parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                        default=60)
    parser.add_argument('--experience-replay', type=int, default=0)
    # parser.add_argument('--restore-from', default='../Trained_model/20epoch_every.pt',
    #                     help='Path for loading the model.')
    parser.add_argument('--agent', action='store', dest='agent_save',
                        default='agent_checkpoint_QED.pt',
                        help='Path to an RNN checkpoint file to use as a Agent.')
    parser.add_argument('--protein-dir', action='store', dest='protein_dir',
                        default='./usecase_protein_embedding/CDK4',
                        help='Path where store protein target informations.')
    # parser.add_argument('--save-file-path', action='store', dest='save_dir',
    # help='Path where results and model are saved. Default is data/results/run_<datetime>.')

    args = parser.parse_args()
    args_list = list(vars(args).values())

    try:
        os.remove('every_steps_saved.pkl')
        files = glob.glob('reward_terms*.csv')
        for i in files:
            os.remove(i)
    except:
        pass

    Prior = Token3D(pretrain_path='../RTM_torsion_countinue_v2_epoch20_final_model2/', config=Ada_config)
    Agent = Token3D(pretrain_path='../RTM_torsion_countinue_v2_epoch20_final_model2/', config=Ada_config)
    restore_from = '../Trained_model/20epoch_rmse_40steps.pt'
    #restore_from = '20epoch_rmse_40steps.pt'

    prior_param_dict = {key.replace("module.", ""): value for key, value in
                        torch.load(restore_from, map_location='cuda').items()}
    agent_param_dict = {key.replace("module.", ""): value for key, value in
                        torch.load(restore_from, map_location='cuda').items()}

    Prior.load_state_dict(prior_param_dict)
    Agent.load_state_dict(agent_param_dict)

    args_list.append(Prior)
    args_list.append(Agent)

    mp.spawn(train_agent,
             args=args_list,
             nprocs=args.world_size,
             join=True)
