# -*- coding:utf-8 -*-
import copy
import glob, os, pickle, random, tqdm
from collections import defaultdict
from argparse import ArgumentParser

from rdkit import RDLogger, Chem
from rdkit.Chem import rdmolops, AllChem
import numpy as np

from utils.standardization import clean_confs, get_torsion_angles, mmff_func, get_von_mises_rms, \
    optimize_rotatable_bonds, get_dihedral_vonMises, apply_changes, RMSD

RDLogger.DisableLog('rdApp.*')

parser = ArgumentParser()
# parser.add_argument('--worker_id', type=int, required=True, help='Worker id to determine correct portion')
# parser.add_argument('--out_dir', type=str, required=True, help='Output directory for the pickles')
parser.add_argument('--jobs_per_worker', type=int, default=1000, help='Number of molecules for each worker')
parser.add_argument('--root', type=str, default='../data/conf_test_data/test_mols.pkl',
                    help='Directory with molecules pickle files')
parser.add_argument('--popsize', type=int, default=15, help='Population size for differential evolution')
parser.add_argument('--max_iter', type=int, default=15, help='Maximum number of iterations for differential evolution')
parser.add_argument('--confs_per_mol', type=int, default=100,
                    help='Maximum number of conformers to take for each molecule')
parser.add_argument('--mmff', action='store_true', default=False,
                    help='Whether to relax seed conformers with MMFF before matching')
parser.add_argument('--no_match', action='store_true', default=False, help='Whether to skip conformer matching')
parser.add_argument('--boltzmann', choices=['top', 'resample'], default=None,
                    help='If set, specifies a different conformer selection policy')
args = parser.parse_args()

"""
    Refers to the process of conformer matching to run before the start of training, takes the conformers from
    a subset of the pickle files in the root directory and saves a final pickle for all of the. Example script:

    for i in $(seq 0, 299); do
        python standardize_confs.py --out_dir data/DRUGS/standardized_pickles --root data/DRUGS/drugs/ --confs_per_mol 30 --worker_id $i --jobs_per_worker 1000 &
    done
"""

REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)


def get_von_mises_rms_torsion_to_mol(mol, mol_rdkit, rotable_bonds, conf_id):
    new_dihedrals = np.zeros(len(rotable_bonds))
    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = get_dihedral_vonMises(mol_rdkit,
                                                   mol_rdkit.GetConformer(conf_id), r,
                                                   mol.GetConformer().GetPositions())
    mol_rdkit = apply_changes(mol_rdkit, new_dihedrals, rotable_bonds, conf_id)
    return RMSD(mol_rdkit, mol, conf_id)


def log_error(err):
    print(err)
    return None


average_rsmd = []


def conformer_match(smiles, new_dihedrals):
    '''convert it like confs'''
    mol_rdkit = Chem.MolFromSmiles(smiles)

    AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=1)
    if mol_rdkit.GetNumConformers() ==0:
        print('wrong:',smiles)

    rotable_bonds = get_torsion_angles(mol_rdkit)

    if not rotable_bonds:
        return log_error("no_rotable_bonds")
    new_rdkit = apply_changes(mol_rdkit, new_dihedrals[:len(rotable_bonds)], rotable_bonds, 0)

    return new_rdkit



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


import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    # 599 wrong of save_models_every;
    count = 0

    eval_data_mol = read_data('../data/val_mol_input.pkl')
    generate_mol = pd.read_csv('20epoch_every.csv', header=None).values.reshape(-1, 100).tolist()
    val_df = pd.read_csv('val_list.txt')
    destination = './epo20/epo20_pkl'
    os.makedirs(destination, exist_ok=True)

    # 初始化一个空字典
    val_dict = {}

    # 使用循环将DataFrame的两列转换为字典
    for index, row in val_df.iterrows():
        key = row['smiles']
        value = row['protein_path']
        val_dict[key] = value

    for ind, eval_smiles in tqdm(enumerate(eval_data_mol)):
        # find protein path, you can read pdb here
        pdb_path = val_dict[eval_smiles]
        pred_file = []
        protein_idx = pdb_path.split('/')[0]
        pkl_file = f'{destination}/{protein_idx}.pkl'
        if os.path.exists(pkl_file):
            continue

        # construct mol obj
        gen_mols = generate_mol[ind]
        for gen_mol in gen_mols:
            if gen_mol==np.nan:
                print('nan')

            try:
                smiles, torsion = gen_mol.split('GEO')
                smiles = smiles.replace(' ', '')
                torsion = np.array(torsion.split(' ')[1:]).astype(np.float64)
                pred_mol = conformer_match(smiles, torsion)
                if pred_mol:
                    pred_file.append(pred_mol)

            except Exception as e:
                continue
        
        with open(pkl_file, 'wb') as f:
            pickle.dump(pred_file, f)