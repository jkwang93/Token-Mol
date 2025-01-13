# -*- coding:utf-8 -*-
import glob, os, pickle, random, tqdm, sys
from rdkit import RDLogger, Chem
from rdkit.Chem import rdmolops, AllChem
import numpy as np
import pandas as pd
from utils.standardization import *

RDLogger.DisableLog('rdApp.*')
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


if __name__ == '__main__':
    csv_file = 'output.csv' # Customize file names
    generate_mol = pd.read_csv(csv_file, header=None).values.reshape(-1, 100).tolist() # reshape to (-1, mols)
    destination = './results'
    output_file = csv_file.replace('csv', 'pkl')
    os.makedirs(destination, exist_ok=True)

    pred_file = []
    pkl_file = f'{destination}/{output_file}'

    # construct mol obj
    gen_mols = generate_mol[0]
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