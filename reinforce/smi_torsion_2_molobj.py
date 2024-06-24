# -*- coding:utf-8 -*-
import copy
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import numpy as np
from reward_score import scoring
# from utils.standardization import get_torsion_angles, mmff_func, apply_changes

REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)

def log_error(err):
    print(err)
    return None


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def apply_changes(mol, values, rotable_bonds, conf_id):
    opt_mol = copy.copy(mol)
    [SetDihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]
    return opt_mol

def get_torsion_angles(mol):
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    nodes = set(G.nodes())
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list


def conformer_match(smiles, new_dihedrals):
    '''convert it like confs'''
    mol_rdkit = Chem.MolFromSmiles(smiles)

    AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=1)
    #if mol_rdkit.GetNumConformers() == 0:
        #print('wrong:', smiles)

    rotable_bonds = get_torsion_angles(mol_rdkit)

    if not rotable_bonds:
        return log_error("no_rotable_bonds")
    new_rdkit = apply_changes(mol_rdkit, new_dihedrals[:len(rotable_bonds)], rotable_bonds, 0)

    return new_rdkit


# todo 加速改过程，消融实验使用mmff优化
def construct_molobj(gen_matrix):
    mol_list = []
    for gen_mol in gen_matrix:
        try:
            smiles, torsion = gen_mol.split('GEO')
            smiles = smiles.replace(' ', '')
            torsion = np.array(torsion.split(' ')[1:]).astype(np.float64)
            pred_mol = conformer_match(smiles, torsion)
            mol_list.append(pred_mol)
        except Exception as e:
            mol_list.append(None)
    return mol_list
