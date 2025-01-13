'''
Author: Rui Qin
Date: 2023-12-26 21:13:06
LastEditTime: 2024-05-18 14:39:28
Description: 
'''
import sys, os
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase, RDLogger, RDConfig
from rdkit.Chem import QED
#from clash import check_geometry
from docking import docking_score
#sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
#import sascorer as sa
RDLogger.DisableLog('rdApp.*')
rdBase.DisableLog('rdApp.error')

class Score:
    """
    Calculate the necessary properties for rewarding score.
    In this case, we apply a properties filter, QED score, 
    stereochemisty clash and docking score with QuickVina2 as reward terms.
    You can also DIY any reward terms if you like :-)
    """
    def __init__(self, mol, protein_dir, rank=None):
        self.mol = mol
        self.smi = Chem.MolToSmiles(mol)
        self.qed = QED.qed(mol)
        #self.sa = 1 - 0.1 * sa.calculateScore(mol)
        #self.logp = Crippen.MolLogP(mol)
        #self.mw = Descriptors.ExactMolWt(mol)
        #self.clash = check_geometry(mol)
        self.dock = docking_score(mol, protein_dir, rank)
        self.vina_score = self.dock['affinity']

def scoring(mol_list: list, protein_dir, rank=None):
    """
    Custom the scoring function with proper weights and parameters.

    Args:
        mol_list: NDArray with rdchem.Mol or Nonetype objects.
        Whose length equals to 'seqs[unique_idxs]'.
        protein_dir: Protein Path.

    Output:
        all_scores: List object with all reward scores.
        aver_terms: DataFrame object with average terms of all valid molecules.
        poses: Dict object contains original rdmol object, docking score and docking pose in rdmol object.
    """

    #Custom the predefined parameters
    init_vina = -8.0
    vina_weight = 5
    all_scores, aver_terms, poses = [], [], []

    # DIY the reward here!
    for mol in mol_list:
        # Invalid mol -> 0
        if not mol:
            all_scores.append(0)
            poses.append(None)
            continue
       
        score = Score(mol, protein_dir, rank)
        terms_list = [score.vina_score]
        aver_terms.append(terms_list)
        poses.append(score.dock)
        
        reward = max(0.1, init_vina - score.vina_score + 0.1)
        if reward > 0.1:
            if score.qed >= 0.5:
                reward += 1
            reward *= vina_weight

        all_scores.append(reward)

    all_scores = np.array(all_scores)
    valid = len([mol for mol in mol_list if mol])/len(mol_list)
    
    # Output setting for reward terms.
    index = ['Score','Validity', 'Vina Score']
    aver_score = np.mean(all_scores)
    aver_terms = np.insert(arr=np.mean(aver_terms, axis=0),
                           obj=0, 
                           values=(aver_score, valid))
    try:
        aver_terms = pd.DataFrame(dict(zip(index, aver_terms)), index=[0])
    except: # all generated seqences all invaild
        aver_terms = np.array([0])
        aver_terms = pd.DataFrame(dict(zip(index, aver_terms)), index=[0])

    return all_scores, aver_terms, poses

def eval(mol_list: list, protein_dir, rank=None):
    """
    Eval the model with specific metrics.

    Args:
        mol_list: NDArray with rdchem.Mol or Nonetype objects.
        Whose length equals to 'seqs[unique_idxs]'.
        protein_dir: Protein Path.

    Output:
        aver_terms: List with average terms of all valid molecules.
    """
    terms, valid_list = [], []
    for mol in mol_list:
        if mol:
            #Store the reward terms for further analysis
            score = Score(mol, protein_dir, rank)
            terms_list = [score.vina_score]
            terms.append(terms_list)
            valid_list.append(mol)

    aver_terms = np.array(terms)
    valid = len(valid_list)/len(mol_list)

    aver_terms = np.mean(aver_terms, axis=0).tolist()
    aver_terms.insert(0, valid)
    return aver_terms