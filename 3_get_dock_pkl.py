"""
Author: Rui Qin
Date: 2024-03-15 19:56:42
LastEditTime: 2025-08-20 16:01:11
Description:
"""

import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, RDConfig
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer as sa


def get_result(docked_sdf):
    suppl = Chem.SDMolSupplier(docked_sdf, sanitize=False)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp("REMARK").splitlines()[0].split()[2:]
        results.append(
            EasyDict(
                {
                    "pose": mol,
                    #'mode_id': i,
                    "affinity": float(line[0]),
                    #'rmsd_lb': float(line[1]),
                    #'rmsd_ub': float(line[2]),
                    #'rmsd_ref': rmsd
                }
            )
        )
    return results


def get_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    mols = [i for i in supp][0]
    return mols


def obey_lipinski(mol):
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp := Crippen.MolLogP(mol) >= -2) & (logp <= 5)
    rule_5 = Lipinski.NumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def write_pkl(list, filename):
    with open(filename, "wb") as f:
        pickle.dump(list, f)


def standrize(mol):
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))


if __name__ == "__main__":
    model_path = "results/test"
    model_name = Path(model_path).stem

    overall = []
    pockets = glob.glob(f"{model_path}/*")
    # sdfs = glob.glob(f'{model_path}/*/*.sdf')
    for pok in tqdm(pockets, total=len(pockets)):
        results = []
        sdfs = glob.glob(f"{pok}/SDF/*.sdf")
        sdf_list = sorted([f for f in sdfs if not f.endswith("_out.sdf")])
        dock_list = list(set(sdfs) - set(sdf_list))
        for sdf, dock in zip(sdf_list, dock_list):
            try:
                conf = get_sdf(sdf)
                mol = standrize(conf)
                index = Path(sdf).stem
                # Chem.GetSymmSSSR(mol) # If met 'RingInfo not initialized' error, delete the hashtag before this line
                mol_dict = EasyDict(
                    {
                        "mol": mol,
                        "conf": conf,
                        "index": index,
                        "qed": QED.qed(mol),
                        "SAS": sa.calculateScore(mol),
                        "MW": Descriptors.MolWt(mol),
                        "LogP": Crippen.MolLogP(mol),
                        "TPSA": Descriptors.TPSA(mol),
                        "Lp5": obey_lipinski(mol),
                    }
                )
                result = get_result(dock)[0]
                results.append({**mol_dict, **result})
            except:
                pass
        overall.append(results)

    save_file = f"results/{model_name}_results.pkl"
    write_pkl(overall, save_file)
