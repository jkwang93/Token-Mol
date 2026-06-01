import os
import pickle
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

from utils.standardization import (
    RMSD,
    apply_changes,
    get_dihedral_vonMises,
    get_torsion_angles,
)

RDLogger.DisableLog("rdApp.*")
REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)


def get_von_mises_rms_torsion_to_mol(mol, mol_rdkit, rotable_bonds, conf_id):
    new_dihedrals = np.zeros(len(rotable_bonds))
    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = get_dihedral_vonMises(
            mol_rdkit,
            mol_rdkit.GetConformer(conf_id),
            r,
            mol.GetConformer().GetPositions(),
        )
    mol_rdkit = apply_changes(mol_rdkit, new_dihedrals, rotable_bonds, conf_id)
    return RMSD(mol_rdkit, mol, conf_id)


def log_error(err):
    print(err)
    return None


def conformer_match(smiles, new_dihedrals):
    """convert it like confs"""
    mol_rdkit = Chem.MolFromSmiles(smiles)

    AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=1)
    if mol_rdkit.GetNumConformers() == 0:
        print("wrong:", smiles)

    rotable_bonds = get_torsion_angles(mol_rdkit)

    if not rotable_bonds:
        return log_error("no_rotable_bonds")
    new_rdkit = apply_changes(
        mol_rdkit, new_dihedrals[: len(rotable_bonds)], rotable_bonds, 0
    )

    return new_rdkit


def read_data(path):
    data = []
    with open(path, "rb") as f:
        while True:
            try:
                aa = pickle.load(f)
                data.extend(aa)
            except EOFError:
                break
    return data


if __name__ == "__main__":
    count = 0

    eval_data_mol = read_data("data/test_mol_input.pkl")  # Test set
    generate_mol = (
        pd.read_csv("test.csv", header=None).values.reshape(-1, 100).tolist()
    )  # csv file of generated molecules
    val_df = pd.read_csv("val_list.txt")  # Test set list
    # protein_dir = '../data/crossdocked_pocket10' # Protein dataset
    protein_dir = "data/test_set"
    destination = "results/test"  # Destination path to store sdf files for docking and further analysis
    os.makedirs(destination, exist_ok=True)

    # 初始化一个空字典
    val_dict = {}

    # 使用循环将DataFrame的两列转换为字典
    for index, row in val_df.iterrows():
        key = row["smiles"]
        value = row["protein_path"]
        val_dict[key] = value

    for ind, eval_smiles in tqdm(enumerate(eval_data_mol), total=len(eval_data_mol)):
        # find protein path, you can read pdb here
        pdb_path = val_dict[eval_smiles]
        pred_file = []
        protein_idx = pdb_path.split("/")[0]

        # construct mol obj
        gen_mols = generate_mol[ind]
        for gen_mol in gen_mols:
            if gen_mol == np.nan:
                print("nan")

            try:
                smiles, torsion = gen_mol.split("GEO")
                smiles = smiles.replace(" ", "")
                torsion = np.array(torsion.split(" ")[1:]).astype(np.float64)
                pred_mol = conformer_match(smiles, torsion)
                if pred_mol:
                    pred_file.append(pred_mol)

            except Exception:
                continue

        pocket_dir = os.path.join(destination, protein_idx)
        os.makedirs(pocket_dir, exist_ok=True)
        os.makedirs(f"{pocket_dir}/SDF", exist_ok=True)

        # Write SDF file
        for i, mol in enumerate(pred_file):
            writer = Chem.SDWriter(f"{pocket_dir}/SDF/{i}.sdf")
            writer.write(mol)
            writer.close()

        # Copy pdb and original ligand to the destination path
        command1 = f"cp {protein_dir}/{pdb_path} {pocket_dir}"
        subprocess.run(
            command1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        conf = pdb_path[:-13] + ".sdf"
        command2 = f"cp {protein_dir}/{conf} {pocket_dir}"
        subprocess.run(
            command2, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
