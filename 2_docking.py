import glob
import os
import os.path as osp
import pickle
import shutil
import subprocess
from pathlib import Path

from easydict import EasyDict
from rdkit import Chem
from tqdm import tqdm


def read_sdf(sdf_files):
    mols = []
    with Chem.SDMolSupplier(sdf_files) as supplier:
        for mol in supplier:
            mols.append(mol)
    return mols[0]


def write_sdf(mol, sdf_file, verbose=1):
    writer = Chem.SDWritter(sdf_file)
    if type(mol) is list:
        for i in range(len(mol)):
            writer.write(mol[i])
        writer.close()
    else:
        writer.write(mol)
    if verbose == 1:
        print("saved successfully at {}".format(sdf_file))


def prepare_target(work_dir, protein_file_name, verbose=1):
    """
    work_dir is the dir which .pdb locates
    protein_file_name: .pdb file which contains the protein data
    """
    protein_file = osp.join(work_dir, protein_file_name)
    command = "prepare_receptor -r {protein} -o {protein_pdbqt}".format(
        protein=protein_file, protein_pdbqt=protein_file + "qt"
    )
    if osp.exists(protein_file + "qt"):
        return protein_file + "qt"

    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    print(f"Target Prepare out: {stdout}")
    print(f"Target Prepare err: {stderr}")

    if verbose:
        if osp.exists(protein_file + "qt"):
            print("successfully prepare the target")
        else:
            print("failed")
    return protein_file + "qt"


def prepare_ligand(work_dir, lig_sdf, verbose=1):
    lig_name = lig_sdf
    lig_mol2 = lig_sdf[:-3] + "mol2"
    now_cwd = os.getcwd()
    lig_sdf = osp.join(work_dir, lig_sdf)
    cwd_mol2 = osp.join(now_cwd, lig_mol2)
    work_mol2 = osp.join(work_dir, lig_mol2)
    command = """obabel {lig} -O {lig_mol2}""".format(lig=lig_sdf, lig_mol2=work_mol2)
    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.communicate()
    shutil.copy(work_mol2, now_cwd)
    command = """prepare_ligand -l {lig_mol2}""".format(lig_mol2=work_mol2)
    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.communicate()
    stdout, stderr = proc.communicate()
    print(f"Ligand out: {stdout}")
    print(f"Ligand err: {stderr}")

    lig_pdbqt = lig_name[:-3] + "pdbqt"
    cwd_pdbqt = osp.join(now_cwd, lig_pdbqt)
    work_pdbqt = osp.join(work_dir, lig_pdbqt)
    os.remove(cwd_mol2)
    os.remove(work_mol2)
    if osp.exists(work_pdbqt):
        os.remove(work_pdbqt)
    shutil.move(cwd_pdbqt, work_dir)
    if os.path.exists(lig_pdbqt):
        if verbose:
            print("prepare successfully !")
        else:
            print("generation failed!")
    return lig_pdbqt


def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:, 0].mean()
    centroid_y = lig_xyz[:, 1].mean()
    centroid_z = lig_xyz[:, 2].mean()
    return centroid_x, centroid_y, centroid_z


def docking_with_sdf(
    work_dir,
    protein_pdbqt,
    lig_pdbqt,
    centroid,
    verbose=1,
    out_lig_sdf=None,
    save_pdbqt=False,
):
    """
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    """
    # prepare target
    lig_pdbqt = osp.join(work_dir, lig_pdbqt)
    protein_pdbqt = osp.join(work_dir, protein_pdbqt)
    cx, cy, cz = centroid
    out_lig_sdf_dirname = osp.dirname(lig_pdbqt)
    out_lig_pdbqt_filename = osp.basename(lig_pdbqt).split(".")[0] + "_out.pdbqt"
    out_lig_pdbqt = osp.join(out_lig_sdf_dirname, out_lig_pdbqt_filename)
    if out_lig_sdf is None:
        out_lig_sdf_filename = osp.basename(lig_pdbqt).split(".")[0] + "_out.sdf"
        out_lig_sdf = osp.join(out_lig_sdf_dirname, out_lig_sdf_filename)
    else:
        out_lig_sdf = osp.join(work_dir, out_lig_sdf)

    command = """qvina2.1 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 20 --size_y 20 --size_z 20 \
        --cpu 40 \
        --out {out_lig_pdbqt} \
        --exhaustiveness {exhaust}
        obabel {out_lig_pdbqt} -O {out_lig_sdf} -h""".format(
        receptor_pre=protein_pdbqt,
        ligand_pre=lig_pdbqt,
        centroid_x=cx,
        centroid_y=cy,
        centroid_z=cz,
        out_lig_pdbqt=out_lig_pdbqt,
        exhaust=8,
        out_lig_sdf=out_lig_sdf,
    )
    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.communicate()
    stdout, stderr = proc.communicate()
    # print(f'Dock out: {stdout}')
    print(f"Dock err: {stderr}")

    os.remove(lig_pdbqt)
    if not save_pdbqt:
        os.remove(out_lig_pdbqt)

    if verbose:
        if os.path.exists(out_lig_sdf):
            print("searchable docking is finished successfully")
        else:
            print("docking failed")
    return out_lig_sdf


def get_result(docked_sdf, origin_mol, filename):
    suppl = Chem.SDMolSupplier(docked_sdf, sanitize=False)
    results = []
    for mol in suppl:
        if mol is None:
            continue
        line = mol.GetProp("REMARK").splitlines()[0].split()[2:]
        results.append(
            EasyDict(
                {
                    "filename": filename,
                    "rdmol": origin_mol,
                    "pose": mol,
                    "affinity": float(line[0]),
                }
            )
        )
        break
    # os.remove(docked_sdf)
    return results


def write_pkl(data_list, pkl_file):
    with open(pkl_file, "wb") as f:
        pickle.dump(data_list, f)


if __name__ == "__main__":
    path = "results/test"
    files = glob.glob(f"{path}/*/")

    for i in tqdm(range(len(files))):
        work_dir = files[i] + "/SDF/"
        protein_file = glob.glob(files[i] + "/*.pdb")[0]
        ori_lig_file = max(glob.glob(files[i] + "/*.sdf"), key=len, default="")
        protein_filename = protein_file.split("/")[-1]

        command = f"cp {protein_file} {work_dir}"
        subprocess.run(command, shell=True)
        # protein_filename = work_dir + protein_filename
        prepare_target(work_dir, protein_filename, verbose=0)
        protein_pdbqt = protein_filename + "qt"
        centroid = sdf2centroid(ori_lig_file)
        result_list = []
        for j in [
            f for f in glob.glob(f"{work_dir}/*.sdf") if not f.endswith("_out.sdf")
        ]:
            try:
                origin_mol = read_sdf(j)
                sdf = Path(j).name
                if not Path(j.replace(".sdf", "_out.sdf")).exists():
                    lig_pdbqt = prepare_ligand(work_dir, sdf, verbose=0)
                    docked_sdf = docking_with_sdf(
                        work_dir, protein_pdbqt, lig_pdbqt, centroid, verbose=0
                    )
                else:
                    docked_sdf = j.replace(".sdf", "_out.sdf")
                # result = get_result(docked_sdf=docked_sdf, origin_mol=origin_mol, filename=sdf)
                # result_list.append(result)
            except:
                ...

        # write_pkl(result_list, files[i]+'/gen_docking_results.pkl')
