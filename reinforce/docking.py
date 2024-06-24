import os.path as osp
import os
import sys
from pathlib import Path
from rdkit import Chem, RDLogger
from easydict import EasyDict
import subprocess
import shutil
import random

RDLogger.DisableLog('rdApp.*')

def write_sdf(mol, sdf_file, verbose=0):
    writer = Chem.SDWriter(sdf_file)
    if type(mol) == list:
        for i in range(len(mol)):
            writer.write(mol[i])
        writer.close()
    else:
        writer.write(mol)
    if verbose == 1:
        print('saved successfully at {}'.format(sdf_file))

def prepare_target(work_dir, protein_file_name, verbose=0):
    '''
    work_dir is the dir which .pdb locates
    protein_file_name: .pdb file which contains the protein data
    '''
    protein_file = osp.join(work_dir, protein_file_name)
    command = 'prepare_receptor -r {protein} -o {protein_pdbqt}'.format(protein=protein_file,
                                                            protein_pdbqt = protein_file+'qt')
    if osp.exists(protein_file+'qt'):
        return protein_file+'qt'
        
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    stdout, stderr = proc.communicate()

    if verbose:
        if osp.exists(protein_file+'qt'):
            print('successfully prepare the target')
        else: 
            print('failed')
    return protein_file+'qt'

def prepare_ligand(work_dir, lig_sdf, verbose=0):
    lig_name = lig_sdf
    lig_mol2 = lig_sdf[:-3]+'mol2'
    now_cwd = os.getcwd()
    lig_sdf = osp.join(work_dir, lig_sdf)
    cwd_mol2 = osp.join(now_cwd, lig_mol2)
    work_mol2 = osp.join(work_dir, lig_mol2)
    command = '''obabel {lig} -O {lig_mol2}'''.format(lig=lig_sdf,
                                                        lig_mol2 = work_mol2)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()
    os.remove(lig_sdf)

    shutil.copy(work_mol2, now_cwd)
    command = '''prepare_ligand -l {lig_mol2}'''.format(lig_mol2=cwd_mol2)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()

    lig_pdbqt = lig_name[:-3]+'pdbqt'
    cwd_pdbqt = osp.join(now_cwd, lig_pdbqt)
    work_pdbqt = osp.join(work_dir, lig_pdbqt)
    os.remove(cwd_mol2)
    os.remove(work_mol2)
    if osp.exists(work_pdbqt):
        os.remove(work_pdbqt)
    shutil.move(cwd_pdbqt, work_dir)
    if os.path.exists(lig_pdbqt):
        if verbose:
            print('prepare successfully !')
        else:
            print('generation failed!')
    return lig_pdbqt


def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:,0].mean()
    centroid_y = lig_xyz[:,1].mean()
    centroid_z = lig_xyz[:,2].mean()
    return centroid_x, centroid_y, centroid_z


def docking_with_sdf(work_dir, protein_pdbqt, lig_pdbqt, centroid, verbose=0, out_lig_sdf=None, save_pdbqt=False):
    '''
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''
    # prepare target
    lig_pdbqt = osp.join(work_dir, lig_pdbqt)
    protein_pdbqt = osp.join(work_dir, protein_pdbqt)
    cx, cy, cz = centroid
    out_lig_sdf_dirname = osp.dirname(lig_pdbqt)
    out_lig_pdbqt_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.pdbqt'
    out_lig_pdbqt = osp.join(out_lig_sdf_dirname, out_lig_pdbqt_filename) 
    if out_lig_sdf is None:
        out_lig_sdf_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.sdf'
        out_lig_sdf = osp.join(out_lig_sdf_dirname, out_lig_sdf_filename) 
    else:
        out_lig_sdf = osp.join(work_dir, out_lig_sdf)

    command = '''qvina2.1 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 20 --size_y 20 --size_z 20 \
        --cpu 40 \
        --out {out_lig_pdbqt} \
        --exhaustiveness {exhaust}
        obabel {out_lig_pdbqt} -O {out_lig_sdf} -h'''.format(receptor_pre = protein_pdbqt,
                                            ligand_pre = lig_pdbqt,
                                            centroid_x = cx,
                                            centroid_y = cy,
                                            centroid_z = cz,
                                            out_lig_pdbqt = out_lig_pdbqt,
                                            exhaust = 8,
                                            out_lig_sdf = out_lig_sdf)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()
    stdout, stderr = proc.communicate()

    os.remove(lig_pdbqt)
    if not save_pdbqt:
        os.remove(out_lig_pdbqt)
    
    if verbose: 
        if os.path.exists(out_lig_sdf):
            print('searchable docking is finished successfully')
        else:
            print('docing failed')
    return out_lig_sdf

def docking_in_gpu(work_dir, protein_pdbqt, lig_pdbqt, centroid, verbose=0, out_lig_sdf=None, save_pdbqt=False):
    '''
    Docking with Vina-GPU/QuickVina2-GPU/QuickVina-W-GPU 2.1, default in Vina-GPU 2.1.
    See Ding J, Tang S, Mei Z, et al. Vina-GPU 2.0: Further Accelerating AutoDock Vina and Its Derivatives with Graphics Processing Units.
    Journal of Chemical Information and Modeling, 2023, 63(7): 1982-1998.
    Install in https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1
    
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''
    # prepare target
    work_dir = work_dir
    lig_pdbqt =  osp.join(work_dir, lig_pdbqt)
    protein_pdbqt = osp.join(work_dir, protein_pdbqt)
    # get centroid
    cx, cy, cz = centroid
    out_lig_sdf_dirname = osp.dirname(lig_pdbqt)
    out_lig_pdbqt_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.pdbqt'
    out_lig_pdbqt = osp.join(out_lig_sdf_dirname, out_lig_pdbqt_filename)
    if out_lig_sdf is None:
        out_lig_sdf_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.sdf'
        out_lig_sdf = osp.join(out_lig_sdf_dirname, out_lig_sdf_filename) 
    else:
        out_lig_sdf = osp.join(work_dir, out_lig_sdf)
    if os.path.exists(out_lig_pdbqt):
        #print('searchable docking is finished successfully', file= sys.stderr)
        return out_lig_pdbqt
    
    # Warning: Change docking method and opencl_binary_path by your own settings!!!

    command = ''' AutoDock-Vina-GPU-2-1 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 20 --size_y 20 --size_z 20 \
        --out {out_lig_pdbqt} \
        --thread 8000 \
        --opencl_binary_path /home/sorui/software/Vina-GPU-2.1-main/AutoDock-Vina-GPU-2.1\
        '''.format(receptor_pre = protein_pdbqt,
                    ligand_pre = lig_pdbqt,
                    centroid_x = cx,
                    centroid_y = cy,
                    centroid_z = cz,
                    out_lig_pdbqt = out_lig_pdbqt,
                    )
    subprocess.run(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )

    command2 = f'obabel {out_lig_pdbqt} -O {out_lig_sdf} -h'
    subprocess.run(command2, shell=True)
    os.remove(lig_pdbqt)
    if not save_pdbqt:
        os.remove(out_lig_pdbqt)
    
    if verbose: 
        if os.path.exists(out_lig_sdf):
            print('searchable docking is finished successfully')
        else:
            print('dock.ing failed')
    return out_lig_sdf


def get_result(docked_sdf, ref_mol=None):
    suppl = Chem.SDMolSupplier(docked_sdf,sanitize=False)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        results.append(EasyDict({
            'mol': ref_mol,
            'dock_rdmol': mol,
            'affinity': float(line[0]),
        }))
        if i == 0:
            break
    return results[0]


def docking_score(mol, protein_dir, rank=None, save_sdf=False, use_gpu=False) -> float:

    protein_dir = Path(protein_dir) # Path with protein and ligands files
    protein_dir.mkdir(parents=True, exist_ok=True)
    protein_file = str(list(protein_dir.glob('*.pdb'))[0])
    protein_filename = protein_file.split('/')[-1]
    ori_lig_file = str(next(protein_dir.glob('*_ligand.sdf'))) # We suggest naming the ori ligand like this to avoid mistakes

    # Prepare protein
    prepare_target(protein_dir, protein_filename)
    protein_pdbqt = protein_file.split('/')[-1]+'qt'

    # Get the centroid of docking 
    centroid = sdf2centroid(ori_lig_file)

    # Construct sdf file of generated molecule
    random_id = str(random.randint(0, 255))
    if rank:
        sdf_file = f'gene_{rank}_{random_id}.sdf'
    else:
        sdf_file = f'gene_{random_id}.sdf'
    sdf_dir = os.path.join(protein_dir, sdf_file)
    write_sdf(mol, sdf_dir)

    # Preparing and docking
    lig_pdbqt = prepare_ligand(protein_dir, Path(sdf_dir).name)
    if use_gpu:
        docked_sdf = docking_in_gpu(str(protein_dir), protein_pdbqt, lig_pdbqt, centroid)
    else:
        docked_sdf = docking_with_sdf(str(protein_dir), protein_pdbqt, lig_pdbqt, centroid)
    result = get_result(docked_sdf, ref_mol=mol)

    # Save docking results in sdf (Optional)
    if not save_sdf:
        Path(docked_sdf).unlink()

    # Remove docking pdbqt file        

    return result

if __name__ == '__main__':
    import pickle
    data = []
    with open('./results/multi_rl_1e-5/every_steps_saved.pkl', 'rb') as f:
        while True:
            try:
                seq = pickle.load(f)
                data.extend(seq.items())
            except EOFError:
                break
    
    mols_dict = {}
    for n in range(len(data)):
        if n % 4 == 0:
            steps_n = []
        steps_n.extend(data[n][-1])
        if n % 4 == 3:
            mols_dict[f'Step {n//4 + 1}'] = steps_n
    score = []
    for mol in mols_dict['Step 270']:
        if mol:
            score.append(docking_score(mol, protein_dir='./usecase_protein_embedding/CDK4', use_gpu=True))
        else:
            score.append(None)
    print(score)