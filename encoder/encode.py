# -*- coding:utf-8 -*-
# @Author: jikewang
# @Time: 2023/10/10 16:46
# @File: gen_protein_embeding_haotian.py
from torch_geometric.data import Batch

from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from rdkit import Chem

from utils.feats.protein import get_protein_feature_v2
from Bio.PDB import NeighborSearch, Selection

from models.ResGen import ResGen
from utils.transforms import *
from utils.misc import load_config, transform_data
from utils.reconstruct import *
import sys

def read_sdf(file):
    supp = Chem.SDMolSupplier(file)
    return [i for i in supp]


from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import warnings


def pdb_to_pocket_data(pdb_file, box_size=10.0, mol_file=None, center=None):
    '''
    use the sdf_file as the center
    '''
    if mol_file is not None:
        prefix = mol_file.split('.')[-1]
        if prefix == 'mol2':
            center = Chem.MolFromMol2File(mol_file, sanitize=False).GetConformer().GetPositions()
            center = np.array(center)
        elif prefix == 'sdf':
            supp = Chem.SDMolSupplier(mol_file, sanitize=False)
            center = supp[0].GetConformer().GetPositions()
        else:
            print('The File type of Molecule is not support')
    elif center is not None:
        center = np.array(center)
    else:
        print('You must specify the original ligand file or center')
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('target', pdb_file)[0]
    atoms = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atoms)
    close_residues = []
    dist_threshold = box_size
    for a in center:
        close_residues.extend(ns.search(a, dist_threshold, level='R'))

    if len(close_residues) < 8:
        print('The structures studied are not well defined, maybe the box center is invalid.')
        return None

    close_residues = Selection.uniqueify(close_residues)
    protein_dict = get_protein_feature_v2(close_residues)

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein_dict,
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )
    return data


parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--config', type=str, default='./configs/sample.yml'
# )
parser.add_argument('--config', type=str, default='./encoder/configs/train_res.yml')
parser.add_argument('--ckpt', type=str, default='./encoder/ckpt/val_172.pt')
parser.add_argument(
    '--outdir', type=str, default='generation',
)

parser.add_argument(
    '--device', type=str, default='cuda',
)

parser.add_argument(
    '--check_point', type=str, default='encoder/ckpt/val_172.pt',
    help='load the parameter'
)

parser.add_argument(
    '--pdb_file', action='store', required=True, type=str,
    help='protein file specified for generation'
)

parser.add_argument(
    '--sdf_file', action='store', required=True, type=str,
    help='original ligand sdf_file, only for providing center'
)

parser.add_argument(
    '--output_name', action='store', required=True, type=str,
    help='original ligand sdf_file, only for providing center'
)

parser.add_argument(
    '--center', action='store', required=False, type=str, default=None,
    help='provide center explcitly, e.g., 32.33,25.56,45.67'
)

parser.add_argument(
    '--name_by_pdb', required=False, type=str, default=False,
    help='as its name'
)

args = parser.parse_args()
config = load_config(args.config)

# define the model and transform function (process the data again)
contrastive_sampler = ContrastiveSample()
ligand_featurizer = FeaturizeLigandAtom()
transform = Compose([
    RefineData(),
    LigandCountNeighbors(),
    ligand_featurizer
])

# define the pocket data for generation
if args.sdf_file is not None:
    mol = read_sdf(args.sdf_file)[0]
    atomCoords = mol.GetConformers()[0].GetPositions()
    data = pdb_to_pocket_data(args.pdb_file, center=atomCoords, box_size=10)

if args.center is not None:
    center = np.array([[float(i) for i in args.center.split(',')]])
    data = pdb_to_pocket_data(args.pdb_file, center=center, box_size=10)

if data is None:
    sys.exit('pocket residues is None, please check the box you choose or the PDB file you upload')

ckpt = torch.load(args.ckpt, map_location=args.device)

mask = LigandMaskAll()
composer = Res2AtomComposer(27, ligand_featurizer.feature_dim, ckpt['config'].model.encoder.knn)
masking = Compose([
    mask,
    composer
])

data = transform(data)
data = transform_data(data, masking)

model = ResGen(
    ckpt['config'].model,
    num_classes = 7,
    num_bond_types = 3,
    protein_res_feature_dim = (27,3),
    ligand_atom_feature_dim = (13,1),
).to(args.device)

model.load_state_dict(ckpt['model'])


def get_resgen_protein_embeding(batch):
    protein_vec = batch.pkt_node_vec
    protein_feature = batch.pkt_node_sca.float()
    protein_pos = batch.pkt_node_xyz.float()

    from torch_geometric.nn.pool import knn_graph
    protein_knn_edge_index = knn_graph(protein_pos, k=48, flow='target_to_source', num_workers=16)
    protein_knn_edge_feature = torch.cat([
        torch.ones([len(protein_knn_edge_index[0]), 1], dtype=torch.long),
        torch.zeros([len(protein_knn_edge_index[0]), 3], dtype=torch.long),
    ], dim=-1)

    # first emb
    h_protein = model.protein_res_emb([protein_feature, protein_vec])

    h_compose = model.encoder(
        node_attr=h_protein,
        pos=protein_pos,
        edge_index=protein_knn_edge_index,
        edge_feature=protein_knn_edge_feature,
    )
    return h_compose


wrong_num = 0
model.eval()

with torch.no_grad():
    root = f'./encoder/embeddings'
    os.makedirs(root, exist_ok=True)
    val_list = []
    batch = Batch.from_data_list([data], follow_batch=[])  # batch only contains one data
    batch = batch.to(args.device)
    protein_input = []
    mols_input = []

    batch = batch.to(args.device)

    ''' 
    protein_represent只使用了self.protein_res_emb来得到protein的embedding

        def get_loss(self, compose_feature, compose_vec, idx_protein):

            protein_nodes = (compose_feature[idx_protein], compose_vec[idx_protein])

            h_protein = self.protein_res_emb(protein_nodes)

            return h_protein[0]
    '''
    protein_represent = get_resgen_protein_embeding(batch)[0].cpu().numpy()
    protein_input.append(protein_represent)

    output = args.output_name
    with open(f'{root}/{output}.pkl', 'ab') as file:
        pickle.dump(protein_input, file)

print(wrong_num)
