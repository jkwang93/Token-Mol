# Token-Mol 1.0
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

The repository is for **Token-Mol 1.0：tokenized drug design with large language model**

📢 Our Working has been published on [***Nature Communications***](https://www.nature.com/articles/s41467-025-59628-y)!

## Latest Update
> We have now supplemented the encoder for the pocket, so there is no need to obtain the GVP code from ResGen separately to obtain the pickle file of the protein embedding.  
> For information on how to use the encoder, please refer to the updated encoding part of **Generation** section.

## Environment

The codes for pre-training and fine-tuning have been test on **Windows and Linux** system.  
Codes for reinforcement learning with docking can stably run **only on Linux**.

### Python Dependencies
```
Python == 3.8
Pytorch == 1.13.1
RDKit == 2022.09.1
transformers == 4.24.0
networkx == 2.8.4
pandas == 1.5.3
scipy == 1.10.0
easydict (any version)
```

If want to customize pocket, additional dependencies should be install from ResGen:
```
#pyg
pyg_lib == 0.4.0
torch_cluster == 1.6.1
torch_scatter == 2.1.1
torch_sparse == 0.6.17
torch_spline_conv == 1.2.2

biopython
```

### Software Dependencies

**CUDA 11.6**

**For docking/reinforcement learning:**
| Software    | Source |
| ----------- | ----------- |
| QuickVina2  | [Download](https://qvina.github.io/)|
| ADFRsuite   | [Download](https://ccsb.scripps.edu/adfr/downloads/)|
| Open Babel  | [Download](https://open-babel.readthedocs.io/en/latest/Installation/install.html)|

> **Warning**: When utilizing ADFRsuite, it is necessary to add it to the system path, which causes the Python symlink to point to the Python 2.7 executable in ADFRsuite `bin` directory. Therefore, we recommend using `python3` commend instead of `python`.

## Data

You can download directly or get access to datasets according to the following table:  

| Task   | Source or access|
| ----------- | ----------- |
|Pre-training|[GEOM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF)|
|Conformation generation|[Tora3D](https://github.com/zimeizhng/Tora3D)|
|Property prediction|[MolecularNet](https://moleculenet.org/datasets-1) & [ADME](https://tdcommons.ai/single_pred_tasks/adme/)|
|Pocket-based generation|[ResGen](https://github.com/HaotianZhangAI4Science/ResGen)|

## Pre-training
We pre-trained model with **GPT2** architecture with [**HuggingFace🤗 Transformers**](https://huggingface.co/docs/transformers/model_doc/gpt2).  
**The pre-trained model can be directly download** at [Zenedo](https://zenodo.org/records/13428828).
> **Warning**: The model should be further fine-tuned before being used for any downstream tasks.

## Fine-tuning
  
Before running fine-tuning, move weight and configuration files of pre-trained model into `Pretrain_model` folder.  
Validation set have been preset in `data` folder, processed training set can be downloaded in [here](https://zenodo.org/records/13578841).

```
# Path of training set and validation set have been preset in the code
python pocket_fine_tuning_rmse.py --batch_size 32 --epochs 40 --lr 5e-3 --every_step_save_path Trained_model/pocket_generation
```

The finetuned model will be saved at `Trained_model/pocket_generation.pt`.

## Generation
You can directly download the fine-tuned checkpoint `pocket_generation.pt` [here](https://zenodo.org/records/13428828).

**Encoded** single pocket `example/ARA2A.pkl` or multiple pockets `data/test_protein_represent.pkl` can be used as input for generation.  

> Total number of generated molecules each pockets = batch size * epoch
```
# single pocket
python gen.py --model_path ./Trained_model/pocket_generation.pt --protein_path ./example/ARA2A.pkl --output_path ARA2A.csv --batch 25 --epoch 4
```
```
# multiple pockets
python gen.py --model_path ./Trained_model/pocket_generation.pt --protein_path ./data/test_protein_represent.pkl --output_path test.csv --batch 25 --epoch 4
```

#### Encoding
To encode your own custom pocket, you first need to download the [ResGen's trained model](https://drive.google.com/file/d/1RoVnHBLuPGFh2qGlCMoCRKtY-Rribp0j/view?usp=sharing) and place it in the `encoder/ckpt` folder.

Then run:
```
python encoder/encode.py --pdb_file <pdb_file> --sdf_file <sdf_file> --output_name <name>
```

Here `sdf_file` is the reference ligand within the pocket, and `output_name` is the filename of pickle file that you can found it at `encoder/embeddings/<output_name>.pkl`.

Original pockets in pdb format are attached at `data/test_set.zip` and `example/3rfm_a2a-pocket10.pdb`.

### Post-processing
The generated molecules should be processed and **converted from sequences to RDKit RDMol objects** and then used for subsequent applications.  

Output file `*.csv` will be input to `confs_to_mols_pocket.py` (or `confs_to_mols_pocket_multi.py` for preset test set). 

**We recommend manually changing the following two lines in the code.**

```
# Customize file names
csv_file = 'output.csv'
# Modify reshape dimension to the number of generated molecules, default 100
generate_mol = pd.read_csv(csv_file, header=None).values.reshape(-1, 100).tolist()
```

Then, run directly
```
python confs_to_mols_pocket.py
```
Processed molecules will save at `results` folder in `pickle` format.

## Reinforcement learning
### Reward score
You can customize reward score in `reinforce/reward_score.py`,and there are detailed description in the code.

### Running
> Before running reinforcement learning, target pocket need to be specified and encoded.

**We strongly recommend running the code on a multi-GPU machine as a too small batch size will result in an inability to perform an efficient gradient update.**

```
python ./reinforce/reinforce_multi_gpu.py --restore-from ./Trained_model/pocket_generation.pt --protein-dir ./reinforce/usecase --agent ./reinforce.pt --world-size 4 --batch-size 8 --num-steps 1000 --max-length 85 
```

|Args|Description|
|--|--|
|--restore-from|Model checkpoint|
|--protein-dir|Path of target pocket|
|--agent|Agent model save path|
|--world-size|World size (DDP)|
|--batch-size|Batch size on a single GPU|
|--num-steps|Total running steps|
|--max-length|Max length of sequence|

After testing, code under the above arguments can run on machine with at least 4*`Quadro RTX8000 (48GB vRAM)`.  

You can also check molecules generated in each steps in `every_steps_saved.pkl` and detailed reward terms in `reward_terms*.csv`.
