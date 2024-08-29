# Token-Mol
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Token-Mol 1.0：tokenized drug design with large language model

[Preprint](https://arxiv.org/abs/2407.07930)


## Environment

The code has been test on **Windows and Linux** system.

### Python Dependencies
```
Python == 3.8
Pytorch >= 1.13.1
RDKit >= 2022.09.1
transformers >= 4.24.0
networkx >= 2.8.4
pandas >= 1.5.3
scipy == 1.10.0
easydict (any version)
```

### Software Dependencies

CUDA 11.6

**For docking/reinforcement learning:**
| Software    | URL |
| ----------- | ----------- |
| QuickVina2  | [Download](https://qvina.github.io/)|
| ADFRsuite   | [Download](https://ccsb.scripps.edu/adfr/downloads/)|
| Open Babel  | [Download](https://open-babel.readthedocs.io/en/latest/Installation/install.html)|