# HF-GNN
Source code for paper: "Hypergraph Functional Group Information-enhanced Graph Neural Networks for Protein-Ligand Binding Affinity Prediction"
![image](https://github.com/ILangXu/HF-GNN/assets/37317304/dcc11144-63ab-4552-9465-b7ee65a041a6)
## Abstract
High accurate protein-ligand binding affinity prediction plays important roles in drug discovery. Graph neural network (GNN) based protein-ligand binding affinity prediction is widely accepted, because molecules can be naturally represented as graph. However, existing methods seldom consider higher-order unpaired atomic information, which limits the use of intramolecular information. Furthermore, the distance difference of non-covalent interactions is ignored when aggregating intermolecular information. To overcome these limitations, we proposed a novel hypergraph functional group information-enhanced GNNs model named HF-GNN. HF-GNN learns intermolecular affinity information by an intramolecular learning module and a distance-aware intermolecular interaction information learning module. The intramolecular learning module fuses hypergraph functional group information with intramolecular paired atomic information and unpaired functional group information using graph convolution and hypergraph convolution. The distance-aware learning module updates node embeddings while preserving chemical bond information, and learn interaction force information between protein and ligand atoms using pooling layers. In comparation with other methods on several benchmark datasets, we demonstrate the superiority of HF-GNN.

## Dataset

All data used in this paper are publicly available and can be accessed here:

PDBbind v2016 and v2019: http://www.pdbbind.org.cn/download.php
2013 and 2016 core sets: http://www.pdbbind.org.cn/casf.php
The CSAR-HiQ dataset can be downloaded [here](http://www.csardock.org/).

## Requirements
python >= 3.7
networkx==2.5
numpy==1.19.2
pandas==1.1.5
pymol==0.1.0
rdkit==2022.9.2
torch==1.10.2
torch_geometric==2.0.3
scipy==1.5.2
