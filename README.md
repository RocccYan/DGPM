# Empowering Dual-Level Graph Self-Supervised Pretraining with Motif Discovery (AAAI'24).
<a href='https://ojs.aaai.org/index.php/AAAI/article/view/28774'>

Quick Start
---
using pip: `requirements.txt`
`pip install -r requirements.txt`

Pretrain
---
the pretraining includes two steps, the first is to train node-level and motif-level models separately. The second step is to combine the two encoder and train with a cross-affiliation classification task.
For more details, please check the original paper.

Finetune
---
### Datasets for Transfer Learning

The datasets for transfer learning include ZINC [@irwin2012zinc] for pretraining and eight downstream datasets from MoleculeNet [@wu2018moleculenet].  
ZINC is a free database of commercially available compounds for virtual screening. ZINC contains over 230 million purchasable compounds in ready-to-dock 3D formats. It also contains over 750 million purchasable compounds that can be searched for analogs. For simplicity, we use a minimal set of node and bond features that unambiguously describe the two-dimensional structure of molecules, obtained via RDKit [@landrum2006rdkit].

- **Node features**:
  - Atom number: [1, 118]
  - Chirality tag: {unspecified, tetrahedral cw, tetrahedral ccw, other}

- **Edge features**:
  - Bond type: {single, double, triple, aromatic}
  - Bond direction: {–, endupright, enddownright}

Eight binary graph classification datasets are used to evaluate model performance:

- **BBBP** [@martins2012bayesian]: Blood-brain barrier penetration (membrane permeability).  
- **Tox21** [@mayr2016deeptox]: Toxicity data on 12 biological targets, including nuclear receptors and stress response pathways.  
- **ToxCast** [@richard2016toxcast]: Toxicology measurements based on over 600 in vitro high-throughput screenings.  
- **SIDER** [@kuhn2016sider]: Database of marketed drugs and adverse drug reactions (ADR), grouped into 27 system organ classes.  
- **ClinTox** [@novick2013sweetlead]: Qualitative data classifying drugs approved by the FDA and those that have failed clinical trials for toxicity reasons.  
- **MUV** [@gardiner2011effectiveness]: Subset of PubChem BioAssay obtained by applying a refined nearest neighbor analysis, designed for validating virtual screening techniques.  
- **HIV** [@riesen2008iam]: Experimentally measured abilities to inhibit HIV replication.  
- **BACE** [@subramanian2016computational]: Qualitative binding results for a set of inhibitors of human β-secretase 1.

## Settings

### Running Environment
All experiments are conducted on Linux servers equipped with an Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz, 88GB RAM, and NVIDIA V100 GPUs. Models of pretraining and downstream tasks are implemented in *PyTorch* version 1.8.1, *Pytorch Geometric* 2.2.0 with *CUDA* version 10.1, *scikit-learn* version 0.24.1 and *Python* 3.7.  
For node feature reconstruction, we implement our model based on the code in [https://github.com/THUDM/GraphMAE](https://github.com/THUDM/GraphMAE).  
For molecular property prediction, we implement our model based on the code in [https://github.com/snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns).

---

### Model Configuration
For unsupervised representation learning, we search the optimal number of EdgePool layers from `{2,3,4,5,6,7,8}` with the allowed maximum size of motifs from `2^2` to `2^8` for different datasets.  
For evaluation, the parameter `c` of SVM is searched in `{10^-3, 10^-2, 0.1, 1, 10}`. The detailed hyper-parameters by datasets are shown in [Table 2](#table-2).  

For transfer learning of molecule property prediction, we search the optimal number of EdgePool layers from `{2,3,4}`.  
Besides, we adopt a single-layer MLP as discriminator to adapt to the different numbers of prediction tasks among downstream datasets.  
The detailed model configurations of transfer learning for molecular property prediction are shown in [Table 1](#table-1).

---

#### Table 1. Hyper-parameters of transfer learning molecular property prediction experiments <a id="table-1"></a>

| Hyper-parameters   | ZINC  | BBBP  | Tox21 | ToxCast | SIDER | ClinTox | MUV   | HIV   | BACE  |
|--------------------|-------|-------|-------|---------|-------|---------|-------|-------|-------|
| masking rate       | 0.50  | -     | -     | -       | -     | -       | -     | -     | -     |
| # EdgePool layers  | 2     | 2     | 2     | 2       | 2     | 2       | 2     | 2     | 2     |
| hidden size        | 128   | 128   | 128   | 128     | 128   | 128     | 128   | 128   | 128   |
| # MLP layers       | -     | 1     | 1     | 1       | 1     | 1       | 1     | 1     | 1     |
| max epoch          | 100   | 10    | 10    | 10      | 10    | 10      | 10    | 10    | 10    |
| batch size         | 256   | 64    | 128   | 128     | 32    | 32      | 128   | 128   | 32    |
| pooling            | mean  | mean  | mean  | mean    | mean  | mean    | mean  | mean  | mean  |
| learning rate      | 0.005 | 0.001 | 0.001 | 0.001   | 0.001 | 0.001   | 0.001 | 0.001 | 0.001 |
| weight decay       | 2e-04 | 1e-04 | 1e-04 | 1e-04   | 1e-04 | 1e-04   | 1e-04 | 1e-04 | 1e-04 |

---

#### Table 2. Hyper-parameters of unsupervised graph classification experiments <a id="table-2"></a>

| Hyper-parameters   | IMDB-B | IMDB-M | PROTEINS | COLLAB | MUTAG | REDDIT-B | NCI1  |
|--------------------|--------|--------|----------|--------|-------|----------|-------|
| masking rate       | 0.50   | 0.50   | 0.50     | 0.75   | 0.75  | 0.75     | 0.25  |
| # EdgePool layers  | 3      | 3      | 3        | 5      | 3     | 6        | 3     |
| hidden size        | 128    | 128    | 128      | 128    | 128   | 128      | 128   |
| max epoch          | 100    | 100    | 100      | 200    | 100   | 200      | 100   |
| batch size         | 256    | 256    | 256      | 128    | 256   | 128      | 256   |
| pooling            | mean   | mean   | mean     | mean   | mean  | mean     | mean  |
| learning rate      | 0.005  | 0.001  | 0.005    | 0.005  | 0.005 | 0.001    | 0.005 |
| weight decay       | 2e-04  | 2e-05  | 2e-04    | 1e-04  | 2e-04 | 2e-05    | 2e-04 |
