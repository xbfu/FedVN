## Introduction
This is our implementation of our paper *Virtual Nodes Can Help: Tackling Distribution Shifts in Federated Graph Learning.*

**TL;DR**: A novel perspective to tackle distribution shifts in federated graph learning through client-specific graph augmentation with virtual nodes.  

**Framework Overview**:
The following figure demonstrates an overview of our proposed FedVN. 
The intuition of FedVN is to let the clients manipulate their local data through learnable graph augmentation strategies 
in order that the global GNN model can be trained over identical manipulated graph data without any distribution shift 
across clients. To achieve this, the key point is to design a proper scheme for graph augmentation. Inspired by recent
studies about virtual nodes in graph learning, we propose to learn graph augmentation with extra virtual nodes to 
eliminate distribution shifts in FGL. More specifically, the clients in FedVN collaboratively learn a set of shared 
virtual nodes while training a global GNN model. Considering the cross-client distribution shift, FedVN enables each 
client to learn a personalized edge predictor that determines how the VNs connect its local graphs.


<div align=center><img src="https://anonymous.4open.science/r/FedVN-10024/FedVN.png" height="200px"/></div>

## Parameters

| Parameter         |           Description                                 | 
|-------------------|-------------------------------------------------------|
| dataset_name      |   Dataset to use (default: motif)                     |
| domain            |   Data partition setting (default: basis)             |
| data_path         |   Data directory (default: ./data/)                   |
| num_local_graphs  |   Number of local graphs (default: 1000)              |
| epochs            |   Number of local epochs (default: 1)                 |
| batch_size        |   Batch size of training (default: 32)                |
| lr                |   Learning rate (default: 0.001)                      |
| rounds            |   Number of rounds (default: 200)                     |
| hidden            |   Hidden size of the mode (default: 100)              |
| hidden_eg         |   Hidden size in the edge generator (default: 100)    |
| num_vn            |   Number of virtual nodes (default: 20)               |
| lambda_1          |   The value of $\lambda_1$ (default: 0.1)              |
| lambda_2          |   The value of $\lambda_2$ (default: 1.0)              |
| t                 |   Temperature (default: 0.1)                          |
| gpu_id            |   GPU device ID (default: 0)                          |

## Usage

Here is an example to run FedVN on Motif/Basis

##### 1. Prepare environment
Install the environment for [GOOD](https://github.com/divelab/GOOD) [1].
```
git clone https://github.com/divelab/GOOD.git && cd GOOD
pip install -e .
```

##### 2. Prepare dataset 
```
cd .. && unzip data/GOODMotif.zip -d data 
```

##### 3. Run code

```
python fedvn.py --dataset_name=motif --domain=basis --data_path=./data/ --num_vn=20 --lambda1=0.1 lambda2=1 
```

## Other dataset 

|   Dataset     |                   Link        | 
|---------------|-------------------------------|
|   CMNIST      | https://drive.google.com/file/d/1F2r2kVmA0X07AXyap9Y_rOM6LipDzwhq/view?usp=sharing |
|   ZINC        | https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view?usp=sharing |
|   SST2        | https://drive.google.com/file/d/1lGNMbQebKIbS-NnbPxmY4_uDGI7EWXBP/view?usp=sharing |

## Reference
[1] Gui, Shurui, Xiner Li, Limei Wang, and Shuiwang Ji. "Good: A graph out-of-distribution benchmark." Advances in Neural Information Processing Systems 35 (2022): 2059-2073.
