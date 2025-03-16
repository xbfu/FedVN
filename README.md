## Introduction
This is our implementation of our paper *Virtual Nodes Can Help: Tackling Distribution Shifts in Federated Graph Learning.*

**TL;DR**: A novel perspective to tackle distribution shifts in federated graph learning through client-specific graph augmentation with virtual nodes.  

**Framework Overview**:
Federated Graph Learning (FGL) enables multiple clients to jointly train powerful graph learning models, e.g., Graph Neural Networks (GNNs), without sharing their local graph data for graph-related downstream tasks, such as graph property prediction. In the real world, however, the graph data can suffer from significant distribution shifts across clients as the clients may collect their graph data for different purposes. In particular, graph properties are usually associated with invariant label-relevant substructures (i.e., subgraphs) across clients, while label-irrelevant substructures can appear in a client-specific manner. The issue of distribution shifts of graph data hinders the efficiency of GNN training and leads to serious performance degradation in FGL. 
To tackle the aforementioned issue, we propose a novel FGL framework entitled FedVN that eliminates distribution shifts through client-specific graph augmentation strategies with multiple learnable Virtual Nodes (VNs). Specifically, FedVN lets the clients jointly learn a set of shared VNs while training a global GNN model. To eliminate distribution shifts, each client trains a personalized edge generator that determines how the VNs connect local graphs in a client-specific manner.
Furthermore, we provide theoretical analyses indicating that FedVN can eliminate distribution shifts of graph data across clients. Extensive experiments on four datasets under five settings demonstrate the superiority of our proposed FedVN over nine baselines.

![](https://github.com/xbfu/FedVN/blob/main/FedVN.png)

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

##### 1. Prepare Environment
Install the environment for [GOOD](https://github.com/divelab/GOOD) [1].
```
git clone https://github.com/divelab/GOOD.git && cd GOOD
pip install -e .
```

##### 2. Prepare Dataset 
```
cd .. && unzip data/GOODMotif.zip -d data 
```

##### 3. Run Code

```
python fedvn.py --dataset_name=motif --domain=basis --data_path=./data/ --num_vn=20 --lambda1=0.1 --lambda2=1 
```

## Other Dataset 

|   Dataset     |                   Link        | 
|---------------|-------------------------------|
|   CMNIST      | https://drive.google.com/file/d/1F2r2kVmA0X07AXyap9Y_rOM6LipDzwhq/view?usp=sharing |
|   ZINC        | https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view?usp=sharing |
|   SST2        | https://drive.google.com/file/d/1lGNMbQebKIbS-NnbPxmY4_uDGI7EWXBP/view?usp=sharing |

## Hardware Environment

We run our experiments using a server with 4 NVIDIA RTX 4000 GPUs. Each GPU has 8 GB of memory. The server is equipped with 512 GB memory and 2 10-core Intel CPUs.

## Reference
[1] Gui, Shurui, Xiner Li, Limei Wang, and Shuiwang Ji. "Good: A graph out-of-distribution benchmark." Advances in Neural Information Processing Systems 35 (2022): 2059-2073.
