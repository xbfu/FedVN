## Introduction
This is our implementation of our paper *Virtual Nodes Can Help: Tackling Distribution Shifts in Federated Graph Learning.*

**TL;DR**: A novel perspective to tackle distribution shifts in federated graph learning through client-specific graph augmentation with virtual nodes.  

**Framework Overview**:
This figure demonstrates an overview of our proposed FedVN. 
The intuition of FedVN is to let the clients manipulate their local data through learnable graph augmentation strategies 
in order that the global GNN model can be trained over identical manipulated graph data without any distribution shift 
across clients. To achieve this, the key point is to design a proper scheme for graph augmentation. Inspired by recent
studies about virtual nodes in graph learning, we propose to learn graph augmentation with extra virtual nodes to 
eliminate distribution shifts in FGL. More specifically, the clients in FedVN collaboratively learn a set of shared 
virtual nodes while training a global GNN model. Considering the cross-client distribution shift, FedVN enables each 
client to learn a personalized edge predictor that determines how the VNs connect its local graphs.


<img src="https://anonymous.4open.science/r/FedVN-10024/FedVN.png" hight="100px"/>


## Dependencies


matplotlib==3.7.1  
numpy==1.21.5  
scikit_learn==1.0.2  
torch==1.13.1  
torch-cluster==1.6.3  
torch_geometric==2.4.0  
torch-scatter==2.1.2  
torch-sparse==0.6.18


## Parameters

| Parameter         |           Description                     | 
|-------------------|-------------------------------------------|
| dataset_name      |   Dataset to use (default: motif)         |
| domain            |   Data partition setting (default: basis) |
| data_path         |   Data directory (default: ./data/)       |
| num_local_graphs  |   Number of local graphs (default: 1000)  |
| epochs            |   Number of local epochs (default: 1)     |
| batch_size        |   Batch size of training (default: 32)    |

## Usage

Here is an example to run FedVN on Motif/Basis

##### 1. GOOD installation  
Install GOOD following this repo: https://github.com/divelab/GOOD

##### 2. Prepare dataset 
```
unzip data/GOODMotif.zip -d data 
```

##### 3. Run code

```
python fedvn --dataset_name=motif --domain=basis --data_path=./data/ --num_vn=20 --lambda1=0.1 lambda2=1 
```

## Other dataset 

|   Dataset     |                   Link        | 
|---------------|-------------------------------|
|   CMNIST      | https://drive.google.com/file/d/1F2r2kVmA0X07AXyap9Y_rOM6LipDzwhq/view?usp=sharing |
|   ZINC        | https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view?usp=sharing |
|   SST2        | https://drive.google.com/file/d/1lGNMbQebKIbS-NnbPxmY4_uDGI7EWXBP/view?usp=sharing |
