## SAMCL: Subgraph-Aligned Multi-View Contrastive Learning for Graph Anomaly Detection
___
This is  the source code for "SAMCL: Subgraph-Aligned Multi-View Contrastive Learning for Graph Anomaly Detection"


### Requirements
___
- Python >= 3.7
- Pytorch >= 1.8.1
- Numpy >= 1.21.6
- Scipy >= 1.7.3
- scikit-learn >= 0.24.1
- networkx == 2.5.1
- dgl == 0.4


### Organization

We have already put the datasets with inject anomaly in folder named `dataset`. You can also create a folder named `raw_datasets` in root directory to store other downloaded datasets. The directory structure should be organized as follows: 

```
.
├── ...
├── datasets
 │   ├── topology_dist
 │   │   ├── acm_padding.pt
 │   │   ├── blogcatalog_padding.pt
 │   │   ├── ...
 │   ├── ACM.mat
 │   ├── BlogCatalog.mat
 │   ├── citeseer.mat
├── pkl
├── ...
```

### Generate topology distance 
   run `topolofy_dist.py` first (in Pytorch >= 1.8.0, networkx >= 2.7)
   
### Running
___
Take Cora dataset as an example:

    python main.py --dataset cora --lr 2e-3 --alpha 0.5 --beta 0.4 --K_1 2 --K_2 8
___
To train and evaluate on other datasets:

    python main.py --dataset citeseer --lr 2e-3 --alpha 0.5 --beta 0.8 --K_1 6 --K_2 4
    python main.py --dataset BlogCatalog --lr 1e-2 --alpha 0.7 --beta 0.8 --K_1 4 --K_2 8
    python main.py --dataset ACM --lr 5e-3 --alpha 0.3 --beta 0.1 --K_1 6 --K_2 8
    python main.py --dataset pubmed --lr 2e-3 --alpha 0.7 --beta 0.6 --K_1 4 --K_2 8

