import os
from tqdm import tqdm
from utils import *



def register_topology(dataset, adj):
    MAX_HOP = 100
    topo_file = f"./dataset/topology_dist/{dataset.lower()}_padding.pt"
    exist = os.path.isfile(topo_file)
    if not exist:
        nx_graph = nx.from_scipy_sparse_array(adj)
        node_num = adj.shape[0]
        generator = dict(nx.shortest_path_length(nx_graph))
        topology_dist = torch.zeros((node_num, node_num)) # we shift the node index with 1, in order to store 0-index for padding nodes
        mask = torch.zeros((node_num, node_num)).bool()

        for i in tqdm(range(0, node_num)):
            # print(f"processing {i}-th node")
            for j in range(0, node_num):
                if j in generator[i].keys():
                    topology_dist[i][j] = generator[i][j]
                else:
                    topology_dist[i][j] = MAX_HOP
                    mask[i][j] = True # record nodes that do not have connections
        torch.save(topology_dist, topo_file)
    else:
        topology_dist = torch.load(topo_file)

    return topology_dist



def gen_batch_topology_dist(full_topology_dist, node_idx1, node_idx2):
    batch_size = node_idx1.shape[0]
    batch_subpology_dist = [full_topology_dist.index_select(dim=0, index=node_idx1[i]).
                            index_select(dim=1, index=node_idx2[i]) for i in range(batch_size)]
    batch_subpology_dist = torch.stack(batch_subpology_dist)
    return batch_subpology_dist




if __name__ == '__main__':
    '''
        Generate the topology distance of each node in the graph. This need a new folder named topology_dist under
        dataset path. The topology distance is used in Eq. (10).
        '''

    '''
    unified_dataset_interface
    Args:
        dataset_name: BlogCatalog, Flickr,  ACM,  cora,  citeseer,  pubmed
    Return:
        full_topology_dist
    '''

    # Example of calculating topology of cora dataset.
    dataset_name = 'cora'
    adj, _, _, _, _, _, _, _, _ = load_mat(dataset_name)
    full_topology_dist = register_topology(dataset_name, adj)


    # The topology distance calculation of other datasets is basically the same as above.
    # Change the dataset_name to generate the topology distance of other datasets.

