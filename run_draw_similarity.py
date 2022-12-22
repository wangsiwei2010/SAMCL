from model_HU import *
from utils import *
from topology_dist import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import random
import os
import dgl
import torch
import argparse
from tqdm import tqdm
# import yaml
# from yaml import SafeLoader
import networkx as nx

from scipy.stats import norm
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='CNCL-GAD')
parser.add_argument('--dataset', type=str, default='cora')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--alpha_recon', type=float, default=0.1)
parser.add_argument('--alpha_inter', type=float, default=0.5)
parser.add_argument('--alpha_intra', type=float, default=1)
parser.add_argument('--Sinkhorn_iter_times', type=int, default=5)
parser.add_argument('--Sinkhorn_lamb', type=int, default=20)
parser.add_argument('--topo_t', type=int, default=10, help='temperature for sigmoid in topology dist')
parser.add_argument('--temperature', type=float, default=3, help='temperature for fx')
parser.add_argument('--rectified', type=bool, help='use rectified cost matrix', default=True)
parser.add_argument('--have_neg', type=bool, help='anomaly score and LOSS contain negtive pairs OT', default=False)

parser.add_argument('--subgraph_size_1', type=int, help='view 1', default=4)
parser.add_argument('--subgraph_size_2', type=int, help='view 2', default=8)
parser.add_argument('--restart_prob_1', type=float, help='RWR restart probability on view 1', default=0.9)
parser.add_argument('--restart_prob_2', type=float, help='RWR restart probability on view 2', default=0.3)
parser.add_argument('--subgraph_mode', type=str, default='1/2+1/2')


# parser.add_argument('--subgraph_size', type=int, default=4)

args = parser.parse_args()

# config = yaml.load(open('config.yaml'), Loader=SafeLoader)[args.dataset]
# # combine args and config
# for k, v in config.items():
#     args.__setattr__(k, v)

AUC_list = []


alpha_recon = args.alpha_recon
alpha_inter = args.alpha_inter
alpha_intra = args.alpha_intra
batch_size = args.batch_size
subgraph_size_1 = args.subgraph_size_1
subgraph_size_2 = args.subgraph_size_2

print('Dataset: ', args.dataset)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj, features, labels, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
A = adj

full_topology_dist = register_topology(args, adj)

dgl_graph = adj_to_dgl_graph(adj)
raw_feature = features.todense()
features, _ = preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
b_adj = adj

features = torch.FloatTensor(features[np.newaxis])
raw_feature = torch.FloatTensor(raw_feature[np.newaxis])

adj = torch.FloatTensor(adj[np.newaxis])
b_adj = torch.FloatTensor(b_adj[np.newaxis])

# Initialize model and optimiser
model = Model(n_in=ft_size, n_h=args.embedding_dim, activation='prelu', negsamp_round=args.negsamp_ratio,
              readout=args.readout, hidden_size=args.hidden_size, temperature=args.temperature,
              Sinkhorn_iter_times = args.Sinkhorn_iter_times, lamb=args.Sinkhorn_lamb, is_rectified=args.rectified,
              topo_t = args.topo_t, have_neg = args.have_neg)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    full_topology_dist = full_topology_dist.to(device)
    model.to(device)
    features = features.to(device)
    raw_feature = raw_feature.to(device)
    adj = adj.to(device)
    b_adj = b_adj.to(device)
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
mse_loss = nn.MSELoss(reduction='mean')

# HU
if nb_nodes % batch_size == 0:
    batch_num = nb_nodes // batch_size
else:
    batch_num = nb_nodes // batch_size + 1






def draw_similarity_distribution(file_path, epoch):
    data = np.loadtxt(file_path)
    label = data[:, 0]
    subgraph_positive_simlarity = data[:, 1]
    total_normal = subgraph_positive_simlarity[label == 0]
    total_abnormal = subgraph_positive_simlarity[label == 1]

    plt.figure()
    x_lables = ['Normal', 'Abnormal']
    n, bins, patches = plt.hist([total_normal, total_abnormal], bins=50, color=['green', 'red'], stacked=False, label=x_lables, alpha=0.5)
    # 拟合一条最佳正态分布曲线
    y_nor = norm.pdf(bins, np.mean(total_normal), np.std(total_normal))
    y_abnor = norm.pdf(bins, np.mean(total_abnormal), np.std(total_abnormal))
    plt.plot(bins, y_nor, 'g--')
    plt.plot(bins, y_abnor, 'r--')
    plt.legend(loc='upper left')
    plt.title('Similarity_' + str(epoch))
    save_dir = 'similarity_fig/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + 'similarity_' + str(epoch))





# # Train model
with tqdm(total=args.epochs) as pbar:
    pbar.set_description('Training')

    for epoch in range(args.epochs):

        model.train()

        all_idx = list(range(nb_nodes))

        random.shuffle(all_idx)

        loss_list = []

        total_loss = 0.

        subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A)

        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            lbl = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

            ba = []
            bf = []
            bf_2 = []
            br = []
            raw = []
            raw_2 = []
            subgraph_idx = []
            subgraph_idx_2 = []

            Z_l = torch.full((cur_batch_size,), 1.)
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
            added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
            added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_adj_zero_col_2[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                Z_l = Z_l.to(device)
                lbl = lbl.to(device)
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                cur_adj_r = b_adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                cur_feat = features[:, subgraphs_1[i], :]
                cur_feat_2 = features[:, subgraphs_2[i], :]
                raw_f = raw_feature[:, subgraphs_1[i], :]
                raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                ba.append(cur_adj)
                br.append(cur_adj_r)
                bf.append(cur_feat)
                bf_2.append(cur_feat_2)
                raw.append(raw_f)
                raw_2.append(raw_f_2)
                subgraph_idx.append(subgraphs_1[i])
                subgraph_idx_2.append(subgraphs_2[i])

            ba = torch.cat(ba)
            br = torch.cat(br)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)

            br = torch.cat((br, added_adj_zero_row_2), dim=1)
            br = torch.cat((br, added_adj_zero_col_2), dim=2)


            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            bf_2 = torch.cat(bf_2)
            bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)

            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)
            raw_2 = torch.cat(raw_2)
            raw_2 = torch.cat((raw_2[:, :-1, :], added_feat_zero_row, raw_2[:, -1:, :]), dim=1)

            subgraph_idx = torch.Tensor(subgraph_idx)
            subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
            subgraph_idx = subgraph_idx.int()
            subgraph_idx_2 = subgraph_idx_2.int()
            if torch.cuda.is_available():
                subgraph_idx = subgraph_idx.to(device)
                subgraph_idx_2 = subgraph_idx_2.to(device)

            #/---------------------MODEL-----------------------/#
            node_recons_1, node_recons_2, disc_1, disc_2, inter_loss_1, inter_loss_2, ot_1, ot_2 = \
                model(bf, ba, raw, subgraph_size_1 - 1, bf_2, br, raw_2, subgraph_size_2 - 1,
                      full_topology_dist, subgraph_idx, subgraph_idx_2)


            loss_recon = 0.5 * (mse_loss(node_recons_1, raw[:, -1, :]) + mse_loss(node_recons_2, raw_2[:, -1, :]))
            intra_loss_1 = b_xent(disc_1, lbl)
            intra_loss_2 = b_xent(disc_2, lbl)
            loss_intra = torch.mean( (intra_loss_1 + intra_loss_2) / 2)
            loss_inter = torch.mean( (inter_loss_1 + inter_loss_2) / 2)

            loss = alpha_recon * loss_recon + alpha_inter * loss_inter + alpha_intra * loss_intra
            # print("Reconstruction Loss:{:5}, CoLa Loss:{:5}, OT Loss:{:5}".format(loss_recon, loss_intra, loss_inter))

            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            if not is_final_batch:
                total_loss += loss


        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
        loss_list.append(mean_loss)



        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'pkl/best_' + args.dataset + '.pkl')
            # # # Inference phase
            print('Loading {}th epoch'.format(epoch))
            path = 'pkl/best_' + args.dataset + '.pkl'
            model.load_state_dict(torch.load(path))
            simlarity = np.zeros(nb_nodes)
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A)

            for batch_idx in range(batch_num):
                optimiser.zero_grad()
                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                ba = []
                bf = []
                bf_2 = []
                br = []
                raw = []
                raw_2 = []
                subgraph_idx = []
                subgraph_idx_2 = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
                added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
                added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_adj_zero_col_2[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    added_adj_zero_row = added_adj_zero_row.to(device)
                    added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                    added_adj_zero_col = added_adj_zero_col.to(device)
                    added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                    added_feat_zero_row = added_feat_zero_row.to(device)

                for i in idx:
                    cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                    cur_adj2 = b_adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                    cur_feat = features[:, subgraphs_1[i], :]
                    raw_f = raw_feature[:, subgraphs_1[i], :]
                    cur_feat_2 = features[:, subgraphs_2[i], :]
                    raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                    ba.append(cur_adj)
                    br.append(cur_adj2)
                    bf.append(cur_feat)
                    bf_2.append(cur_feat_2)
                    raw.append(raw_f)
                    raw_2.append(raw_f_2)
                    subgraph_idx.append(subgraphs_1[i])
                    subgraph_idx_2.append(subgraphs_2[i])

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                br = torch.cat(br)
                br = torch.cat((br, added_adj_zero_row_2), dim=1)
                br = torch.cat((br, added_adj_zero_col_2), dim=2)

                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
                bf_2 = torch.cat(bf_2)
                bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)
                raw = torch.cat(raw)
                raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)
                raw_2 = torch.cat(raw_2)
                raw_2 = torch.cat((raw_2[:, :-1, :], added_feat_zero_row, raw_2[:, -1:, :]), dim=1)

                subgraph_idx = torch.Tensor(subgraph_idx)
                subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
                subgraph_idx = subgraph_idx.int()
                subgraph_idx_2 = subgraph_idx_2.int()
                if torch.cuda.is_available():
                    subgraph_idx = subgraph_idx.to(device)
                    subgraph_idx_2 = subgraph_idx_2.to(device)

                # /---------------------MODEL-----------------------/#

                with torch.no_grad():
                    node_res_1, node_res_2, logits_1, logits_2, inter_loss_1, inter_loss_2, ot_1, ot_2 = \
                        model(bf, ba, raw, subgraph_size_1 - 1, bf_2, br, raw_2, subgraph_size_2 - 1,
                              full_topology_dist, subgraph_idx, subgraph_idx_2)

                    sim = 1 - (ot_1 + ot_2) / 2 # positive pair ot
                    simlarity[idx] = sim.squeeze(dim=1).detach().cpu().numpy()

            save_dir = 'similarity/'
            os.makedirs(save_dir, exist_ok=True)
            similarity_path = save_dir + args.dataset + '_similarity_' + str(epoch + 1) + '.txt'
            f = open(similarity_path, 'w')
            for i in range(nb_nodes):
                f.write(str(ano_label[i]) + '   ' + str(simlarity[i]) + '\n')
            f.close()
            draw_similarity_distribution(similarity_path, (epoch + 1))
        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)

    # loss_path = args.dataset + '_loss.txt'
    # f = open(loss_path, 'w')
    # for line in loss_list:
    #     f.write(str(line) + '\n')
    # f.close()

