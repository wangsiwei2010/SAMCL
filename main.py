from model import *
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
import networkx as nx


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='SAMCL')
# Set argument
parser = argparse.ArgumentParser(description='CNCL-GAD')
parser.add_argument('--dataset', type=str, default='cora')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
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
parser.add_argument('--alpha', type=float)
parser.add_argument('--beta', type=float)
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--Sinkhorn_iter_times', type=int, default=5)
parser.add_argument('--Sinkhorn_lamb', type=int, default=20)
parser.add_argument('--topo_t', type=int, default=10, help='temperature for sigmoid in topology dist')
parser.add_argument('--temperature', type=float, default=3, help='temperature for fx')
parser.add_argument('--rectified', type=bool, help='use rectified cost matrix', default=True)
parser.add_argument('--have_neg', type=bool, help='anomaly score and LOSS contain negtive pairs OT', default=True)
parser.add_argument('--neg_top_k', type=float, help='top max k of OT to select negtive pairs', default=20)

parser.add_argument('--K_1', type=int, help='view 1')
parser.add_argument('--K_2', type=int, help='view 2')
parser.add_argument('--restart_prob_1', type=float, help='RWR restart probability on view 1', default=0.9)
parser.add_argument('--restart_prob_2', type=float, help='RWR restart probability on view 2', default=0.3)
parser.add_argument('--subgraph_mode', type=str, default='1+2')

args = parser.parse_args()


if args.lr is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        args.lr = 2e-3
    elif args.dataset == 'BlogCatalog':
        args.lr = 1e-2
    elif args.dataset == 'ACM':
        args.lr = 5e-3

if args.beta is None:
    if args.dataset == 'cora':
        args.beta = 0.4
    elif args.dataset in ['BlogCatalog', 'citeseer']:
        args.beta = 0.8
    elif args.dataset == 'ACM':
        args.beta = 0.1
    elif args.dataset == 'pubmed':
        args.beta = 0.6

if args.alpha is None:
    if args.dataset in ['cora', 'citeseer']:
        args.alpha = 0.5
    elif args.dataset in ['BlogCatalog', 'pubmed']:
        args.alpha = 0.7
    elif args.dataset == 'ACM':
        args.alpha = 0.3

if args.K_1 is None:
    if args.dataset == 'cora':
        args.K_1 = 2
    elif args.dataset in ['BlogCatalog', 'pubmed']:
        args.K_1 = 4
    elif args.dataset in ['citeseer', 'ACM']:
        args.K_1 = 6

if args.K_2 is None:
    if args.dataset == 'citeseer':
        args.K_2 = 4
    elif args.dataset in ['cora', 'ACM', 'BlogCatalog', 'pubmed']:
        args.K_2 = 8



# config = yaml.load(open('config.yaml'), Loader=SafeLoader)[args.dataset]
# # combine args and config
# for k, v in config.items():
#     args.__setattr__(k, v)

AUC_list = []

alpha_recon = args.beta
alpha_inter = args.alpha
alpha_intra = args.gama
batch_size = args.batch_size
subgraph_size_1 = args.K_1
subgraph_size_2 = args.K_2


print('Dataset: ', args.dataset)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


seed = args.seed
dgl.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj, features, labels, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
A = adj

full_topology_dist = register_topology(args.dataset, adj)

dgl_graph = adj_to_dgl_graph(adj)
raw_feature = features.todense()
features, _ = preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()


features = torch.FloatTensor(features[np.newaxis])
raw_feature = torch.FloatTensor(raw_feature[np.newaxis])

adj = torch.FloatTensor(adj[np.newaxis])


# Initialize model and optimiser
model = Model(n_in=ft_size, n_h=args.embedding_dim, activation='prelu', negsamp_round=args.negsamp_ratio,
            readout=args.readout, hidden_size=args.hidden_size, temperature=args.temperature,
            Sinkhorn_iter_times = args.Sinkhorn_iter_times, lamb=args.Sinkhorn_lamb, is_rectified=args.rectified,
            topo_t = args.topo_t, have_neg = args.have_neg, neg_top_k = args.neg_top_k)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    full_topology_dist = full_topology_dist.to(device)
    model.to(device)
    features = features.to(device)
    raw_feature = raw_feature.to(device)
    adj = adj.to(device)
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
mse_loss = nn.MSELoss(reduction='mean')


if nb_nodes % batch_size == 0:
    batch_num = nb_nodes // batch_size
else:
    batch_num = nb_nodes // batch_size + 1



# # Train model
with tqdm(total=args.epochs) as pbar:
    pbar.set_description('Training')

    for epoch in range(args.epochs):

        model.train()

        all_idx = list(range(nb_nodes))

        random.shuffle(all_idx)


        loss_1 = 0.
        loss_2 = 0.
        loss_3 = 0.
        loss_record = 0.

        total_loss = 0.


        subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2)


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
            ba_2 = []
            bf = []
            bf_2 = []
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
                cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                cur_feat = features[:, subgraphs_1[i], :]
                cur_feat_2 = features[:, subgraphs_2[i], :]
                raw_f = raw_feature[:, subgraphs_1[i], :]
                raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                ba.append(cur_adj)
                ba_2.append(cur_adj_2)
                bf.append(cur_feat)
                bf_2.append(cur_feat_2)
                raw.append(raw_f)
                raw_2.append(raw_f_2)
                subgraph_idx.append(subgraphs_1[i])
                subgraph_idx_2.append(subgraphs_2[i])

            ba = torch.cat(ba)
            ba_2 = torch.cat(ba_2)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            ba_2 = torch.cat((ba_2, added_adj_zero_row_2), dim=1)
            ba_2 = torch.cat((ba_2, added_adj_zero_col_2), dim=2)


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
            node_recons_1, node_recons_2, disc_1, disc_2, inter_loss_1, inter_loss_2, _, _, _, _ = \
                model(bf, ba, raw, subgraph_size_1 - 1, bf_2, ba_2, raw_2, subgraph_size_2 - 1,
                        full_topology_dist, subgraph_idx, subgraph_idx_2)


            loss_recon = 0.5 * (mse_loss(node_recons_1, raw[:, -1, :]) + mse_loss(node_recons_2, raw_2[:, -1, :]))
            intra_loss_1 = b_xent(disc_1, lbl)
            intra_loss_2 = b_xent(disc_2, lbl)
            loss_intra = torch.mean((intra_loss_1 + intra_loss_2) / 2)
            loss_inter = torch.mean((inter_loss_1 + inter_loss_2) / 2)

            loss = alpha_recon * loss_recon + alpha_inter * loss_inter + alpha_intra * loss_intra

            loss.backward()
            optimiser.step()



            loss = loss.detach().cpu().numpy()
            if not is_final_batch:
                total_loss += loss


        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes


        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'pkl/best_' + args.dataset + '.pkl')
        else:
            cnt_wait += 1


        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)




# # Inference phase
# print('Loading {}th epoch'.format(best_t))
path = 'pkl/best_' + args.dataset + '.pkl'
model.load_state_dict(torch.load(path))
multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))


with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_test_rounds):

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A,subgraph_size_1, subgraph_size_2)

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
            ba_2 = []
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
                cur_adj2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                cur_feat = features[:, subgraphs_1[i], :]
                raw_f = raw_feature[:, subgraphs_1[i], :]
                cur_feat_2 = features[:, subgraphs_2[i], :]
                raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                ba.append(cur_adj)
                ba_2.append(cur_adj2)
                bf.append(cur_feat)
                bf_2.append(cur_feat_2)
                raw.append(raw_f)
                raw_2.append(raw_f_2)
                subgraph_idx.append(subgraphs_1[i])
                subgraph_idx_2.append(subgraphs_2[i])

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            ba_2 = torch.cat(ba_2)
            ba_2 = torch.cat((ba_2, added_adj_zero_row_2), dim=1)
            ba_2 = torch.cat((ba_2, added_adj_zero_col_2), dim=2)

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
                node_res_1, node_res_2, logits_1, logits_2, inter_loss_1, inter_loss_2, sim_all_1, sim_all_2, \
                sim_pos_1, sim_pos_2 = model(bf, ba, raw, subgraph_size_1 - 1, bf_2, ba_2, raw_2, subgraph_size_2 - 1,
                                                full_topology_dist, subgraph_idx, subgraph_idx_2)


                logits_1 = torch.squeeze(logits_1)
                logits_1 = torch.sigmoid(logits_1)

                logits_2 = torch.squeeze(logits_2)
                logits_2 = torch.sigmoid(logits_2)

            pdist = nn.PairwiseDistance(p=2)
            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()
            scaler3 = MinMaxScaler()


            score_co1 = - (logits_1[:cur_batch_size] - logits_1[cur_batch_size:]).cpu().numpy()
            score_co2 = - (logits_2[:cur_batch_size] - logits_2[cur_batch_size:]).cpu().numpy()
            score_co = (score_co1 + score_co2) / 2

            score_re = (pdist(node_res_1, raw[:, -1, :]) + pdist(node_res_2, raw_2[:, -1, :])) / 2
            score_re = score_re.cpu().numpy()

            score_ot = - (sim_pos_1 + sim_pos_2) / 2
            score_ot = score_ot.cpu().numpy()

            #nomalize
            ano_score_co = scaler1.fit_transform(score_co.reshape(-1, 1)).reshape(-1)
            ano_score_re = scaler2.fit_transform(score_re.reshape(-1, 1)).reshape(-1)
            ano_score_ot = scaler3.fit_transform(score_ot.reshape(-1, 1)).reshape(-1)

            ano_scores = ano_score_co + alpha_recon * ano_score_re + alpha_inter * ano_score_ot # anomaly score have ot(pos)

            multi_round_ano_score[round, idx] = ano_scores

        pbar_test.update(1)


ano_score_final = np.mean(multi_round_ano_score, axis=0)
auc = roc_auc_score(ano_label, ano_score_final)

print()
print('AUC:{:.4f}'.format(auc))


mat_path = './results'
os.makedirs(mat_path, exist_ok=True)
strAUC = str(int(auc*10000))
sio.savemat(mat_path + '/score_' + args.dataset + '_AUC' + strAUC + '.mat', {'score': ano_score_final})
