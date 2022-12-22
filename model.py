import torch
import torch.nn as nn
import torch.nn.functional as F
from topology_dist import *
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, du, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class Decoder(nn.Module):
    def __init__(self, n_in, n_h, hidden_size = 128):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.network1 = nn.Sequential(
            nn.Linear(n_h , self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network2 = nn.Sequential(
            nn.Linear(n_h * 2, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network3 = nn.Sequential(
            nn.Linear(n_h * 3, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network4 = nn.Sequential(
            nn.Linear(n_h * 4, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network5 = nn.Sequential(
            nn.Linear(n_h * 5, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network6 = nn.Sequential(
            nn.Linear(n_h * 6, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network7 = nn.Sequential(
            nn.Linear(n_h * 7, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )

    def forward(self, h_raw, subgraph_size):
        sub_size = h_raw.shape[1]
        batch_size = h_raw.shape[0]
        sub_node = h_raw[:, :sub_size - 2, :]
        input_res = sub_node.reshape(batch_size, -1)
        if subgraph_size == 1:
            node_recons = self.network1(input_res)
        elif subgraph_size == 2:
            node_recons = self.network2(input_res)
        elif subgraph_size == 3:
            node_recons = self.network3(input_res)
        elif subgraph_size == 4:
            node_recons = self.network4(input_res)
        elif subgraph_size == 5:
            node_recons = self.network5(input_res)
        elif subgraph_size == 6:
            node_recons = self.network6(input_res)
        elif subgraph_size == 7:
            node_recons = self.network7(input_res)
        return node_recons


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def similarity( reps1, reps2 ):
    reps1_unit = F.normalize(reps1, dim=-1)
    reps2_unit = F.normalize(reps2, dim=-1)
    if len(reps1.shape) == 2:
        sim_mat = torch.einsum("ik,jk->ij", [reps1_unit, reps2_unit])
    elif len(reps1.shape) == 3:
        sim_mat = torch.einsum('bik,bjk->bij', [reps1_unit, reps2_unit])
    else:
        print(f"{len(reps1.shape)} dimension tensor is not supported for this function!")
    return sim_mat


def Sinkhorn( out1, avg_out1, out2, avg_out2,
             Sinkhorn_iter_times=5, lamb=20, rescale_ratio=None):

    cost_matrix = 1 - similarity(out1, out2)
    if rescale_ratio is not None:
        cost_matrix = cost_matrix * rescale_ratio

    # Sinkhorn iteration
    with torch.no_grad():
        r = torch.bmm(out1, avg_out2.transpose(1, 2))
        r[r <= 0] = 1e-8
        r = r / r.sum(dim=1, keepdim=True)
        c = torch.bmm(out2, avg_out1.transpose(1, 2))
        c[c <= 0] = 1e-8
        c = c / c.sum(dim=1, keepdim=True)
        P = torch.exp(-1 * lamb * cost_matrix)
        u = (torch.ones_like(c) / c.shape[1])
        for i in range(Sinkhorn_iter_times):
            v = torch.div(r, torch.bmm(P, u))
            u = torch.div(c, torch.bmm(P.transpose(1, 2), v))
        u = u.squeeze(dim=-1)
        v = v.squeeze(dim=-1)
        transport_matrix = torch.bmm(torch.bmm(matrix_diag(v), P), matrix_diag(u))
    assert cost_matrix.shape == transport_matrix.shape

    S = torch.mul(transport_matrix, 1 - cost_matrix).sum(dim=1).sum(dim=1, keepdim=True)

    return S







class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, have_neg = False, hidden_size = 128,
                 temperature=0.4, Sinkhorn_iter_times = 5, lamb=20, is_rectified=True, topo_t=2, neg_top_k=50):
        super(Model, self).__init__()
        self.read_mode = readout
        self.hidden_size = hidden_size
        self.gcn = GCN(n_in, n_h, activation)
        self.decoder = Decoder(n_in, n_h, self.hidden_size)
        self.temperature = temperature
        self.Sinkhorn_iter_times = Sinkhorn_iter_times
        self.lamb = lamb
        self.rectified = is_rectified
        self.topo_t = topo_t
        self.have_neg = have_neg
        self.neg_top_k = neg_top_k

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.discriminator = Discriminator(n_h, negsamp_round)


    # --------------------Inter-view / Cross-view / RoSA loss---------------#
    def InterViewLoss(self, h1, h2, rescale_ratio, have_neg = False, neg_top_k = 50):
        h1_new = h1.clone()
        h2_new = h2.clone()
        h1_new[:, [-2, -1], :] = h1_new[:, [-1, -2], :]
        h2_new[:, [-2, -1], :] = h2_new[:, [-1, -2], :]

        h_graph_1 = h1_new[:, : -1, :]  # (B, subgraph_size, D)
        h_node_1 = h1_new[:, -1, :][:, None, :]  # (B,1,D)
        h_graph_2 = h2_new[:, : -1, :]  # (B, subgraph_size, D)
        h_node_2 = h2_new[:, -1, :][:, None, :]  # (B,1,D)

        # h_graph_1 = h1[:, : -1, :]  # (B, subgraph_size, D)
        # h_node_1 = h1[:, -1, :][:, None, :]  # (B,1,D)
        # h_graph_2 = h2[:, : -1, :]  # (B, subgraph_size, D)
        # h_node_2 = h2[:, -1, :][:, None, :]  # (B,1,D)

        # positive
        fx = lambda x: torch.exp(x / self.temperature)
        if rescale_ratio is not None:
            # rescale_ratio about 0.5+-
            sim_pos = Sinkhorn(h_graph_1, h_node_1, h_graph_2, h_node_2, self.Sinkhorn_iter_times, self.lamb, rescale_ratio)
            loss_pos = fx( sim_pos ) * 2
        else:
            sim_pos = Sinkhorn(h_graph_1, h_node_1, h_graph_2, h_node_2, self.Sinkhorn_iter_times, self.lamb)
            loss_pos = fx( sim_pos )

        # negative
        if have_neg:
            neg_sim_list = []
            loss_neg_total = 0
            sim_neg = 0
            batch_size = h_node_1.shape[0]
            neg_index = list(range(batch_size))
            for i in range((batch_size - 1)):
                neg_index.insert(0, neg_index.pop(-1))
                out1_perm = h_graph_1[neg_index].clone()
                out2_perm = h_graph_2[neg_index].clone()
                avg_out1_perm = h_node_1[neg_index].clone()
                avg_out2_perm = h_node_2[neg_index].clone()
                sim_neg1 = Sinkhorn(h_graph_1, h_node_1, out1_perm, avg_out1_perm, self.Sinkhorn_iter_times, self.lamb)
                sim_neg2 = Sinkhorn(h_graph_1, h_node_1, out2_perm, avg_out2_perm, self.Sinkhorn_iter_times, self.lamb)
                sim_neg += (sim_neg1 + sim_neg2) / 2
                loss_neg_total += fx(sim_neg1) + fx(sim_neg2)
                # top k neg
                neg_sim_list.append(torch.squeeze(sim_neg1).detach().cpu().numpy())
                neg_sim_list.append(torch.squeeze(sim_neg2).detach().cpu().numpy())

            # top max k as negative pairs
            neg_sim = torch.tensor(np.array(neg_sim_list), requires_grad=True)
            neg_sim = neg_sim.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            neg_sim = torch.sort(neg_sim, descending=False, dim=0)[0]
            loss_neg_top_k = neg_sim[:neg_top_k, :]
            loss_neg_total = torch.mean(loss_neg_top_k, dim=0)
            loss_neg_total = torch.unsqueeze(loss_neg_total, 1)


            loss = -torch.log((loss_pos) / (loss_neg_total + loss_pos)) #!!!
            sim_neg = sim_neg / (batch_size - 1)
            sim_all = sim_pos - sim_neg
        else:
            loss = -torch.log( loss_pos )
            sim_all = sim_pos

        return loss, sim_all, sim_pos



    def forward(self, feature1, adj1, raw1, size1, feature2, adj2, raw2, size2,
                full_topology_dist, batch_g_idx, batch_g_idx_2, sparse=False):

        h1 = self.gcn(feature1, adj1, sparse)
        h2 = self.gcn(feature2, adj2, sparse)
        h_raw_1 = self.gcn(raw1, adj1, sparse)
        h_raw_2 = self.gcn(raw2, adj2, sparse)

        if self.rectified:
            topology_dist = gen_batch_topology_dist(full_topology_dist, batch_g_idx, batch_g_idx_2)
            rescale_ratio = torch.sigmoid(topology_dist / self.topo_t)
        else:
            rescale_ratio = None

        # --------------------Reconstruction loss---------------#
        # assert h_raw_1.shape == h_raw_2.shape
        node_recons_1 = self.decoder(h_raw_1, size1)
        node_recons_2 = self.decoder(h_raw_2, size2)

        # --------------------Intra-view / CoLa loss---------------#
        if self.read_mode != 'weighted_sum':
            h_node_1 = h1[:, -1, :]
            h_graph_read_1 = self.read(h1[:, : -1, :])
            h_node_2 = h2[:, -1, :]
            h_graph_read_2 = self.read(h2[:, : -1, :])
        else:
            h_node_1 = h1[:, -1, :]
            h_graph_read_1 = self.read(h1[:, : -1, :], h1[:, -2: -1, :])
            h_node_2 = h2[:, -1, :]
            h_graph_read_2 = self.read(h2[:, : -1, :], h2[:, -2: -1, :])

        disc_1 = self.discriminator(h_graph_read_1, h_node_1)
        disc_2 = self.discriminator(h_graph_read_2, h_node_2)

        # --------------------Inter-view / Cross-view / RoSA loss---------------#
        if self.rectified:
            Inter_loss_1, sim_all_1, sim_pos_1 = self.InterViewLoss(h1, h2, rescale_ratio, self.have_neg, self.neg_top_k)
            Inter_loss_2, sim_all_2, sim_pos_2 = self.InterViewLoss(h2, h1, rescale_ratio.transpose(1,2), self.have_neg, self.neg_top_k)
        else:
            Inter_loss_1, sim_all_1, sim_pos_1 = self.InterViewLoss(h1, h2, None, self.have_neg, self.neg_top_k)
            Inter_loss_2, sim_all_2, sim_pos_2 = self.InterViewLoss(h2, h1, None, self.have_neg, self.neg_top_k)


        return node_recons_1, node_recons_2, disc_1, disc_2, Inter_loss_1, Inter_loss_2, sim_all_1, sim_all_2, sim_pos_1, sim_pos_2
