from random import random

import networkx as nx
import scipy.sparse as sp
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pylab as pl
from utils import load_mat, adj_to_dgl_graph, preprocess_features, normalize_adj
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, average_precision_score
import heapq
from sklearn.preprocessing import normalize



def cal_similarity_mean(file_path):
    data = np.loadtxt(file_path)
    label = data[:, 0]
    subgraph_positive_simlarity = data[:, 1]
    total_normal = subgraph_positive_simlarity[label == 0]
    total_abnormal = subgraph_positive_simlarity[label == 1]

    normal_mean = np.mean(total_normal)
    abnormal_mean = np.mean(total_abnormal)

    return normal_mean, abnormal_mean


def get_F1_score(dataname):
    adj, features, labels, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(dataname)
    path = './score/score_' + dataname + '.mat'
    data = sio.loadmat(path)
    k = ano_label.sum()
    print(k)
    score = data['score'].squeeze()
    print(score)
    y_pred = np.zeros(score.shape[0])
    index = heapq.nlargest(150, range(len(score)), score.take)
    y_pred[index] = 1
    print('presicion:', precision_score(ano_label, y_pred))
    print('F1_score:', f1_score(ano_label, y_pred))
    print('Recall:', recall_score(ano_label, y_pred))
    auc = roc_auc_score(ano_label, score)
    print('auc:', auc)
    print('acc:', accuracy_score(ano_label, y_pred))
    print('AP:', average_precision_score(ano_label, y_pred))


for i in range(10):
    file_path = './similarity/cora' + '_similarity_' + str((i+1)*10) + '.txt'
    normal_mean, anomaly_mean = cal_similarity_mean(file_path)
    print('Normal:{:.4f}   Abnormal:{:.4f}   Diff:{:.4f}'.format(normal_mean, anomaly_mean, (anomaly_mean - normal_mean)))
