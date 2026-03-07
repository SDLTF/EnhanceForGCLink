import pandas as pd
import torch
from torch.utils.data import Dataset
import random as rd
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class scRNADataset(Dataset):
    def __init__(self, train_set, num_gene, flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag

    def __getitem__(self, idx):
        train_data = self.train_set[:, :2]
        train_label = self.train_set[:, -1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len, 2])
            train_tan[:, 0] = 1 - train_label
            train_tan[:, 1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)

    def Adj_Generate(self, TF_set, direction=False, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)  

        for pos in self.train_set:

            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0

        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()  

        return adj


class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self, data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T

    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.from_numpy(values).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor

def Evaluation(y_true, y_pred, flag=False):
    """
    Robust evaluation: accepts torch.Tensor (CPU/GPU) or numpy.ndarray.
    Always converts inputs to 1D numpy arrays (float64 for scores, int for labels),
    removes NaN/Inf consistently, and handles single-class edge cases.
    """
    # ---- y_pred -> numpy 1D float64 ----
    if flag:
        # multi-class: take last column as positive score
        if isinstance(y_pred, np.ndarray):
            y_p = y_pred[:, -1]
        else:
            y_p = y_pred[:, -1].detach().cpu().numpy()
    else:
        if isinstance(y_pred, np.ndarray):
            y_p = y_pred
        else:
            y_p = y_pred.detach().cpu().numpy()

    y_p = np.asarray(y_p, dtype=np.float64).reshape(-1)

    # ---- y_true -> numpy 1D int ----
    if isinstance(y_true, np.ndarray):
        y_t = y_true
    else:
        y_t = y_true.detach().cpu().numpy()

    y_t = np.asarray(y_t).reshape(-1).astype(int)

    # ---- drop NaN/Inf in y_pred (and corresponding y_true) ----
    mask = np.isfinite(y_p)
    if not np.all(mask):
        y_p = y_p[mask]
        y_t = y_t[mask]

    # ---- sklearn undefined if only one class ----
    if y_t.size == 0 or len(np.unique(y_t)) < 2:
        return float("nan"), float("nan"), float("nan")

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)
    AUPR = average_precision_score(y_true=y_t, y_score=y_p)

    pos_rate = float(np.mean(y_t))
    AUPR_norm = AUPR / pos_rate if pos_rate > 0 else float("nan")

    return AUC, AUPR, AUPR_norm


def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr


def Network_Statistic(data_type, net_scale, net_type):

    if net_type == 'STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165, 'hHEP500': 0.379, 'hHEP1000': 0.377, 'mDC500': 0.085,
               'mDC1000': 0.082, 'mESC500': 0.345, 'mESC1000': 0.347, 'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565, 'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError
