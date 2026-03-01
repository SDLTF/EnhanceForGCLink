import pandas as pd
import numpy as np
import random
import os
import sys
import time
import argparse
from sklearn.model_selection import train_test_split
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from scGNN import GENELink
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation
# import GCL.losses as L
import GCL.augmentors as A
# from GCL.models import DualBranchContrast


parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('-epochs', type=int, default=20, help='Number of epoch.')
parser.add_argument('-num_head', type=list, default=[3, 3], help='Number of head attentions.')
parser.add_argument('-alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('-hidden_dim', type=int, default=[128, 64, 32], help='The dimension of hidden layer')
parser.add_argument('-output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('-batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('-loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('-seed', type=int, default=8, help='Random seed')
parser.add_argument('-Type', type=str, default='dot', help='score metric')
parser.add_argument('-flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('-reduction', type=str, default='concate', help='how to integrate multihead attention')
parser.add_argument('-sample', type=str, default='sample1', help='sample')
parser.add_argument('-cell_type', type=str, default='hESC', help='cell_type')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GCLink(nn.Module):
    def __init__(self, encoder):
        super(GCLink, self).__init__()
        self.encoder = encoder

    def forward(self, data_feature, adj, train_data, augment=True):
        # 先把 adj 规范化一次（coalesce）
        adj = adj.coalesce()
        index = adj.indices()
        size = adj.size()

        if augment:
            x1, edge_index1, edge_weight1 = aug1(data_feature, index)
            x2, edge_index2, edge_weight2 = aug2(data_feature, index)

            v1 = torch.ones((edge_index1.shape[1]), device=device)
            v2 = torch.ones((edge_index2.shape[1]), device=device)

            adj1 = torch.sparse_coo_tensor(edge_index1, v1, size).to(device)
            adj2 = torch.sparse_coo_tensor(edge_index2, v2, size).to(device)
        else:
            # eval/test：不做随机删边，直接复用原图
            adj1 = adj.to(device)
            adj2 = adj.to(device)

        embed1, tf_embed1, target_embed1, pred1 = self.encoder(data_feature, adj1, train_data)
        embed2, tf_embed2, target_embed2, pred2 = self.encoder(data_feature, adj2, train_data)

        return embed1, tf_embed1, target_embed1, pred1, embed2, tf_embed2, target_embed2, pred2, adj1, adj2


def gnn_train(data_feature, adj1, adj2, gnn_model, optimizer, scheduler):

    pretrain_loss = 0.0
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        gnn_model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        z1, train_tf1, train_target1, pred1 = gnn_model(data_feature, adj1, train_x)
        z2, train_tf2, train_target2, pred2 = gnn_model(data_feature, adj2, train_x)

        if args.flag:
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)

        loss1 = F.binary_cross_entropy(pred1, train_y)
        loss2 = F.binary_cross_entropy(pred2, train_y)

        loss = loss1 + loss2
        # loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        pretrain_loss += loss.item()

    return float(pretrain_loss), pred1, pred2


def pretrain(data_feature, adj, model, epochs=20):

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    index = adj.coalesce().indices()  
    v = adj.coalesce().values()
    size = adj.coalesce().size()

    for epoch in range(1, epochs + 1):

        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)

        v1 = torch.ones((edge_index1.shape[1])).to(device)
        v2 = torch.ones((edge_index2.shape[1])).to(device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v1, size)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size)

        pre_train_loss, pred1, pred2 = gnn_train(data_feature, adj1, adj2, model, optimizer, scheduler)
        print('Epoch:{}'.format(epoch), 'pre-train loss:{:.5F}'.format(pre_train_loss))

def _l2norm(z, eps=1e-12):
    return z / (z.norm(dim=1, keepdim=True) + eps)

@torch.no_grad()
def _ema_update(teacher, student, m: float):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

@torch.no_grad()
def _knn_hyper_refine(z, k: int):
    """
    方案A：用 teacher embedding 动态构造 KNN 超边。
    简化实现：每个节点 i 的“超边”= {i + topK 近邻}，然后用均值聚合得到 teacher 的 hyper-embedding。
    """
    z = _l2norm(z)
    sim = z @ z.t()                 # [N, N] cosine similarity
    idx = sim.topk(k + 1, dim=1).indices[:, 1:]   # 去掉自己，取 topK 近邻 [N, K]
    neigh = z[idx]                  # [N, K, D]
    hyper = torch.cat([z.unsqueeze(1), neigh], dim=1).mean(dim=1)  # [N, D]
    return hyper
@torch.no_grad()
def build_pos_mask(num_nodes: int, pos_edges: torch.Tensor, device):
    """
    pos_edges: [E,2] long tensor on device
    return: dense boolean adjacency mask [N,N] where True means positive edge exists
    适用于 N~1000 级别（你这里 TF+1000），1e6 bool 量级很安全。
    """
    mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    u = pos_edges[:, 0].long()
    v = pos_edges[:, 1].long()
    mask[u, v] = True
    return mask

@torch.no_grad()
def sample_negative_edges(num_nodes: int, pos_mask: torch.Tensor, num_neg: int, device):
    """
    在 [0,N)×[0,N) 里随机采样负边，过滤掉 pos_mask=True 和 u==v。
    """
    # 过采样再过滤，避免 while 循环太久
    oversample = max(num_neg * 3, 1024)
    us = torch.randint(0, num_nodes, (oversample,), device=device)
    vs = torch.randint(0, num_nodes, (oversample,), device=device)
    keep = (~pos_mask[us, vs]) & (us != vs)
    us = us[keep]
    vs = vs[keep]
    if us.numel() < num_neg:
        # 极端情况下递归补一点
        extra_u, extra_v = sample_negative_edges(num_nodes, pos_mask, num_neg - us.numel(), device)
        us = torch.cat([us, extra_u], dim=0)
        vs = torch.cat([vs, extra_v], dim=0)
    return us[:num_neg], vs[:num_neg]

def edge_score_distill_loss(s_logit: torch.Tensor, t_logit: torch.Tensor):
    """
    蒸馏边打分：用 logit 做 MSE（比 sigmoid 后更稳定）
    """
    return F.mse_loss(s_logit, t_logit.detach())

def _hola_loss(student_z, teacher_hz, beta: float):
    zs = _l2norm(student_z)
    zt = _l2norm(teacher_hz).detach()

    # (1) consistency（表征一致性）
    L_cons = (zs - zt).pow(2).sum(dim=1).mean()

    # (2) relational distillation（关系蒸馏：相似度矩阵对齐）
    Ss = zs @ zs.t()
    St = zt @ zt.t()
    L_rel = (Ss - St).pow(2).mean()

    return L_cons + beta * L_rel

def pretrain_hola(data_feature, adj, student_encoder, teacher_encoder,
                  epochs=20, lr=3e-3, ema_m=0.9, k_hyper=10, lam_hola=0.5, beta_rel=0.0):

    optimizer = torch.optim.Adam(student_encoder.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    index = adj.coalesce().indices()
    size = adj.coalesce().size()

    num_nodes = data_feature.shape[0]
    pos_mask = build_pos_mask(num_nodes, train_data, device)
    for epoch in range(1, epochs + 1):
        student_encoder.train()
        optimizer.zero_grad()

        # 两视图增强图
        _, edge_index1, _ = aug1(data_feature, index)
        _, edge_index2, _ = aug2(data_feature, index)

        v1 = torch.ones((edge_index1.shape[1]), device=device)
        v2 = torch.ones((edge_index2.shape[1]), device=device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v1, size).to(device)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size).to(device)

        # student forward（用全量 train_data 做“输入样本”，你脚本里 train_data 是全训练边对）
        z1, _, _, _ = student_encoder(data_feature, adj1, train_data)
        z2, _, _, _ = student_encoder(data_feature, adj2, train_data)

        with torch.no_grad():
            t1, _, _, _ = teacher_encoder(data_feature, adj1, train_data)
            t2, _, _, _ = teacher_encoder(data_feature, adj2, train_data)
            th1 = _knn_hyper_refine(t1, k_hyper)
            th2 = _knn_hyper_refine(t2, k_hyper)

        hola = _hola_loss(z1, th2, beta_rel) + _hola_loss(z2, th1, beta_rel)
        loss = lam_hola * hola

        loss.backward()
        optimizer.step()
        scheduler.step()
        _ema_update(teacher_encoder, student_encoder, ema_m)

        print(f"Epoch:{epoch} pre-train(HOLA) loss:{loss.item():.5f}")

        

def train(model, teacher_encoder, optimizer, scheduler,
          ema_m: float, k_hyper: int, lam_hola: float, beta_rel: float):
    """
    Supervised training (BCE) + HOLA(A) regularization:
      loss = BCE(view1) + BCE(view2) + lam_hola * [hola(s1<-t2) + hola(s2<-t1)]
    Notes:
      - model is GCLink(encoder)
      - teacher_encoder is EMA copy of encoder (GENELink)
      - GCLink.forward must return adj1, adj2 at the end
    """
    running_loss = 0.0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        # labels shape
        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        # Forward: student on two augmented views (inside GCLink.forward)
        embed1, _, _, pred1, embed2, _, _, pred2, adj1, adj2 = model(data_feature, adj, train_x)

        # BCE (supervised)
        if args.flag:
            pred1_prob = torch.softmax(pred1, dim=1)
            pred2_prob = torch.softmax(pred2, dim=1)
        else:
            pred1_prob = torch.sigmoid(pred1)
            pred2_prob = torch.sigmoid(pred2)

        loss_BCE1 = F.binary_cross_entropy(pred1_prob, train_y)
        loss_BCE2 = F.binary_cross_entropy(pred2_prob, train_y)

        # Teacher (EMA) forward on SAME augmented graphs
        with torch.no_grad():
            t_embed1, _, _, _ = teacher_encoder(data_feature, adj1, train_x)
            t_embed2, _, _, _ = teacher_encoder(data_feature, adj2, train_x)

            # Scheme A: dynamic KNN hypergraph refinement (teacher side)
            t_hyper1 = _knn_hyper_refine(t_embed1, k_hyper)
            t_hyper2 = _knn_hyper_refine(t_embed2, k_hyper)

        # HOLA(A): two-way consistency + relational distillation
        hola = _hola_loss(embed1, t_hyper2, beta_rel) + _hola_loss(embed2, t_hyper1, beta_rel)

        # Total loss
        loss = loss_BCE1 + loss_BCE2 # + lam_hola * hola

        # Optimize student
        loss.backward()
        optimizer.step()
        scheduler.step()

        # EMA update teacher AFTER student step
        _ema_update(teacher_encoder, model.encoder, ema_m)

        running_loss += loss.item()

    return float(running_loss)


# Load Data
exp_file = 'Specific Dataset/' + args.cell_type + '/TFs+1000/BL--ExpressionData.csv'
tf_file = 'Specific Dataset/' + args.cell_type + '/TFs+1000/TF.csv'

train_file = 'Data/Specific/' + args.cell_type + ' 1000/' + args.sample + '/Train_set.csv'
test_file = 'Data/Specific/' + args.cell_type + ' 1000/' + args.sample + '/Test_set.csv'
val_file = 'Data/Specific/' + args.cell_type + ' 1000/' + args.sample + '/Validation_set.csv'

# Normalization
data_input = pd.read_csv(exp_file, index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()    

print('feature shape ', feature.shape)

tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)

feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)

data_feature = feature.to(device)
tf = tf.to(device)

train_data = pd.read_csv(train_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values

train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf, loop=args.loop)

adj = adj2saprse_tensor(adj)

train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)

# Construct Model
# contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(device)
encoder = GENELink(input_dim=feature.size()[1],
                 hidden1_dim=args.hidden_dim[0],
                 hidden2_dim=args.hidden_dim[1],
                 hidden3_dim=args.hidden_dim[2],
                 output_dim=args.output_dim,
                 num_head1=args.num_head[0],
                 num_head2=args.num_head[1],
                 alpha=args.alpha,
                 device=device,
                 type=args.Type,
                 reduction=args.reduction
                 ).to(device)

teacher_encoder = copy.deepcopy(encoder).to(device)
for p in teacher_encoder.parameters():
    p.requires_grad_(False)

ema_m = 0.99   # HOLA 动量系数，可调 0.99~0.996
k_hyper = 10   # KNN 超边大小（每个节点取 K 个近邻）
lam_hola = 0.5 # 先对齐你原来的 0.5*con_loss
beta_rel = 0.0 # 关系蒸馏权重

adj = adj.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)

model_path = 'model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Data Augmentation
aug1 = A.Identity()
aug2 = A.EdgeRemoving(pe=0.2)

# Pretrain encoder
pre_epochs = 10
if pre_epochs > 0:
    pretrain_hola(data_feature, adj,
              student_encoder=encoder,
              teacher_encoder=teacher_encoder,
              epochs=pre_epochs,
              lr=3e-3,
              ema_m=ema_m,
              k_hyper=k_hyper,
              lam_hola=lam_hola,
              beta_rel=beta_rel)

model = GCLink(encoder=encoder)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

# Train model
AUC_Threshold = 0
for epoch in range(args.epochs):
    model.train()
    running_loss = train(model, teacher_encoder, optimizer, scheduler,
                     ema_m=ema_m, k_hyper=k_hyper, lam_hola=lam_hola, beta_rel=beta_rel)

    
    model.eval()
    _, _, _, _, _, _, _, score, _, _ = model(data_feature, adj, validation_data, augment=False)

    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)
    
    AUC, AUPR, AUPR_norm = Evaluation(
        y_pred=score, y_true=validation_data[:, -1], flag=args.flag)
    
    print('Epoch:{}'.format(epoch + 1),
          'train loss:{:.5F}'.format(running_loss),
          'AUC:{:.3F}'.format(AUC),
          'AUPR:{:.3F}'.format(AUPR))
    
    if AUC > AUC_Threshold:
        AUC_Threshold = AUC
        torch.save(model.state_dict(), model_path + '/' + args.cell_type + '_best_model' + '.pkl')

# Load best model and test
model.load_state_dict(torch.load(model_path + '/' + args.cell_type + '_best_model' + '.pkl'))
model.eval()

_, _, _, _, _, _, _, score, _, _ = model(data_feature, adj, test_data, augment=False)


if args.flag:
    score = torch.softmax(score, dim=1)
else:
    score = torch.sigmoid(score)

AUC, AUPR, AUPR_norm = Evaluation(
    y_pred=score, y_true=test_data[:, -1], flag=args.flag)
print(score)
print('test_AUC:{:.3F}'.format(AUC), 'test_AUPR:{:.3F}'.format(AUPR))

