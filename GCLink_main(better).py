import pandas as pd
import numpy as np
import random
import os
import sys
import time
import argparse
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", message=".*dropout_adj.*deprecated.*")


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
import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast


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

    def forward(self, data_feature, adj, train_data):
        # ✅ 评估阶段：不用增强，跑一次原图
        if not self.training:
            embed, tf_embed, target_embed, pred = self.encoder(data_feature, adj, train_data)
            # 为了不改外部解包逻辑，返回两份相同的
            return embed, tf_embed, target_embed, pred, embed, tf_embed, target_embed, pred

        # ✅ 训练阶段：照旧做增强
        index = adj.coalesce().indices()
        size = adj.coalesce().size()

        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)

        adj1 = build_sparse_adj(edge_index1, edge_weight1, size, device)
        adj2 = build_sparse_adj(edge_index2, edge_weight2, size, device)

        embed1, tf_embed1, target_embed1, pred1 = self.encoder(x1, adj1, train_data)
        embed2, tf_embed2, target_embed2, pred2 = self.encoder(x2, adj2, train_data)

        return embed1, tf_embed1, target_embed1, pred1, embed2, tf_embed2, target_embed2, pred2

def build_sparse_adj(edge_index, edge_weight, size, device):
    # edge_index: [2, E]
    if edge_weight is None:
        values = torch.ones((edge_index.shape[1],), device=device)
    else:
        values = edge_weight.to(device)
        if values.dim() != 1:
            values = values.view(-1)
    return torch.sparse_coo_tensor(edge_index, values, size).coalesce()

def gnn_train(data_feature, adj1, adj2, gnn_model, optimizer, loss_fn):
    pretrain_loss = 0.0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        gnn_model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1).float()

        # pred1/pred2 是 logits（你已确认）
        _, _, _, pred1 = gnn_model(data_feature, adj1, train_x)
        _, _, _, pred2 = gnn_model(data_feature, adj2, train_x)

        loss1 = loss_fn(pred1, train_y)
        loss2 = loss_fn(pred2, train_y)
        loss = loss1 + loss2

        loss.backward()  # 不需要 retain_graph=True
        torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
        optimizer.step()

        pretrain_loss += loss.item()

    return float(pretrain_loss)

def pretrain(data_feature, adj, model, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    index = adj.coalesce().indices()
    size = adj.coalesce().size()

    for epoch in range(1, epochs + 1):
        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)

        adj1 = build_sparse_adj(edge_index1, edge_weight1, size, device)
        adj2 = build_sparse_adj(edge_index2, edge_weight2, size, device)

        pre_train_loss = gnn_train(data_feature, adj1, adj2, model, optimizer, loss_fn)

        scheduler.step()
        print(f'Epoch:{epoch} pre-train loss:{pre_train_loss:.5f}')


def train(model, contrast_model, optimizer, loss_fn, epoch, warmup_epochs=10, con_w=0.1):
    """
    返回：
      avg_total, avg_bce, avg_con, lam
    """
    model.train()

    # warmup：前 warmup_epochs 轮线性升到 con_w
    if warmup_epochs <= 0:
        lam = con_w
    else:
        lam = con_w * min(1.0, epoch / float(warmup_epochs))

    total_sum, bce_sum, con_sum = 0.0, 0.0, 0.0
    nb = 0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        optimizer.zero_grad()

        if args.flag:
            # 多分类时你原来怎么处理就保持；这里只写二分类分支为主
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1).float()

        embed1, _, _, pred1, embed2, _, _, pred2 = model(data_feature, adj, train_x)

        con_loss = contrast_model(h1=embed1, h2=embed2)

        # pred1/pred2 是 logits：直接用 BCEWithLogitsLoss
        loss_BCE1 = loss_fn(pred1, train_y)
        loss_BCE2 = loss_fn(pred2, train_y)
        loss_bce = loss_BCE1 + loss_BCE2

        loss_total = loss_bce + lam * con_loss

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_sum += float(loss_total.item())
        bce_sum += float(loss_bce.item())
        con_sum += float(con_loss.item())
        nb += 1

    avg_total = total_sum / max(nb, 1)
    avg_bce = bce_sum / max(nb, 1)
    avg_con = con_sum / max(nb, 1)
    return avg_total, avg_bce, avg_con, lam


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
contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(device)
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
pre_epochs = 20
if pre_epochs > 0:
    pretrain(data_feature, adj, encoder, pre_epochs)

model = GCLink(encoder=encoder)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
loss_fn = torch.nn.BCEWithLogitsLoss()
best_aupr = -1.0
best_path = os.path.join(model_path, f"{args.cell_type}_best_model.pkl")

for epoch in range(1, args.epochs + 1):
    avg_total, avg_bce, avg_con, lam = train(
        model, contrast_model, optimizer, loss_fn,
        epoch=epoch, warmup_epochs=10, con_w=0.1
    )

    model.eval()
    _, _, _, _, _, _, _, score_logits = model(data_feature, adj, validation_data)

    if args.flag:
        score = torch.softmax(score_logits, dim=1)
    else:
        score = torch.sigmoid(score_logits)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1], flag=args.flag)

    print(
        f"Epoch:{epoch} "
        f"loss(total/bce/con)={avg_total:.4f}/{avg_bce:.4f}/{avg_con:.4f} "
        f"lam={lam:.3f} "
        f"AUC:{AUC:.3f} AUPR:{AUPR:.3f}"
    )

    if AUPR > best_aupr:
        best_aupr = AUPR
        torch.save(model.state_dict(), best_path)

    scheduler.step()

# 用主训练的 best 跑 test
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

_, _, _, _, _, _, _, score_logits = model(data_feature, adj, test_data)

if args.flag:
    score = torch.softmax(score_logits, dim=1)
else:
    score = torch.sigmoid(score_logits)

AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1], flag=args.flag)
print('best_val_AUPR:{:.3f}'.format(best_aupr))
print('test_AUC:{:.3F}'.format(AUC), 'test_AUPR:{:.3F}'.format(AUPR))

# Load best model and test

model.eval()

_, _, _, _, _, _, _, score = model(data_feature, adj, test_data)

if args.flag:
    score = torch.softmax(score, dim=1)
else:
    score = torch.sigmoid(score)

AUC, AUPR, AUPR_norm = Evaluation(
    y_pred=score, y_true=test_data[:, -1], flag=args.flag)
print(score)
print('test_AUC:{:.3F}'.format(AUC), 'test_AUPR:{:.3F}'.format(AUPR))

