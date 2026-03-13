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


# macOS 适配：移除 CUDA 相关设置
if sys.platform != 'darwin':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from scGNNv2 import GENELink
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
parser.add_argument(
    '-dataset',
    type=str,
    default='STRING',
    choices=['STRING', 'Non-Specific', 'Specific'],
    help='dataset family: STRING, Non-Specific, or Specific'
)
parser.add_argument(
    '-tf_num',
    type=int,
    default=1000,
    choices=[500, 1000],
    help='TF subset size used in folder names, e.g. 500 or 1000'
)
parser.add_argument('-resume', action='store_true', help='resume training from checkpoint')
parser.add_argument('-resume_ckpt', type=str, default='', help='checkpoint path for resume')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# macOS 适配：当前训练流程依赖 sparse COO 邻接，MPS 对该路径支持不完整，需回退 CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS is available, but sparse COO ops are not fully supported in this pipeline. Falling back to CPU.')
    device = torch.device('cpu')
else:
    device = torch.device('cpu')

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

class SimSiamHead(nn.Module):
    def __init__(self, in_dim, proj_dim=64, pred_dim=32):
        super(SimSiamHead, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),

            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),

            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False)
        )

        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),

            nn.Linear(pred_dim, proj_dim)
        )

    def forward(self, z1, z2):
        p1 = self.predictor(self.projector(z1))
        p2 = self.predictor(self.projector(z2))

        z1_detach = self.projector(z1).detach()
        z2_detach = self.projector(z2).detach()

        return p1, p2, z1_detach, z2_detach

def simsiam_loss(p1, p2, z1, z2):
    loss1 = -F.cosine_similarity(p1, z2, dim=1).mean()
    loss2 = -F.cosine_similarity(p2, z1, dim=1).mean()
    return 0.5 * (loss1 + loss2)

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

def train(model, simsiam_head, optimizer, loss_fn, epoch, warmup_epochs=10, con_w=0.1):
    """
    返回：
      avg_total, avg_bce, avg_con, lam
    """
    model.train()
    simsiam_head.train()

    if warmup_epochs <= 0:
        lam = con_w
    else:
        lam = con_w * min(1.0, epoch / float(warmup_epochs))

    total_sum, bce_sum, con_sum = 0.0, 0.0, 0.0
    nb = 0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1).float()

        embed1, _, _, pred1, embed2, _, _, pred2 = model(data_feature, adj, train_x)

        p1, p2, z1_detach, z2_detach = simsiam_head(embed1, embed2)
        con_loss = simsiam_loss(p1, p2, z1_detach, z2_detach)

        loss_BCE1 = loss_fn(pred1, train_y)
        loss_BCE2 = loss_fn(pred2, train_y)
        loss_bce = loss_BCE1 + loss_BCE2

        loss_total = loss_bce + lam * con_loss

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(simsiam_head.parameters()), 1.0
        )
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
dataset_expr_dir = {
    'STRING': 'STRING Dataset',
    'Non-Specific': 'Non-Specific Dataset',
    'Specific': 'Specific Dataset',
}[args.dataset]

exp_file = os.path.join(dataset_expr_dir, args.cell_type, f'TFs+{args.tf_num}', 'BL--ExpressionData.csv')
tf_file = os.path.join(dataset_expr_dir, args.cell_type, f'TFs+{args.tf_num}', 'TF.csv')

split_dir = os.path.join('Data', args.dataset, f'{args.cell_type} {args.tf_num}', args.sample)
train_file = os.path.join(split_dir, 'Train_set.csv')
test_file = os.path.join(split_dir, 'Test_set.csv')
val_file = os.path.join(split_dir, 'Validation_set.csv')

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

simsiam_head = SimSiamHead(
    in_dim=args.hidden_dim[1],
    proj_dim=args.hidden_dim[1],
    pred_dim=args.hidden_dim[2]
).to(device)


adj = adj.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)

model_path = 'model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

run_name = f"{args.cell_type}_{args.sample}_{args.Type}_seed{args.seed}"
best_path = os.path.join(model_path, f"{run_name}_best_model.pkl")
best_ckpt_path = os.path.join(model_path, f"{run_name}_best_checkpoint.pt")
resume_path = args.resume_ckpt if args.resume_ckpt else best_ckpt_path
can_resume = args.resume and os.path.exists(resume_path)

if args.resume and not can_resume:
    print(f"Resume checkpoint not found at {resume_path}, train from scratch.")

# Data Augmentation
aug1 = A.Identity()
aug2 = A.EdgeRemoving(pe=0.2)

# Pretrain encoder
pre_epochs = 20
if can_resume:
    print(f"Resume checkpoint found at {resume_path}, skip pretrain stage.")
elif pre_epochs > 0:
    pretrain(data_feature, adj, encoder, pre_epochs)

model = GCLink(encoder=encoder)
model = model.to(device)

optimizer = Adam(
    list(model.parameters()) + list(simsiam_head.parameters()),
    lr=args.lr
)

scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
loss_fn = torch.nn.BCEWithLogitsLoss()
best_aupr = -1.0
best_epoch = 0
start_epoch = 1
warmup_epochs = 20
con_w = 0.02

if can_resume:
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'simsiam_head_state_dict' in checkpoint:
        simsiam_head.load_state_dict(checkpoint['simsiam_head_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    best_aupr = float(checkpoint.get('best_val_aupr', -1.0))
    best_epoch = int(checkpoint.get('best_epoch', checkpoint.get('epoch', 0)))
    start_epoch = int(checkpoint.get('epoch', 0)) + 1
    print(f"Resume from {resume_path}: start_epoch={start_epoch}, best_aupr={best_aupr:.3f}")

for epoch in range(start_epoch, args.epochs + 1):
    avg_total, avg_bce, avg_con, lam = train(
        model, simsiam_head, optimizer, loss_fn,
        epoch=epoch, warmup_epochs=warmup_epochs, con_w=con_w
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

    scheduler.step()

    if AUPR > best_aupr:
        best_aupr = float(AUPR)
        best_epoch = epoch
        torch.save(model.state_dict(), best_path)
        torch.save(
            {
                'epoch': epoch,
                'best_epoch': best_epoch,
                'run_name': run_name,
                'args': vars(args),
                'model_state_dict': model.state_dict(),
                'simsiam_head_state_dict': simsiam_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': float(AUC),
                'best_val_aupr': float(AUPR),
                'best_val_aupr_norm': float(AUPR_norm),
            },
            best_ckpt_path
        )

# 用主训练的 best 跑 test
if not os.path.exists(best_ckpt_path):
    raise RuntimeError(f"No best checkpoint found at {best_ckpt_path}.")

best_checkpoint = torch.load(best_ckpt_path, map_location=device)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

_, _, _, _, _, _, _, score_logits = model(data_feature, adj, test_data)

if args.flag:
    score = torch.softmax(score_logits, dim=1)
else:
    score = torch.sigmoid(score_logits)

test_AUC, test_AUPR, test_AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1], flag=args.flag)
print('best_val_AUPR:{:.3f}'.format(best_aupr))
print('best_epoch:{}'.format(best_epoch))
print('best_model_path:{}'.format(best_path))
print('best_checkpoint_path:{}'.format(best_ckpt_path))
print('test_AUC:{:.3F}'.format(test_AUC), 'test_AUPR:{:.3F}'.format(test_AUPR))

results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results_file = os.path.join(results_dir, 'simsiam_run_metrics.csv')
result_record = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'cell_type': args.cell_type,
    'sample': args.sample,
    'Type': args.Type,
    'seed': args.seed,
    'epochs': args.epochs,
    'pretrain_epochs': pre_epochs,
    'best_epoch': best_epoch,
    'best_val_aupr': float(best_aupr),
    'test_auc': float(test_AUC),
    'test_aupr': float(test_AUPR),
    'test_aupr_norm': float(test_AUPR_norm),
    'best_model_path': best_path,
    'best_checkpoint_path': best_ckpt_path,
}
write_header = not os.path.exists(results_file)
pd.DataFrame([result_record]).to_csv(results_file, mode='a', header=write_header, index=False)
print('result_csv:{}'.format(results_file))

# Load best model and test

# model.eval()

# _, _, _, _, _, _, _, score = model(data_feature, adj, test_data)

# if args.flag:
#     score = torch.softmax(score, dim=1)
# else:
#     score = torch.sigmoid(score)

# AUC, AUPR, AUPR_norm = Evaluation(
#     y_pred=score, y_true=test_data[:, -1], flag=args.flag)
# print(score)
# print('test_AUC:{:.3F}'.format(AUC), 'test_AUPR:{:.3F}'.format(AUPR))

