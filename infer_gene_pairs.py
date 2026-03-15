import os
import argparse
import warnings
import random
import difflib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message=".*dropout_adj.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from scGNNv2 import GENELink
from utils import scRNADataset, load_data, adj2saprse_tensor


# -------------------------
# 设备
# -------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("MPS 可用，但这里改用 CPU。")
    device = torch.device("cpu")
else:
    device = torch.device("cpu")


# -------------------------
# 和训练时一致的外壳
# -------------------------
class GCLink(nn.Module):
    def __init__(self, encoder):
        super(GCLink, self).__init__()
        self.encoder = encoder

    def forward(self, data_feature, adj, edge_data):
        embed, tf_embed, target_embed, pred = self.encoder(data_feature, adj, edge_data)
        return embed, tf_embed, target_embed, pred


# -------------------------
# 读数据
# -------------------------
def load_everything(dataset, cell_type, tf_num, sample, loop=False):
    dataset_expr_dir = {
        "Non-Specific": "Non-Specific Dataset",
        "Specific": "Specific Dataset",
        "STRING": "STRING Dataset",
    }[dataset]

    exp_file = os.path.join(dataset_expr_dir, cell_type, f"TFs+{tf_num}", "BL--ExpressionData.csv")
    tf_file = os.path.join(dataset_expr_dir, cell_type, f"TFs+{tf_num}", "TF.csv")
    split_dir = os.path.join("Data", dataset, f"{cell_type} {tf_num}", sample)
    train_file = os.path.join(split_dir, "Train_set.csv")

    if not os.path.exists(exp_file):
        raise FileNotFoundError(f"找不到表达矩阵文件: {exp_file}")
    if not os.path.exists(tf_file):
        raise FileNotFoundError(f"找不到 TF 文件: {tf_file}")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"找不到训练集文件: {train_file}")

    print("开始读取数据...")
    print("exp_file  =", exp_file)
    print("tf_file   =", tf_file)
    print("train_file=", train_file)

    data_input = pd.read_csv(exp_file, index_col=0)
    gene_names = data_input.index.to_list()
    print("前20个基因名 =", gene_names[:20])

    loader = load_data(data_input)
    feature = loader.exp_data()
    feature = torch.from_numpy(feature).float()

    tf_idx = pd.read_csv(tf_file, index_col=0)["index"].values.astype(np.int64)
    tf_idx = torch.from_numpy(tf_idx).long()

    train_data_np = pd.read_csv(train_file, index_col=0).values
    train_load = scRNADataset(train_data_np, feature.shape[0], flag=False)
    adj = train_load.Adj_Generate(tf_idx, loop=loop)
    adj = adj2saprse_tensor(adj)

    print("数据读取完成")
    print("gene_num =", feature.shape[0])
    print("feature_dim =", feature.shape[1])

    return {
        "feature": feature.to(device),
        "adj": adj.to(device),
        "gene_names": gene_names,
        "train_data_np": train_data_np,
        "tf_idx": tf_idx.to(device),
    }


# -------------------------
# 建模
# -------------------------
def build_model(feature, hidden_dim, output_dim, num_head, alpha, score_type, reduction):
    encoder = GENELink(
        input_dim=feature.size(1),
        hidden1_dim=hidden_dim[0],
        hidden2_dim=hidden_dim[1],
        hidden3_dim=hidden_dim[2],
        output_dim=output_dim,
        num_head1=num_head[0],
        num_head2=num_head[1],
        alpha=alpha,
        device=device,
        type=score_type,
        reduction=reduction
    ).to(device)

    model = GCLink(encoder=encoder).to(device)
    return model


# -------------------------
# 载入 checkpoint
# -------------------------
def load_checkpoint(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("已加载 checkpoint['model_state_dict']")
    else:
        model.load_state_dict(checkpoint)
        print("已加载纯 state_dict")

    model.eval()
    return model


# -------------------------
# 生成边
# -------------------------
def make_all_pairs(num_nodes, include_self=False):
    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if (not include_self) and (i == j):
                continue
            src_list.append(i)
            dst_list.append(j)

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    label = np.zeros_like(src, dtype=np.int64)

    edge_data = np.stack([src, dst, label], axis=1)
    return edge_data


def make_pairs_for_one_gene(gene_idx, num_nodes, include_self=False):
    src = []
    dst = []

    for j in range(num_nodes):
        if (not include_self) and (j == gene_idx):
            continue
        src.append(gene_idx)
        dst.append(j)

    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    label = np.zeros_like(src, dtype=np.int64)

    edge_data = np.stack([src, dst, label], axis=1)
    return edge_data


# -------------------------
# 分批推断
# -------------------------
@torch.no_grad()
def predict_edges(model, feature, adj, edge_data_np, batch_size=4096):
    scores = []

    total = edge_data_np.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_np = edge_data_np[start:end]
        batch_tensor = torch.from_numpy(batch_np).long().to(device)

        _, _, _, logits = model(feature, adj, batch_tensor)
        prob = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
        scores.append(prob)

        if start % max(batch_size * 20, 1) == 0:
            print(f"推断进度: {start}/{total}")

    scores = np.concatenate(scores, axis=0)
    return scores


# -------------------------
# 保存 Excel（可选）
# -------------------------
def try_save_excel(df, path):
    try:
        df.to_excel(path, index=False)
        print(f"Excel 已保存到: {path}")
    except Exception as e:
        print(f"保存 Excel 失败: {path}")
        print("原因：当前环境可能没有安装 openpyxl")
        print("可运行：pip install openpyxl")
        print("错误信息：", e)


# -------------------------
# 保存结果
# -------------------------
def save_result(edge_data_np, scores, gene_names, out_csv, top_k=50):
    df = pd.DataFrame({
        "src_idx": edge_data_np[:, 0],
        "dst_idx": edge_data_np[:, 1],
        "src_gene": [gene_names[i] for i in edge_data_np[:, 0]],
        "dst_gene": [gene_names[i] for i in edge_data_np[:, 1]],
        "score": scores
    })

    base_name = os.path.splitext(out_csv)[0]

    # 1) 全量结果
    df_all = df.sort_values("score", ascending=False).reset_index(drop=True)
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"全部结果已保存到: {out_csv}")

    # 2) 全网全局 TopK
    df_global_topk = df_all.head(top_k).copy()
    global_topk_csv = f"{base_name}_global_top{top_k}.csv"
    global_topk_xlsx = f"{base_name}_global_top{top_k}.xlsx"

    df_global_topk.to_csv(global_topk_csv, index=False, encoding="utf-8-sig")
    print(f"全网全局 Top {top_k} CSV 已保存到: {global_topk_csv}")
    try_save_excel(df_global_topk, global_topk_xlsx)

    # 3) 每个 src_gene 各自 TopK
    df_each_src_topk = (
        df.sort_values(["src_gene", "score"], ascending=[True, False])
          .groupby("src_gene", group_keys=False)
          .head(top_k)
          .reset_index(drop=True)
    )

    each_src_topk_csv = f"{base_name}_each_src_top{top_k}.csv"
    each_src_topk_xlsx = f"{base_name}_each_src_top{top_k}.xlsx"

    df_each_src_topk.to_csv(each_src_topk_csv, index=False, encoding="utf-8-sig")
    print(f"每个 src_gene 的 Top {top_k} CSV 已保存到: {each_src_topk_csv}")
    try_save_excel(df_each_src_topk, each_src_topk_xlsx)

    # 4) 每个 src_gene 的 Top1
    df_each_src_top1 = (
        df.sort_values(["src_gene", "score"], ascending=[True, False])
          .groupby("src_gene", group_keys=False)
          .head(1)
          .reset_index(drop=True)
    )

    each_src_top1_csv = f"{base_name}_each_src_top1.csv"
    each_src_top1_xlsx = f"{base_name}_each_src_top1.xlsx"

    df_each_src_top1.to_csv(each_src_top1_csv, index=False, encoding="utf-8-sig")
    print(f"每个 src_gene 的 Top 1 CSV 已保存到: {each_src_top1_csv}")
    try_save_excel(df_each_src_top1, each_src_top1_xlsx)

    return df_all, df_global_topk, df_each_src_topk, df_each_src_top1


# -------------------------
# 主函数
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True, help="best_checkpoint.pt 路径")
    parser.add_argument("--dataset", type=str, default="Specific",
                        choices=["Non-Specific", "Specific", "STRING"])
    parser.add_argument("--cell_type", type=str, default="hESC")
    parser.add_argument("--tf_num", type=int, default=1000, choices=[500, 1000])
    parser.add_argument("--sample", type=str, default="sample1")

    parser.add_argument("--hidden_dim", nargs=3, type=int, default=[128, 64, 32])
    parser.add_argument("--output_dim", type=int, default=16)
    parser.add_argument("--num_head", nargs=2, type=int, default=[3, 3])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--Type", type=str, default="dot")
    parser.add_argument("--reduction", type=str, default="concate")
    parser.add_argument("--loop", action="store_true")

    parser.add_argument("--query_gene", type=str, default=None, help="只查某一个基因")
    parser.add_argument("--top_k", type=int, default=50, help="Top K")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--include_self", action="store_true")
    parser.add_argument("--output", type=str, default="infer_scores.csv")

    args = parser.parse_args()

    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    print("使用设备:", device)

    # 1. 读数据
    obj = load_everything(
        dataset=args.dataset,
        cell_type=args.cell_type,
        tf_num=args.tf_num,
        sample=args.sample,
        loop=args.loop
    )

    feature = obj["feature"]
    adj = obj["adj"]
    gene_names = obj["gene_names"]
    num_nodes = feature.shape[0]

    # 2. 建模
    model = build_model(
        feature=feature,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_head=args.num_head,
        alpha=args.alpha,
        score_type=args.Type,
        reduction=args.reduction
    )

    # 3. 加载 checkpoint
    model = load_checkpoint(model, args.checkpoint)

    # 4. 生成待预测边
    if args.query_gene is None:
        print("模式: 全部基因对推断")
        edge_data_np = make_all_pairs(
            num_nodes=num_nodes,
            include_self=args.include_self
        )
    else:
        print(f"模式: 单基因推断 -> {args.query_gene}")
        query_gene = args.query_gene.strip()

        gene_map = {g.upper(): g for g in gene_names}

        if query_gene.upper() not in gene_map:
            print(f"你输入的基因名不存在: {args.query_gene}")

            candidates = difflib.get_close_matches(
                query_gene.upper(),
                list(gene_map.keys()),
                n=10,
                cutoff=0.3
            )

            if candidates:
                print("你可能想查的是：")
                for c in candidates:
                    print("  ", gene_map[c])

            print("表达矩阵里前20个基因名如下：")
            print(gene_names[:20])

            raise ValueError("请复制上面的真实基因名重新查询")

        real_gene_name = gene_map[query_gene.upper()]
        gene_idx = gene_names.index(real_gene_name)

        print(f"已匹配基因名: {real_gene_name}")

        edge_data_np = make_pairs_for_one_gene(
            gene_idx=gene_idx,
            num_nodes=num_nodes,
            include_self=args.include_self
        )

    print("待预测边数 =", edge_data_np.shape[0])

    # 5. 推断
    scores = predict_edges(
        model=model,
        feature=feature,
        adj=adj,
        edge_data_np=edge_data_np,
        batch_size=args.batch_size
    )

    # 6. 保存
    df_all, df_global_topk, df_each_src_topk, df_each_src_top1 = save_result(
        edge_data_np=edge_data_np,
        scores=scores,
        gene_names=gene_names,
        out_csv=args.output,
        top_k=args.top_k
    )

    print(f"\n全网全局 Top {args.top_k}：")
    print(df_global_topk.to_string(index=False))

    print(f"\n每个 src_gene 各自 Top {args.top_k}（前20行预览）：")
    print(df_each_src_topk.head(20).to_string(index=False))

    print("\n每个 src_gene 的 Top 1（前20行预览）：")
    print(df_each_src_top1.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
