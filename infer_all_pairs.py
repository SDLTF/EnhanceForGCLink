import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# ====== 根据你仓库结构调整 import 路径 ======
from GCLink_main_SimSiam import build_model, load_data, DEVICE

# --- 工具函数 ---
def load_checkpoint(model, checkpoint_path):
    """加载训练好的模型参数"""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def compute_all_pair_scores(model, features, edge_index):
    """
    对所有 gene pair 计算预测分数
    features: 节点特征矩阵 (N x F)
    edge_index: 所有可能的基因对 (2 x M)
    """
    with torch.no_grad():
        # 把所有边放进模型 decoder 得到预测分数
        scores = model.predict_edge_scores(features, edge_index) \
                 .detach().cpu().numpy().flatten()
    return scores


def generate_all_pairs(num_nodes):
    """
    生成所有基因对 (不重复组合)
    返回 (2, M) 的 edge_index
    """
    pairs = np.array([(i, j) for i in range(num_nodes) for j in range(num_nodes) if i < j]).T
    return torch.LongTensor(pairs)


def infer_and_save(
    model, features, gene_names, out_csv="pred_scores.csv"
    ):
    """推断 + 保存结果 CSV"""
    # 生成所有 gene pairs
    edge_index = generate_all_pairs(features.shape[0])  # 2 x (N choose 2)

    # 计算预测分数
    scores = compute_all_pair_scores(model, features, edge_index)

    # 保存为 CSV
    import pandas as pd
    df = pd.DataFrame({
        "gene1": gene_names[edge_index[0].numpy()],
        "gene2": gene_names[edge_index[1].numpy()],
        "score": scores
    })
    df.to_csv(out_csv, index=False)
    print(f"Saved all pair scores to {out_csv}")


def get_top_related(df, query_gene, top_k=20):
    """输出与目标基因最相关的 top_k 基因对"""
    # 找到所有含 query_gene 的行
    related = df[(df["gene1"] == query_gene) | (df["gene2"] == query_gene)]
    related = related.sort_values("score", ascending=False)
    return related.head(top_k)


# ===== 脚本主入口 =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="训练好的模型 ckpt 路径")
    parser.add_argument("--output", default="pred_scores.csv", help="输出预测分数 CSV")
    parser.add_argument("--query_gene", default=None, help="如果指定基因，显示前 n 个相关")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    # ---- 1. 加载数据 ----
    data, gene_names = load_data()  # 假设你的 load_data 返回 (Data, Names)
    features = data.x.to(DEVICE)

    # ---- 2. 构建模型 ----
    model = build_model(data).to(DEVICE)
    model = load_checkpoint(model, args.checkpoint)

    # ---- 3. 推断所有 pair 分数 ----
    infer_and_save(model, features, gene_names, out_csv=args.output)

    # ---- 4. 如果指定 query_gene, 显示 top 相关 ----
    if args.query_gene:
        import pandas as pd
        df = pd.read_csv(args.output)
        top_related = get_top_related(df, args.query_gene, args.top_k)
        print(f"\nTop {args.top_k} related pairs of {args.query_gene}:")
        print(top_related)
