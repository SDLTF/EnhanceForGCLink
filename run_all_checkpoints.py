import os
import re
import glob
import subprocess
import sys
import argparse

INFER_SCRIPT = "infer_gene_pairs.py"
MODEL_ROOT = "model"

MODEL_DIRS = [
    os.path.join(MODEL_ROOT, "model_Non-Specific"),
    os.path.join(MODEL_ROOT, "model_Specific"),
    os.path.join(MODEL_ROOT, "model_STRING"),
]

# 格式 1:
# Specific_tf1000_hESC_sample1_edge_mlp_seed8_best_checkpoint.pt
PATTERN_FULL = re.compile(
    r'^(?P<dataset>Non-Specific|Specific|STRING)_tf(?P<tf_num>\d+)_(?P<cell_type>.+?)_(?P<sample>sample\d+)_(?P<score_type>.+?)_seed(?P<seed>\d+)_best_checkpoint\.pt$'
)

# 格式 2:
# hESC_sample1_edge_mlp_seed8_best_checkpoint.pt
PATTERN_STRING = re.compile(
    r'^(?P<cell_type>.+?)_(?P<sample>sample\d+)_(?P<score_type>.+?)_seed(?P<seed>\d+)_best_checkpoint\.pt$'
)


def parse_ckpt_name(filename, parent_dir):
    """
    返回:
    {
        "dataset": ...,
        "tf_num": ...,
        "cell_type": ...,
        "sample": ...,
        "score_type": ...,
        "seed": ...
    }
    """
    m = PATTERN_FULL.match(filename)
    if m:
        return m.groupdict()

    if os.path.basename(parent_dir) == "model_STRING":
        m = PATTERN_STRING.match(filename)
        if m:
            info = m.groupdict()
            info["dataset"] = "STRING"
            info["tf_num"] = guess_string_tf_num(info["cell_type"], info["sample"])
            return info

    return None


def guess_string_tf_num(cell_type, sample):
    """
    STRING 文件名里没有 tf_num。
    优先找 1000，再找 500。
    """
    candidates = [
        os.path.join("Data", "STRING", f"{cell_type} 1000", sample),
        os.path.join("Data", "STRING", f"{cell_type} 500", sample),
    ]

    for path in candidates:
        if os.path.exists(path):
            if "1000" in path:
                return "1000"
            if "500" in path:
                return "500"

    return "1000"


def safe_name(filename):
    return os.path.splitext(filename)[0]


def expected_outputs_exist(out_dir, top_k):
    """
    判断这个 checkpoint 是否已经跑完。
    这里只检查最关键的几个输出文件。
    """
    required_files = [
        os.path.join(out_dir, "all_pairs.csv"),
        os.path.join(out_dir, f"all_pairs_global_top{top_k}.csv"),
        os.path.join(out_dir, f"all_pairs_each_src_top{top_k}.csv"),
        os.path.join(out_dir, "all_pairs_each_src_top1.csv"),
    ]
    return all(os.path.exists(f) for f in required_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=50, help="传给 infer_gene_pairs.py 的 top_k")
    parser.add_argument("--force", action="store_true", help="即使已有结果也强制重跑")
    args = parser.parse_args()

    if not os.path.exists(INFER_SCRIPT):
        print(f"[错误] 找不到推断脚本: {INFER_SCRIPT}")
        sys.exit(1)

    all_pt_files = []

    for model_dir in MODEL_DIRS:
        if not os.path.exists(model_dir):
            print(f"[跳过] 目录不存在: {model_dir}")
            continue

        pt_files = glob.glob(os.path.join(model_dir, "*.pt"))
        if not pt_files:
            print(f"[跳过] 目录下没有 .pt 文件: {model_dir}")
            continue

        print(f"[发现] {model_dir} 中有 {len(pt_files)} 个 .pt 文件")
        all_pt_files.extend(pt_files)

    if not all_pt_files:
        print("没有找到任何 .pt 文件")
        sys.exit(0)

    print(f"\n总共找到 {len(all_pt_files)} 个 .pt 文件\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for pt_file in all_pt_files:
        ckpt_name = os.path.basename(pt_file)
        parent_dir = os.path.dirname(pt_file)

        print("=" * 80)
        print(f"[开始] 处理 checkpoint: {pt_file}")

        meta = parse_ckpt_name(ckpt_name, parent_dir)
        if meta is None:
            print(f"[跳过] 文件名不符合规则，无法自动解析: {ckpt_name}")
            skip_count += 1
            continue

        dataset = meta["dataset"]
        tf_num = meta["tf_num"]
        cell_type = meta["cell_type"]
        sample = meta["sample"]
        score_type = meta["score_type"]

        print(f"  dataset   = {dataset}")
        print(f"  tf_num    = {tf_num}")
        print(f"  cell_type = {cell_type}")
        print(f"  sample    = {sample}")
        print(f"  Type      = {score_type}")

        data_dir = os.path.join("Data", dataset, f"{cell_type} {tf_num}", sample)
        if not os.path.exists(data_dir):
            print(f"[跳过] 对应数据目录不存在: {data_dir}")
            skip_count += 1
            continue

        out_dir = os.path.join(parent_dir, "results", safe_name(ckpt_name))
        os.makedirs(out_dir, exist_ok=True)

        if (not args.force) and expected_outputs_exist(out_dir, args.top_k):
            print(f"[跳过] 结果已存在: {out_dir}")
            skip_count += 1
            continue

        out_csv = os.path.join(out_dir, "all_pairs.csv")

        cmd = [
            sys.executable, INFER_SCRIPT,
            "--checkpoint", pt_file,
            "--dataset", dataset,
            "--cell_type", cell_type,
            "--tf_num", tf_num,
            "--sample", sample,
            "--Type", score_type,
            "--output", out_csv,
            "--top_k", str(args.top_k)
        ]

        print("[运行命令]")
        print(" ".join(cmd))

        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"[完成] 推断成功: {pt_file}")
            print(f"[输出目录] {out_dir}")
            success_count += 1
        else:
            print(f"[失败] 推断出错: {pt_file}")
            print(f"[返回码] {result.returncode}")
            fail_count += 1

    print("\n全部处理结束。")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")


if __name__ == "__main__":
    main()
