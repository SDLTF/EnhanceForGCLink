import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# 1) 配置
# =========================
methods = ["GCLink-Enhanced", "GCLink", "GENELink", "CNNC", "GNE", "DeepSEM", "GENIE3", "GRNBoost2"]
groups = ["Specific", "Non-Specific", "STRING"]
datasets = ["mHSC-E", "mHSC-L", "mHSC-GM", "mESC", "mDC", "hESC", "hHEP"]

# =========================
# 2) 原论文数据
#    原始顺序:
#    [GCLink, GENELink, CNNC, GNE, DeepSEM, GENIE3, GRNBoost2]
# =========================
paper_data = {
    "TFs+1000": {
        "Specific": {
            "mHSC-E": [0.915, 0.892, 0.897, 0.801, 0.562, 0.539, 0.555],
            "mHSC-L": [0.832, 0.813, 0.735, 0.658, 0.543, 0.494, 0.483],
            "mHSC-GM": [0.924, 0.905, 0.876, 0.794, 0.539, 0.539, 0.541],
            "mESC":   [0.780, 0.755, 0.737, 0.471, 0.315, 0.329, 0.326],
            "mDC":    [0.135, 0.115, 0.047, 0.083, 0.041, 0.047, 0.047],
            "hESC":   [0.554, 0.497, 0.421, 0.328, 0.170, 0.145, 0.143],
            "hHEP":   [0.776, 0.693, 0.645, 0.540, 0.418, 0.372, 0.355],
        },
        "Non-Specific": {
            "mHSC-E": [0.142, 0.102, 0.022, 0.030, 0.061, 0.066, 0.062],
            "mHSC-L": [0.142, 0.121, 0.046, 0.044, 0.122, 0.116, 0.109],
            "mHSC-GM": [0.214, 0.239, 0.034, 0.052, 0.092, 0.104, 0.092],
            "mESC":   [0.062, 0.038, 0.021, 0.021, 0.023, 0.023, 0.023],
            "mDC":    [0.147, 0.064, 0.021, 0.020, 0.026, np.nan, 0.025],
            "hESC":   [0.046, 0.039, 0.015, 0.017, 0.019, 0.016, 0.016],
            "hHEP":   [0.060, 0.049, 0.014, 0.018, 0.017, np.nan, 0.016],
        },
        "STRING": {
            "mHSC-E": [0.217, 0.193, 0.028, 0.060, 0.137, 0.134, 0.112],
            "mHSC-L": [0.277, 0.269, 0.083, 0.053, 0.323, 0.306, 0.307],
            "mHSC-GM": [0.285, 0.237, 0.045, 0.081, 0.214, 0.245, 0.211],
            "mESC":   [0.232, 0.129, 0.052, 0.045, 0.055, 0.048, 0.046],
            "mDC":    [0.393, 0.263, 0.054, 0.564, 0.053, 0.048, 0.047],
            "hESC":   [0.271, 0.165, 0.039, 0.048, 0.046, 0.043, 0.041],
            "hHEP":   [0.276, 0.188, 0.028, 0.043, 0.040, 0.049, 0.041],
        },
    },
    "TFs+500": {
        "Specific": {
            "mHSC-E": [0.910, 0.889, 0.887, 0.775, 0.576, 0.548, 0.571],
            "mHSC-L": [0.843, 0.817, 0.582, 0.668, 0.554, 0.515, 0.517],
            "mHSC-GM": [0.902, 0.894, 0.869, 0.781, 0.546, 0.531, 0.534],
            "mESC":   [0.773, 0.759, 0.782, 0.467, 0.315, 0.329, 0.321],
            "mDC":    [0.149, 0.113, 0.048, 0.075, 0.046, 0.048, 0.044],
            "hESC":   [0.541, 0.506, 0.226, 0.346, 0.171, 0.144, 0.147],
            "hHEP":   [0.781, 0.681, 0.479, 0.524, 0.388, 0.369, 0.357],
        },
        "Non-Specific": {
            "mHSC-E": [0.163, 0.136, 0.026, 0.034, 0.081, 0.060, 0.060],
            "mHSC-L": [0.116, 0.113, 0.050, 0.049, 0.112, 0.137, 0.137],
            "mHSC-GM": [0.239, 0.222, 0.036, 0.039, 0.095, np.nan, 0.100],
            "mESC":   [0.074, 0.036, 0.024, 0.023, 0.023, np.nan, 0.024],
            "mDC":    [0.166, 0.146, 0.025, 0.026, 0.036, 0.031, 0.031],
            "hESC":   [0.054, 0.036, 0.017, 0.021, 0.022, 0.020, 0.019],
            "hHEP":   [0.073, 0.046, 0.016, 0.022, 0.020, np.nan, 0.020],
        },
        "STRING": {
            "mHSC-E": [0.248, 0.228, 0.103, 0.058, 0.166, 0.151, 0.144],
            "mHSC-L": [0.329, 0.284, 0.090, 0.067, 0.335, np.nan, 0.284],
            "mHSC-GM": [0.381, 0.356, 0.057, 0.087, 0.283, 0.333, 0.283],
            "mESC":   [0.251, 0.127, 0.058, 0.056, 0.049, 0.047, 0.050],
            "mDC":    [0.408, 0.262, 0.070, 0.057, 0.059, np.nan, 0.055],
            "hESC":   [0.274, 0.203, 0.038, 0.058, 0.051, 0.045, 0.043],
            "hHEP":   [0.293, 0.236, 0.033, 0.058, 0.048, 0.059, 0.051],
        },
    },
}

# =========================
# 3) 你们优化后的 GCLink
# =========================
optimized_gclink = {
    "TFs+1000": {
        "Specific": {"mHSC-E": 0.931, "mHSC-L": 0.838, "mHSC-GM": 0.923, "mESC": 0.846, "mDC": 0.171, "hESC": 0.587, "hHEP": 0.809},
        "Non-Specific": {"mHSC-E": 0.168, "mHSC-L": 0.141, "mHSC-GM": 0.284, "mESC": 0.104, "mDC": 0.141, "hESC": 0.032, "hHEP": 0.051},
        "STRING": {"mHSC-E": 0.390, "mHSC-L": 0.298, "mHSC-GM": 0.380, "mESC": 0.323, "mDC": 0.434, "hESC": 0.304, "hHEP": 0.358},
    },
    "TFs+500": {
        "Specific": {"mHSC-E": 0.914, "mHSC-L": 0.845, "mHSC-GM": 0.899, "mESC": 0.842, "mDC": 0.161, "hESC": 0.580, "hHEP": 0.800},
        "Non-Specific": {"mHSC-E": 0.102, "mHSC-L": 0.160, "mHSC-GM": 0.112, "mESC": 0.160, "mDC": 0.133, "hESC": 0.123, "hHEP": 0.107},
        "STRING": {"mHSC-E": 0.303, "mHSC-L": 0.297, "mHSC-GM": 0.416, "mESC": 0.324, "mDC": 0.448, "hESC": 0.305, "hHEP": 0.373},
    },
}

# =========================
# 4) 插入新列
# =========================
            # 插入新列
plot_data = {}
for setting in paper_data:
    plot_data[setting] = {}
    for group in paper_data[setting]:
        plot_data[setting][group] = {}
        for ds in paper_data[setting][group]:
            old_vals = paper_data[setting][group][ds]
            plot_data[setting][group][ds] = [
                optimized_gclink[setting][group][ds],   # 优化版 GCLink 放第一列
                old_vals[0],                           # 原始 GCLink
                old_vals[1],                           # GENELink
                old_vals[2],                           # CNNC
                old_vals[3],                           # GNE
                old_vals[4],                           # DeepSEM
                old_vals[5],                           # GENIE3
                old_vals[6],                           # GRNBoost2
            ]


# =========================
# 5) 拼 DataFrame，保留分组空行
# =========================
def build_df(setting_name):
    rows = []
    idx = []
    for group in groups:
        for ds in datasets:
            rows.append(plot_data[setting_name][group][ds])
            idx.append(ds)
    return pd.DataFrame(rows, index=idx, columns=methods)


df_1000 = build_df("TFs+1000")
df_500 = build_df("TFs+500")

# =========================
# 6) 颜色
# =========================
cmap = mpl.cm.Blues.copy()
cmap.set_bad("#d9d9d9")

# =========================
# 7) 全局样式：尽量贴近原图
# =========================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
})

fig = plt.figure(figsize=(14, 12), facecolor="#ffffff")

# 主体区域更接近原图，给左侧组标签留足空间
ax1 = fig.add_axes([0.22, 0.18, 0.335, 0.67])   # left, bottom, width, height
ax2 = fig.add_axes([0.57, 0.18, 0.335, 0.67])
cax = fig.add_axes([0.22, 0.10, 0.61, 0.02])

def draw_panel(ax, df, title, show_group_labels=False, show_ylabels=True):
    ax.set_facecolor("#ececec")
    im = ax.imshow(df.values, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="none")

    # 细白色分组分隔线
    sep_positions = [6.5, 13.5]
    for y in sep_positions:
        ax.hlines(y, -0.5, df.shape[1] - 0.5, colors="white", linewidth=3, zorder=4)

    # x ticks
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(
        df.columns,
        rotation=45,
        ha="left",
        rotation_mode="anchor",
        fontsize=12,
        fontweight="bold"
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=2, length=0)

    # y ticks
    ax.set_yticks(np.arange(len(df.index)))
    if show_ylabels:
        ax.set_yticklabels(df.index, fontsize=14, fontweight="bold")
    else:
        ax.set_yticklabels([""] * len(df.index))
    ax.tick_params(axis="y", length=0, pad=1)

    # title
    ax.set_title(title, fontsize=20, fontweight="bold", pad=25)

    # remove frame
    for s in ax.spines.values():
        s.set_visible(False)

    # 找到 GCLink+ 这一列
    gclink_plus_col = list(df.columns).index("GCLink-Enhanced")

    # 每一行的 SOTA 列
    sota_pos = {}
    for i in range(df.shape[0]):
        row = df.iloc[i].values.astype(float)
        if np.all(np.isnan(row)):
            continue
        max_val = np.nanmax(row)
        sota_cols = np.where(np.isclose(row, max_val, atol=1e-12, equal_nan=False))[0]
        sota_pos[i] = set(sota_cols.tolist())

    yellow = "#fff2a8"
    red = "#c40000"

    # 先画 SOTA 背景
    for i in range(df.shape[0]):
        if i not in sota_pos:
            continue
        for j in sota_pos[i]:
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=yellow, edgecolor="none", zorder=2
            )
            ax.add_patch(rect)

    # 再写数字
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iat[i, j]
            if np.isnan(v):
                continue

            is_sota = i in sota_pos and j in sota_pos[i]
            is_gclink_plus_sota = is_sota and (j == gclink_plus_col)

            ax.text(
                j, i, f"{v:.3f}",
                ha="center", va="center",
                fontsize=11,
                color=red if is_gclink_plus_sota else "black",
                fontweight="bold" if is_gclink_plus_sota else "normal",
                zorder=5
            )

    # 左侧组标签
    if show_group_labels:
        centers = [3, 10, 17]
        names = ["Specific", "Non-Specific", "STRING"]
        for y, name in zip(centers, names):
            ax.text(-2.2, y, name, ha="right", va="center", fontsize=20, fontweight="bold")

    return im


im = draw_panel(ax1, df_1000, "TFs+1000", show_group_labels=True, show_ylabels=True)
draw_panel(ax2, df_500, "TFs+500", show_group_labels=False, show_ylabels=False)


# colorbar
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_ticks(np.linspace(0, 1, 6))
cb.ax.tick_params(labelsize=16, width=2, length=5)
cb.outline.set_linewidth(2)

# 不再加长 caption，避免破坏原图风格
plt.savefig("figure2_better_with_opt.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig("figure2_better_with_opt.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
