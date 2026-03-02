这是一个对原版 GCLink 做改良的仓库。

# Before 3.1

3.1 以前我们尝试了使用 HOLA(Hypergraph Consistency Learning with Relational Distillation) 去替换 GCLink 中对比学习的那一部分。然而结果不尽人意。HOLA 虽然给出了比较快的学习速度，但是 AUC 和 AUPR 相较于原版交友下滑。

---

# 3.1

3.1 我们尝试对 GCLink 进行了如下改良：

## A. 训练稳定性与收敛

### A1. 修正学习率调度

* **改动**：删除 batch 循环里的 `scheduler.step()`；在每个 epoch 末尾调用一次。
* **理由**：batch 级 StepLR 会导致学习率在一个 epoch 内衰减很多次，后期几乎不更新。
* **效果证据**：预训练 loss 明显更低、更能持续下降
  * 原版 pretrain：约 **266–270** 区间震荡
  * 优化后 pretrain：降到约 **232**（20 epoch）

### A2. 损失函数数值稳定

* **改动**：保持 `pred` 为 logits，loss 直接用 `BCEWithLogitsLoss`。
* **理由**：更稳、更不容易出现数值/梯度问题；为 `pos_weight` 做准备。
* **效果证据**：训练过程更稳定，后续加入对比 warm-up 后 BCE 可持续下降（见 C1 的 loss 分解）。

## B. 对比学习/增强真正生效

### B1. 修复增强逻辑 bug

* **改动**：`GCLink.forward()` 中，encoder 输入从 `data_feature` 改为 `x1` / `x2`。
* **理由**：之前增强特征生成了但没用上，对比信号会变弱或不一致。
* **效果证据**：验证 AUC/AUPR 较原版明显提升（尤其 AUPR 早期就上升到 0.83+）。


### B2. 对比项权重策略

* **改动**：`loss = bce + lam(epoch) * con_loss`，lam 从小逐步到目标（你用到 0.1）。
* **理由**：避免对比项在前期压制监督 BCE，训练更稳。
* **效果证据**：loss 分解显示 `bce` 逐步下降、`con` 也下降；total 上升来自 lam 增加，是合理现象。

### B3. 训练日志改造

* **改动**：记录并打印平均 `total/bce/con`。
* **理由**：避免被“总 loss 上升”误导；能定位到底是哪一项导致波动。
* **效果证据**：你看到 `bce` 从约 0.622 → 0.576 逐步下降，说明监督目标确实在学。

## C. 评估与模型选择

### C1. 评估阶段禁用随机增强

* **改动**：在 `GCLink.forward()` 加 `if not self.training:` 分支，eval/test 直接用原图，不做 EdgeRemoving。
* **理由**：如果 eval 时还随机删边，val/test 会高度随机，甚至出现“val 高但 test 低”的假象。
* **效果证据**：修复后 test 与 val 对齐，指标稳定。


### C2. 正确保存并加载 “验证集最优” checkpoint 再跑 test

* **改动**：在主训练 loop 中按 **val AUPR** 保存 best；训练结束 `load best` 再 test。
* **理由**：避免加载到旧文件/预训练阶段文件/非最优 epoch 的模型，从而导致 test 被算歪。
* **效果证据**：最终 test 达到与你 val 接近的水平（见下方最终对比）。


* **原版最终 test**：AUC **0.910**，AUPR **0.790**
* **优化后最终 test**：AUC **0.928**，AUPR **0.852**
* **提升**：AUC **+0.018**，AUPR **+0.062**

---
# 3.2

下面这段你可以直接写进工作汇报里：我把 **dot / bilinear / edge\_mlp** 三种 decoder 分别是什么、怎么计算 TF→Target 的打分（logit）、各自特点写清楚了，并且配上你这次实验的提升数据。

---

## Decoder 是什么（在 GCLink 里做什么）

在 GCLink 的 link prediction 里，encoder 会分别得到 TF 节点向量 $u$ 和 target 基因向量 $v$（维度 $d$）。**decoder 的作用**就是把 $(u,v)$ 映射成一条边存在的打分（logit）$s(u,v)$，再经过 sigmoid 得到概率 $p=\sigma(s)$。


## 1. dot decoder（基线）

**定义：**

$$
s(u,v) = u^\top v
$$

也就是 embedding 的点积（dot product）。

**特点：**

* **无额外参数**，计算最简单；
* 关系形式相对“线性/简单”，表达力有限；
* 点积本身偏“对称相似性”，对有向关系（TF→Target）不一定最合适（虽然训练时输入方向固定也能学出差异，但 decoder 本身偏相似度）。

**基线结果（dot）：**

* test\_AUC = **0.928**
* test\_AUPR = **0.852**&#x20;


## 2. bilinear decoder

**定义：**

$$
s(u,v) = u^\top W v
$$

其中 $W\in \mathbb{R}^{d\times d}$ 是可学习参数矩阵。

**特点：**

* 相比 dot，多了一个矩阵 $W$，能学习“TF→Target 的关系变换”；
* **天然支持有向关系**：一般 $u^\top W v \neq v^\top W u$；
* 表达力强于 dot，但仍属于“单层线性关系”。

**结果**

* test\_AUC = **0.928**
* test\_AUPR = **0.855**&#x20;
  **相对 dot 提升：**
* test\_AUPR **+0.003**（0.852 → 0.855）
* test\_AUC **+0.000**


## 3. edge\_mlp decoder（边特征 + MLP）

**定义：**先构造更丰富的边特征，再用小型 MLP 输出 logit：

$$
\phi(u,v)=\big[u,\; v,\; u\odot v,\; |u-v|\big]
$$

$$
s(u,v)=\text{MLP}(\phi(u,v))
$$

其中：

* $u\odot v$：逐元素乘（interaction）
* $|u-v|$：逐元素差的绝对值（distance-like）

**特点：**

* 表达力最强：既包含原始向量，又包含交互项和差异项；
* MLP 引入非线性，能拟合更复杂的 TF→Target 判别边界；
* 计算略贵、参数更多，但在 GRN/link prediction 里常常能显著提高 AUPR。

**结果**

* best\_val\_AUPR = **0.872**
* test\_AUC = **0.934**
* test\_AUPR = **0.869**&#x20;
  **相对 dot 提升：**
* test\_AUPR **+0.017**（0.852 → 0.869，约 +2.0%）
* test\_AUC **+0.006**（0.928 → 0.934）
* best\_val\_AUPR **+0.020**（相对 bilinear/dot 都更高）

总的来说，在保持 encoder 与训练设置一致的情况下，我们将 GCLink 的 link decoder 从基线 **dot（点积 $u^\top v$）**替换为 **bilinear（双线性 $u^\top W v$）**与 **edge\_mlp（边特征 $[u,v,u\odot v,|u-v|]$ + MLP）**。其中 **edge\_mlp** 带来显著提升：test\_AUPR 从 **0.852 提升到 0.869（+0.017）**，test\_AUC 从 **0.928 提升到 0.934（+0.006）**；bilinear 仅带来 test\_AUPR **+0.003** 的小幅收益。  &#x20;
