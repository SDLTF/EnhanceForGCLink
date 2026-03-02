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
