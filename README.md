这是一个对原版 GCLink 做改良的仓库。

# Before 3.1

3.1 以前我们尝试了使用 HOLA(Hypergraph Consistency Learning with Relational Distillation) 去替换 GCLink 中对比学习的那一部分。然而结果不尽人意。HOLA 虽然给出了比较快的学习速度，但是 AUC 和 AUPR 相较于原版交友下滑。

关键数据请看：`hola.txt`，`contra.txt`（对照组），`GCLink_main(hola).py`

> 碎碎念：因为我是 3.2 才开始写这个的，所以 3.1 之前的事情我全忘了！  
> 但是有一说一 HOLA 跑的是真的快，我怀疑跟 HOLA 的“强压迫”式监督学习有关系  
> 吐槽：做了一天发现有 0 个作用哈哈哈

---

# 3.1

3.1 我尝试对 GCLink 进行了如下改良：

## A. 训练稳定性与收敛

### A1. 修正学习率调度

* **改动**：删除 batch 循环里的 `scheduler.step()`；在每个 epoch 末尾调用一次。
* **理由**：batch 级 StepLR 会导致学习率在一个 epoch 内衰减很多次，后期几乎不更新。
* **效果证据**：预训练 loss 明显更低、更能持续下降
  * 原版 pretrain：约 **266–270** 区间震荡
  * 优化后 pretrain：降到约 **232**（20 epoch）
> 所以我说能不能让会写代码的人来写代码！为什么会犯这种问题！

### A2. 损失函数数值稳定

* **改动**：保持 `pred` 为 logits，loss 直接用 `BCEWithLogitsLoss`。
* **理由**：更稳、更不容易出现数值/梯度问题；为 `pos_weight` 做准备。
* **效果证据**：训练过程更稳定，后续加入对比 warm-up 后 BCE 可持续下降（见 C1 的 loss 分解）。

## B. 对比学习 / 增强真正生效

### B1. 修复增强逻辑 bug

* **改动**：`GCLink.forward()` 中，encoder 输入从 `data_feature` 改为 `x1` / `x2`。
* **理由**：之前增强特征生成了但没用上，对比信号会变弱或不一致。
* **效果证据**：验证 AUC/AUPR 较原版明显提升（尤其 AUPR 早期就上升到 0.83+）。


### B2. 对比项权重策略

* **改动**：`loss = bce + lam(epoch) * con_loss`，lam 从小逐步到目标（在这个例子中我用到的是 0.1）。
* **理由**：避免对比项在前期压制监督 BCE，训练更稳。
* **效果证据**：loss 分解显示 `bce` 逐步下降、`con` 也下降；total 上升来自 lam 增加，是合理现象。

### B3. 训练日志改造

* **改动**：记录并打印平均 `total/bce/con`。
* **理由**：避免被“总 loss 上升”误导；能定位到底是哪一项导致波动。
* **效果证据**：你看到 `bce` 从约 0.622 → 0.576 逐步下降，说明监督目标确实在学。

> 还行！多少有了点东西！

## C. 评估与模型选择

### C1. 评估阶段禁用随机增强

* **改动**：在 `GCLink.forward()` 加 `if not self.training:` 分支，eval/test 直接用原图，不做 EdgeRemoving。
* **理由**：如果 eval 时还随机删边，val/test 会高度随机，甚至出现“val 高但 test 低”的假象。
* **效果证据**：修复后 test 与 val 对齐，指标稳定。
> 吐槽：再问一次到底是为什么要随机删边……

### C2. 正确保存并加载 “验证集最优” checkpoint 再跑 test

* **改动**：在主训练 loop 中按 **val AUPR** 保存 best；训练结束 `load best` 再 test。
* **理由**：避免加载到旧文件/预训练阶段文件/非最优 epoch 的模型，从而导致 test 被算歪。
* **效果证据**：最终 test 达到与你 val 接近的水平（见下方最终对比）。
> 这真是个隐藏 bug，说实话我很想把以前的 HOLA 再跑一次，我怀疑以前也有这个问题

* **原版最终 test**：AUC **0.910**，AUPR **0.790**
* **优化后最终 test**：AUC **0.928**，AUPR **0.852**
* **提升**：AUC **+0.018**，AUPR **+0.062**

详情代码请看：`GCLink_main(better).py`，`better.txt`。

---
# 3.2

以下是今天干的事情

## A. 把训练 loss 打印改成“可解释”的形式

* 把原来只打印一个 `train loss`（而且容易是累加值）改成 `loss(total/bce/con)` + `lam`
* **目的**：区分“监督 BCE”与“对比损失 con”的贡献，避免被 total loss 误导。
* **结果**：确认了 total 上升主要来自 `lam` warmup，而 BCE 是在下降的（我的意思是训练是健康的）。

## B. 修复了评估阶段的关键问题

* **核心改动**：在 `GCLink.forward()` 里加 `if not self.training:` 分支，让 **eval/test 直接用原图**，不做 EdgeRemoving 这种随机删边增强。
* **目的**：避免 “val 看起来很高但 test 很低/不稳定” 的假象。
* **结果**：修完后 val/test 对齐，test 结果稳定可信。
> 吐槽：到底是为什么回想着在这种非常复杂的对应情况下做随机删边！我不理解！

## C. 修复了模型选择流程：真正保存并加载“验证集最优”的 checkpoint 再测 test

* 把保存 best 的逻辑放回主训练 loop（按 **val AUPR** 保存），训练结束 `load best` 再测试。同时避免了 `pretrain()` 误包含训练/测试导致 best 文件被覆盖的问题。
* **结果**：最终 test 结果从“偏低/不可信”变成和 val 一致（说明流程正确，也说明 bug 给他修好了）。
> 完蛋了！前一天的 bug 没修好！

## D. 开始做 decoder ablation：新增 bilinear 和 edge\_mlp 两种 decoder

* **dot**：点积 $u^\top v$（原版）
* **bilinear**：双线性 $u^\top W v$（因为考虑到基因和对应表达用有向边描述更好）
* **edge\_mlp**：边特征 $[u,v,u\odot v,|u-v|]$ + MLP（更强的表达）
* 结论是：**edge\_mlp > bilinear > dot**（这个提升在 val/test 都成立）。
> 疑似 SD 同学实在想做点东西出来于是盯上了 decoder，很难想象明天 encoder 还会不会活着

## E. 继续“压榨 edge\_mlp”！

大概是今天下午的事情，我把他的四位拓展到了六维！

* 边特征从 4 项扩展到 6 项：
  $[u,v,u\odot v,|u-v|,u-v,u+v]$
* MLP 从简单两层升级为更稳更强的结构：
  `Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear(1)`
* 同时修了 bilinear 分支缩进、并讨论了避免重复 test/避免不同 decoder 覆盖 best 文件名等工程细节。

在当前配置下（修复评估 + best checkpoint + edge\_mlp 压榨/或同等版本）跑到：

* `best_val_AUPR ≈ 0.855`
* `test_AUC ≈ 0.929`
* `test_AUPR ≈ 0.857`
  并且 val/test 非常接近，说明泛化与评估流程可靠。
> 跑的也确实感觉要慢了点。我们需要更强大的运行机器！


详情代码请看：`better2.txt`，`betterEdgeMLP.txt`，`scGNNv1.py`，`scGNNv2.py`，`GCLink_main(better2).py`

---
# 3.3

今天没有任何有效进展。

正在尝试用 MAE 替换 Info。

---
# 3.4

今天停工。收拾东西准备回学校了。

---
# 3.7 
今天尝试使用 VICReg 代替 Info。

现在训练和评估流程是正常的。

但 VICReg 在这个设置下并没有带来提升，而且显著引入了更大的指标波动，例如，这玩意在 epoch7/14/16 出现多次 AUPR 跌到 0.62~0.71 的“真波动”，不是之前 Evaluation 的假波动了。

相比之下，原来的 InfoNCE到 epoch19/20 能稳定到 AUPR 0.853~0.855，明显更强。

所以，如果目标是“替代 InfoNCE 且不掉点”，现在这版 VICReg 需要改的不是 con_w，而是 VICReg 的输入方式（用全节点 embedding 做 VICReg，会把大量与边任务无关的节点卷进来，影响 edge ranking）

具体的东西，明天再修。

详情请看 `utils2.py`，`GCLink_main_VICReg.py`，`EMLP_VICReg.txt`

# 3.9
我们抛弃了 VICReg 转向了 SimSiam，跑出了跟历史最高纪录几乎一样好的数据。

现在开始微微调参。

详情请看 `GCLink_main_SimSiam.py`，`SimSiam.txt`