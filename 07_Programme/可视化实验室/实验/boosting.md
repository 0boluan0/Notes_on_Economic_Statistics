<!-- visual-lab:generated:start -->
# 提升算法怎样逐轮修正残差

<!-- bilingual-en:start -->
*Residual stump boosting*
<!-- bilingual-en:end -->

真正拟合每一轮最小二乘树桩，查看最后的增量、加法曲线与独立验证误差。

> [!question] 先预测，再操作
> 只会画两段常数的弱学习器，如何逐渐拟合一条弯曲曲线？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/boosting.mp4]]

约 45 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 45 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 第一轮先给每个输入同一个预测
- **0:04** 下一棵小树，学习的是当前没解释的部分
- **0:10** 第 1 次修正：只加一小步
- **0:16** 第 2 次修正：只加一小步
- **0:22** 第 3 次修正：只加一小步
- **0:28** 第 4 次修正：只加一小步
- **0:35** Boosting 保留旧模型，再累加新的修正

<!-- bilingual-en:start -->
- **0:00** Begin with a constant prediction.
- **0:04** The next weak learner fits the current residuals.
- **0:10** Correction 1: add a shrunken residual fit.
- **0:16** Correction 2: add a shrunken residual fit.
- **0:22** Correction 3: add a shrunken residual fit.
- **0:28** Correction 4: add a shrunken residual fit.
- **0:35** Boosting preserves previous learners and adds corrections.
<!-- bilingual-en:end -->

## 亲手实验

<!-- bilingual-en:start -->
*Try it yourself*
<!-- bilingual-en:end -->

在下方展开“影片参数与延伸对照”，选择“载入本片的模型与数据”，先核对影片中的一个对象或运算，再改变一个条件。其他对照用于继续探索。

<!-- bilingual-en:start -->
Expand the film inputs below and load the model and data used in this film. Check one object or calculation, then change one condition. The other comparisons provide further exploration.
<!-- bilingual-en:end -->

<!-- visual-lab:film:end -->
```visual-lab
id: boosting
height: 980
```

## 观察路线

1. 设置 0 轮，确认初始曲线是训练样本均值。
2. 设置 1 轮、η=1，观察最后增量与总体拟合之间的联系。
3. 固定数据逐轮增加 steps，比较训练下降与验证波动。
4. 改变 η 后重新查看同样轮数的模型；每轮残差都会重新计算，不能简单按比例缩放最终曲线。

## 公式与适用条件

$$
F_0=\bar y,\quad r_i^{(t)}=y_i-F_{t-1}(x_i),\quad h_t=\arg\min_h\sum_i(r_i^{(t)}-h(x_i))^2,\quad F_t=F_{t-1}+\eta h_t
$$

- 只实现平方损失、一维输入和每叶至少 3 个观测的回归树桩。
- 每轮穷举候选中点并以叶内残差均值最小化平方和。
- 学习率在 (0,1]，验证集为 250 个独立合成观测，未参与树桩拟合。

## 回到知识点

- [[00_Knowledge/回归残差|回归残差]]：每轮用观测减去当前拟合作为新树桩目标，直接展示残差是可观测拟合缺口；不将其等同生成噪声。
- [[00_Knowledge/点预测损失|点预测损失]]：最小二乘树桩的各叶以残差均值更新，且并列显示训练 MSE 与独立验证 MSE。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/tree|决策树怎样划分空间]] · [[07_Programme/可视化实验室/实验/gradient|梯度下降何时收敛]] · [[07_Programme/可视化实验室/实验/overfit|模型复杂度与泛化]] · [[07_Programme/可视化实验室/实验/modelselection|验证集怎样被选择过程过拟合]]

## 来源与核验

- [scikit-learn · Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html) — §1.11.1.2.3 Mathematical formulation; §1.11.1.2.4 Loss Functions; §1.11.1.2.5 Shrinkage。支持：加法模型、负梯度拟合和收缩学习率；本页只实现平方损失与深度一树桩。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
