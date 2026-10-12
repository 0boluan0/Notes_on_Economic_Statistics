<!-- visual-lab:generated:start -->
# 随机特征的插值与经验双降

<!-- bilingual-en:start -->
*Random features, interpolation & empirical double descent*
<!-- bilingual-en:end -->

实际求解固定数据下不同宽度的随机特征回归，观察插值、最小范数与独立验证误差。

> [!question] 先预测，再操作
> 训练误差已经为零，增加特征数为什么仍可能改变验证误差？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/double-descent.mp4]]

约 33 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 33 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先数清参数：宽度之外，还有截距
- **0:04** 等参数数目附近，矩阵可能变得敏感
- **0:10** 沿真实计算结果，逐点看验证风险
- **0:23** 容量曲线，也受正则化与数据影响

<!-- bilingual-en:start -->
- **0:00** Count parameters before locating interpolation.
- **0:04** The interpolation region can be ill-conditioned.
- **0:10** Inspect the actual validation curve, not a promised shape.
- **0:23** Capacity curves depend on data and regularization.
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
id: double-descent
height: 980
```

## 观察路线

1. 保持数据和种子固定，依次看欠参数、接近插值、过参数区域。
2. 确认横轴只数随机特征，参数总数另含截距，因此 m=n−1 是方形设计位置。
3. 增加 λ，解释训练误差可以升高但验证误差可能下降。
4. 更换几个种子；仅对实际出现的风险形状称为经验双降，不能预先保证。

## 公式与适用条件

$$
\phi_j(x)=\sqrt2\cos(w_j^Tx/\sqrt5+b_j),\quad\hat f(x)=\beta_0+\sum_{j=1}^m\beta_j\phi_j(x),\quad\hat\beta\in\arg\min_\beta\|y-X\beta\|^2+\lambda\sum_{j=1}^m\beta_j^2
$$

- x∈R⁵ 为独立标准 Gaussian，w 同样 Gaussian，b 为均匀相位；总特征上限 80。
- 真函数固定为前 10 个特征之和除以 √10，观测再加独立 Gaussian 噪声。验证集含 240 个独立观测。
- λ>0 惩罚非截距系数；λ=0 时欠定解最小化包括截距在内的完整系数范数。

## 回到知识点

- [[00_Knowledge/最小范数最小二乘解|最小范数最小二乘解]]：过参数随机特征模型在多个插值解中选择完整系数欧氏范数最小者；零训练误差不能确定新点预测。
- [[00_Knowledge/岭正则化|岭正则化]]：平方损失中加入系数惩罚；本实验显式排除截距，故其 λ→0 极限与全系数最小范数选择需区分。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/overfit|模型复杂度与泛化]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]] · [[07_Programme/可视化实验室/实验/regularization|Dropout保留均值，却怎样改变损失]] · [[07_Programme/可视化实验室/实验/modelselection|验证集怎样被选择过程过拟合]]

## 来源与核验

- [Belkin et al. · Reconciling modern ML practice and the bias–variance trade-off](https://arxiv.org/pdf/1812.11118) — §2 Random Fourier features; Appendix C.1。支持：随机特征、插值附近风险变化、欠定时最小范数选择。当前五维合成回归数据不是论文 MNIST 结果的复制。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
