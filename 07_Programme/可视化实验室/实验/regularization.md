<!-- visual-lab:generated:start -->
# Dropout保留均值，却怎样改变损失

<!-- bilingual-en:start -->
*Inverted dropout & exact conditional moments*
<!-- bilingual-en:end -->

直接抽取随机掩码，并枚举四个贡献值的全部16种情况，比较模拟结果与精确期望。

> [!question] 先预测，再操作
> 倒置Dropout保持线性读出均值，为什么期望平方损失仍会增加？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/regularization.mp4]]

约 36 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 36 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一次 Dropout，究竟丢掉了什么？
- **0:10** 保留下来的项，还要除以保留概率
- **0:20** 一次会变，平均值保持什么？
- **0:26** 保持期望，不等于每次保持输出

<!-- bilingual-en:start -->
- **0:00** Follow one actual dropout mask.
- **0:10** Inverted dropout rescales the survivors.
- **0:20** Random outputs can have a stable expectation.
- **0:26** Preserving the mean does not preserve each realization.
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
id: regularization
height: 980
```

## 观察路线

1. 默认贡献值为(1,0.6,−0.4,0.8)，先手算不加掩码的输出。
2. 检查掩码热图，说明每次哪些贡献被保留、为何还要除以保留率。
3. 提高p，比较精确均值保持与精确方差增长。
4. 把目标设为无掩码输出，检查期望损失是否仍有正的方差项。

## 公式与适用条件

$$
\tilde y=\sum_j\frac{m_ja_j}{1-p},\ m_j\sim\mathrm{Bernoulli}(1-p),\quad E\tilde y=\sum_ja_j,\quad \mathrm{Var}(\tilde y)=\frac p{1-p}\sum_ja_j^2
$$

- 四个贡献固定，独立Bernoulli掩码；p<1，避免除以零。
- 只对线性读出和平方损失推导精确等价；没有训练、验证集或泛化改善声明。
- 模拟方差用B作分母，对照的是给定贡献值下的总体方差。

## 回到知识点

这个主题尚未连接已有知识原子；定义、假设与来源可在本页和实验内阅读。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]] · [[07_Programme/可视化实验室/实验/overfit|模型复杂度与泛化]] · [[07_Programme/可视化实验室/实验/initialization|初始化如何改变信号与梯度尺度]]

## 来源与核验

- [Srivastava et al. · Dropout (JMLR 2014)](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — §4 Model Description; Fig. 2（训练/测试缩放）。支持：独立随机保留及训练/测试缩放；原论文p表示保留率，本页p表示丢弃率。固定线性读出的倒置缩放、方差和平方损失分解为直接推导。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
