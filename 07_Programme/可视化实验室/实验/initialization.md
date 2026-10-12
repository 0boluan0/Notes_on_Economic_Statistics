<!-- visual-lab:generated:start -->
# 初始化如何改变信号与梯度尺度

<!-- bilingual-en:start -->
*Initialization & actual signal propagation*
<!-- bilingual-en:end -->

真实生成多层随机权重，执行前向与反向计算，比较激活二阶矩、方差和梯度幅度。

> [!question] 先预测，再操作
> He初始化的尺度推导，是否意味着每次运行的每一层方差都严格为1？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/initialization.mp4]]

约 56 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 56 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先追踪第一层的一个神经元
- **0:10** 初始化，先决定信号进入下一层的尺度
- **0:28** 梯度从另一端回来，也有自己的尺度
- **0:38** 为什么要按输入宽度缩放权重？
- **0:47** 这是尺度分析，仍须检查具体网络

<!-- bilingual-en:start -->
- **0:00** Track one actual neuron before summarizing the network.
- **0:10** Initialization sets the scale of transmitted signals.
- **0:28** Backward signal scale deserves a separate check.
- **0:38** Weight variance compensates for accumulated inputs.
- **0:47** A scaling argument is not a guarantee for every finite network.
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
id: initialization
height: 980
```

## 观察路线

1. 固定种子，在三种尺度之间切换，以相同基础随机数比较前向二阶矩。
2. 区分ReLU非零均值带来的二阶矩与中心方差差别。
3. 看第二图时从右向左读，明确梯度来自同一个固定末层线性读出。
4. 换几个种子，检查有限宽度随机波动，而不是只记住一次曲线形状。

## 公式与适用条件

$$
h^{(\ell)}=\phi(W^{(\ell)}h^{(\ell-1)}),\quad \mathrm{Var}(W_{ij})=\frac{c\,g^2}{24},\quad c=1\text{ (Xavier)},\ 2\text{ (He)}
$$

- 各层权重独立正态、零偏置、固定宽度24；Xavier采用原论文方差条件的正态版本，原论文式(16)为均匀分布。不是已训练网络。
- 图的统计汇总48×24个激活值；这些值并非全都相互独立。
- 梯度针对末层各单元和除以√24的标量读出；不是某个真实任务损失。

## 回到知识点

这个主题尚未连接已有知识原子；定义、假设与来源可在本页和实验内阅读。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/activations|激活函数与局部梯度]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]] · [[07_Programme/可视化实验室/实验/normalization|BatchNorm与LayerNorm沿哪个轴统计]] · [[07_Programme/可视化实验室/实验/residual|残差捷径怎样改变梯度乘积]]

## 来源与核验

- [Glorot & Bengio (AISTATS 2010)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — §4.2.1; Eq. (10)–(12), PDF p.5。支持：初始化的层间信号和梯度尺度；等宽Xavier方差1/fan-in。
- [He et al. (ICCV 2015)](https://arxiv.org/pdf/1502.01852) — §2.2; Eq. (6)–(10)。支持：ReLU情形的二阶矩传播与方差2/fan-in；独立性和对称性假设下的尺度分析。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
