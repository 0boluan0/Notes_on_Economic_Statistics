<!-- visual-lab:generated:start -->
# 反向传播把局部导数串起来

<!-- bilingual-en:start -->
*Backpropagation & finite-difference checking*
<!-- bilingual-en:end -->

把一个小网络的中间值和反向梯度逐个展开，用独立有限差分检验参数梯度。

> [!question] 先预测，再操作
> 为什么隐藏单元很活跃，却可能收不到损失梯度？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/backprop.mp4]]

约 52 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 52 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先让一个输入走到损失
- **0:14** 局部导数，把上游变化传回来
- **0:30** 同一个节点，还要把梯度交给参数
- **0:43** 链式法则是在组合敏感度

<!-- bilingual-en:start -->
- **0:00** One input, four operations, one loss.
- **0:14** Upstream sensitivity × local derivative.
- **0:30** Branch from z into its trainable parameters.
- **0:43** The chain rule composes local sensitivities.
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
id: backprop
height: 980
```

## 观察路线

1. 沿第一图从输入读到损失，手算一个中间值。
2. 沿第二图反向读取偏导：上游梯度乘当前局部导数。
3. 把v设为0，区分w梯度为零与v梯度可能非零。
4. 看差分步长曲线；先减小h改善截断误差，再观察浮点误差底部。

## 公式与适用条件

$$
z=wx+b,\ h=\tanh z,\ \hat y=vh,\ L=\tfrac12(\hat y-t)^2,\quad \frac{\partial L}{\partial w}=(\hat y-t)v(1-h^2)x
$$

- 一个标量tanh隐藏单元，平方损失；没有共享参数、多样本平均或随机层。
- 数值梯度不是训练算法，仅作核验；不能依靠单一步长差分证明所有参数处都正确。
- 误差图的对数下限10⁻¹⁶只是显示约定，原始误差可导出。

## 回到知识点

- [[00_Knowledge/链式法则|链式法则]]：沿标量计算图连乘局部导数。
- [[00_Knowledge/多元链式法则|多元链式法则]]：同一损失分别对输入、权重和偏置求导；完整向量反传是雅可比复合。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/activations|激活函数与局部梯度]] · [[07_Programme/可视化实验室/实验/mlp|隐藏单元怎样组合出新形状]] · [[07_Programme/可视化实验室/实验/initialization|初始化如何改变信号与梯度尺度]] · [[07_Programme/可视化实验室/实验/residual|残差捷径怎样改变梯度乘积]]

## 来源与核验

- [Dive into Deep Learning](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html) — §5.3 Forward Propagation, Backward Propagation, and Computational Graphs。支持：计算图缓存前向中间量并按链式法则反向传播；标量例及数值差分由本实验独立推导。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
