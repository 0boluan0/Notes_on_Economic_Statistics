<!-- visual-lab:generated:start -->
# 反向传播把局部导数串起来

<!-- bilingual-en:start -->
*Backpropagation & finite-difference checking*
<!-- bilingual-en:end -->

追踪一个权重的小改动，观察它如何经过曲线影响损失，再沿同一路径理解反向传播为什么逐段相乘。

> [!question] 先预测，再操作
> 为什么隐藏单元很活跃，却可能收不到损失梯度？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/backprop.mp4]]

约 312 秒，中文合成讲解与中文字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 312 seconds, with Chinese synthetic narration and Chinese captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 前向：计算一个结果
- **1:14** 干预：跟着一个小改动走
- **2:25** 反向：逐段累计影响
- **3:47** 更新：改变参数后重新算
- **4:05** 边界：饱和与零权重
- **4:53** 回顾：自己解释一个改动

<!-- bilingual-en:start -->
- **0:00** Forward: compute a result
- **1:14** Intervention: trace a small change
- **2:25** Backward: accumulate sensitivity
- **3:47** Update, then recompute
- **4:05** Saturation and a zero output weight
- **4:53** Recap: explain a change yourself
<!-- bilingual-en:end -->

## 亲手实验

<!-- bilingual-en:start -->
*Try it yourself*
<!-- bilingual-en:end -->

先播放“前向计算”，再进入“追踪一个改动”：灰点留在原处，蓝点沿曲线移动。接着切到反向，观察前向值保持、敏感度逐段相乘。可暂停、拖动进度或选择一个局部运算。影片与交互共用计算与画面；“从影片继续”可载入片中情境。

<!-- bilingual-en:start -->
Play the forward pass, then trace a small weight change: the gray point stays at its original position while the blue point moves along the curve. Switch to the backward pass to see fixed forward values and accumulated sensitivities. Pause, scrub or inspect a local operation. The film and interaction share the same calculation and scene.
<!-- bilingual-en:end -->

<!-- visual-lab:film:end -->
```visual-lab
id: backprop
height: 980
```

## 观察路线

1. 先播放前向，观察这一步的输出怎样成为下一步的输入。
2. 进入“追踪一个改动”，只改变 w：灰点保持，蓝点沿局部曲线移动；解释在哪一段放大、缩小或翻转方向。
3. 进入反向，保持前向值，解释收到的敏感度为什么要乘当前局部斜率。
4. 用相同改动比较标准情境、激活饱和与 v=0，再缩小改变量核对局部预测。
5. 最后执行更新；需要逐项推导、分支求和或矩阵形式时，展开下方核算区。

## 公式与适用条件

$$
z=wx+b,\ h=\tanh z,\ \hat y=vh,\ L=\tfrac12(\hat y-t)^2,\quad \frac{\partial L}{\partial w}=(\hat y-t)v(1-h^2)x
$$

- 主实验为一个样本、一个标量 tanh 隐藏单元与平方损失；扩展区另列共享隐藏值的双路径例子和向量形式。
- 数值梯度不是训练算法，仅作核验；不能依靠单一步长差分证明所有参数处都正确。
- 误差图的对数下限10⁻¹⁶只是显示约定，原始误差可导出。

## 回到知识点

- [[00_Knowledge/链式法则|链式法则]]：沿标量计算图连乘局部导数。
- [[00_Knowledge/多元链式法则|多元链式法则]]：同一损失分别对输入、权重和偏置求导；完整向量反传是雅可比复合。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/activations|激活函数与局部梯度]] · [[07_Programme/可视化实验室/实验/mlp|隐藏单元怎样组合出新形状]] · [[07_Programme/可视化实验室/实验/initialization|初始化如何改变信号与梯度尺度]] · [[07_Programme/可视化实验室/实验/residual|残差捷径怎样改变梯度乘积]]

## 来源与核验

- [Dive into Deep Learning](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html) — §5.3 Forward Propagation, Backward Propagation, and Computational Graphs。支持：计算图缓存前向中间量并按链式法则反向传播；标量例及数值差分由本实验独立推导。
- [Stanford CS231n](https://cs231n.github.io/optimization-2/) — Patterns in backward flow; Gradients for vectorized operations。支持：局部梯度的乘法、共享变量沿多条路径的梯度累加，以及向量化的维度核对。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
