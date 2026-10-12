<!-- visual-lab:generated:start -->
# 激活函数与局部梯度

<!-- bilingual-en:start -->
*Activation functions & local derivatives*
<!-- bilingual-en:end -->

同时观察激活输出和真正对输入求导的曲线，区分饱和、负半轴与不可导拐点。

> [!question] 先预测，再操作
> 把输入放大，梯度一定也变大吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/activations.mp4]]

约 41 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 41 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一个输入，怎样穿过激活函数？
- **0:18** 相同的输入变化，不一定有相同的输出变化
- **0:32** 激活值和导数，是两个不同的问题

<!-- bilingual-en:start -->
- **0:00** Follow a single input through the nonlinearity.
- **0:18** The slope changes the transmitted perturbation.
- **0:32** The output and its sensitivity answer different questions.
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
id: activations
height: 980
```

## 观察路线

1. 先预测观察点输出的正负与范围，再读指标。
2. 固定x=2，将g从1增大到4，同时比较Sigmoid输出和输入导数。
3. 切到ReLU，把观察点移到z=0，区分数学不可导与框架选定的次梯度。
4. 在负半轴比较ReLU与Leaky ReLU，再移动偏置，观察拐点的位置。

## 公式与适用条件

$$
z=gx+b,\quad y=\phi(z),\quad \frac{dy}{dx}=g\phi'(gx+b)
$$

- 仅显示单个标量激活，没有损失函数、数据训练或多层交互。
- Leaky ReLU负半轴斜率固定0.1；ReLU类在z=0处不具有唯一普通导数。
- 曲线是解析表达式的采样，不能用单层非零导数保证整个深层网络不发生梯度消失。

## 回到知识点

- [[00_Knowledge/链式法则|链式法则]]：仿射输入后接激活，对原输入求导须乘输入增益；普通导数不存在的拐点另行标示。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/neuron|一个神经元怎样划分输入空间]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]] · [[07_Programme/可视化实验室/实验/initialization|初始化如何改变信号与梯度尺度]]

## 来源与核验

- [Dive into Deep Learning](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) — §5.1.1 Hidden Layers; §5.1.2 Activation Functions。支持：仿射层的组合、非线性激活、ReLU/sigmoid/tanh及其导数；本实验权重为显式构造。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
