<!-- visual-lab:generated:start -->
# 隐藏单元怎样组合出新形状

<!-- bilingual-en:start -->
*Hidden units & piecewise-linear expression*
<!-- bilingual-en:end -->

展开手工构造的单隐藏层网络，分别查看隐藏激活、带符号贡献和最终输出。

> [!question] 先预测，再操作
> 只叠加线性层，能不能得到这些折线拐点？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/mlp.mp4]]

约 44 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 44 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先做出几个有转折点的部件
- **0:13** 同一个 x，先读取每个部件的贡献
- **0:24** 对每个 x 重复，得到整条输出曲线
- **0:34** 宽度增加的是可组合的部件

<!-- bilingual-en:start -->
- **0:00** Hidden units create reusable pieces.
- **0:13** Read every contribution at the same input.
- **0:24** Pointwise addition creates the full function.
- **0:34** More hidden units provide more pieces to combine.
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
id: mlp
height: 980
```

## 观察路线

1. 从3个ReLU开始，读出三个偏置位置和输出权重1、−2、1。
2. 在x=0处分别计算三个隐藏激活，再用带符号权重相加。
3. 把隐藏激活改为linear；解释没有非线性时两层为何仍只是仿射映射。
4. 增加宽度并移动全部拐点，区分表达式变化和训练获得的能力。

## 公式与适用条件

$$
h_j(x)=\phi(g(x-t_j)),\quad y(x)=b_o+\sum_{j=1}^{m}c_jh_j(x)
$$

- 网络所有权重显式指定，没有优化器、训练数据或泛化评估。
- 增加宽度时拐点和输出系数按已声明构造规则变化，不能当成同一数据上的模型选择实验。
- 本演示是单输入、单隐藏层、单输出；没有声称已复现完整深层网络。

## 回到知识点

- [[00_Knowledge/逐位置前馈网络|逐位置前馈网络]]：两层仿射变换夹非线性激活的组合机制；本实验为一维构造，不演示token间作用。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/activations|激活函数与局部梯度]] · [[07_Programme/可视化实验室/实验/neuron|一个神经元怎样划分输入空间]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]] · [[07_Programme/可视化实验室/实验/overfit|模型复杂度与泛化]]

## 来源与核验

- [Dive into Deep Learning](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) — §5.1.1 Hidden Layers; §5.1.2 Activation Functions。支持：仿射层的组合、非线性激活、ReLU/sigmoid/tanh及其导数；本实验权重为显式构造。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
