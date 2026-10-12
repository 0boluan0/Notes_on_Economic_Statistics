<!-- visual-lab:generated:start -->
# 一个神经元怎样划分输入空间

<!-- bilingual-en:start -->
*A single affine–sigmoid neuron*
<!-- bilingual-en:end -->

在二维响应面上移动权重、偏置和输入点，观察线性边界和局部敏感度。

> [!question] 先预测，再操作
> Sigmoid让输出非线性，0.5决策边界也会弯曲吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/neuron.mp4]]

约 33 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 33 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 神经元先把贡献加起来
- **0:15** 再把加权和，送进非线性函数
- **0:24** 一个权重控制一条局部影响路径

<!-- bilingual-en:start -->
- **0:00** Inputs make weighted contributions.
- **0:15** The sigmoid turns a score into a probability.
- **0:24** A local derivative follows the weighted path.
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
id: neuron
height: 980
```

## 观察路线

1. 保持偏置不变，改变w₁和w₂，预测边界转向。
2. 只改变偏置，观察同一法向量的平行等值线如何平移。
3. 移动输入点，解释为什么饱和区的局部导数比边界附近小。
4. 令两个权重都为零，检查常数输出，解释为什么不应强行画一条边界。

## 公式与适用条件

$$
z=w_1x_1+w_2x_2+b,\quad p=\sigma(z),\quad p=\tfrac12\iff w^Tx+b=0
$$

- 参数手工指定，没有拟合数据或评估分类准确率。
- 热图显示Sigmoid数值，不宣称它已经是校准后的类别概率。
- 分类规则固定为p≥0.5输出1；边界相等时的标签是约定。

## 回到知识点

这个主题尚未连接已有知识原子；定义、假设与来源可在本页和实验内阅读。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/activations|激活函数与局部梯度]] · [[07_Programme/可视化实验室/实验/mlp|隐藏单元怎样组合出新形状]] · [[07_Programme/可视化实验室/实验/logistic|概率、阈值与交叉熵]]

## 来源与核验

- [Dive into Deep Learning](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) — §5.1.1 Hidden Layers; §5.1.2 Activation Functions。支持：仿射层的组合、非线性激活、ReLU/sigmoid/tanh及其导数；本实验权重为显式构造。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
