<!-- visual-lab:generated:start -->
# 残差捷径怎样改变梯度乘积

<!-- bilingual-en:start -->
*Residual connections & Jacobian products*
<!-- bilingual-en:end -->

对比同一标量变换的普通复合与恒等残差复合，显示逐层输出和精确输入雅可比。

> [!question] 先预测，再操作
> 每层多一个“1”的捷径，是否就永远不会梯度消失？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/residual.mp4]]

约 39 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 39 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一条捷径，怎样改变每一层？
- **0:10** 梯度也沿两条路相加
- **0:16** 多层以后，仍要看实际的连乘
- **0:29** 加上恒等路径，也不保证稳定

<!-- bilingual-en:start -->
- **0:00** An identity path changes the layer, not just the diagram.
- **0:10** The two derivative paths meet at the addition.
- **0:16** Depth still multiplies the local factors.
- **0:29** Identity helps a path; it does not bound every product.
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
id: residual
height: 980
```

## 观察路线

1. 先看单层公式，分别写出普通与残差的局部导数。
2. 令a=0，预测两种链的输出和梯度，再操作确认。
3. 使用linear、a=−1检查捷径完全抵消。
4. 增加层数，比较输出幅度和输入敏感度，注意两幅图使用不同的对数变换。

## 公式与适用条件

$$
h_{\ell+1}=a\phi(h_\ell)\ \text{or}\ h_\ell+a\phi(h_\ell),\quad J_{\ell+1}=(a\phi'(h_\ell))J_\ell\ \text{or}\ (1+a\phi'(h_\ell))J_\ell
$$

- 标量状态、各层相同权重、无归一化、无训练；不是完整ResNet性能模拟。
- tanh分支和linear分支分别按当前中间值计算导数。
- 输出采用signed log₁₀(1+|h|)，梯度图采用log₁₀绝对值；符号及精确零保留在原始数据。

## 回到知识点

- [[00_Knowledge/残差连接|残差连接]]：恒等捷径与同形分支逐元素相加；标量实验展示局部导数1+F′。
- [[00_Knowledge/多元链式法则|多元链式法则]]：多层复合的雅可比乘积，包含抵消和放大边界。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]] · [[07_Programme/可视化实验室/实验/initialization|初始化如何改变信号与梯度尺度]] · [[07_Programme/可视化实验室/实验/normalization|BatchNorm与LayerNorm沿哪个轴统计]] · [[07_Programme/可视化实验室/实验/attention|注意力从分数到读取向量]]

## 来源与核验

- [He et al. · Deep Residual Learning (CVPR 2016)](https://arxiv.org/pdf/1512.03385) — §3.1 Residual Learning; §3.2 Identity Mapping by Shortcuts; Eq. (1)。支持：恒等捷径与残差分支相加；仅展示相加子模块。原论文完整残差块还包含其他层和相加后激活，标量迭代及Jacobian边界为本实验推导。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
