<!-- visual-lab:generated:start -->
# 池化留下什么，又丢掉什么

<!-- bilingual-en:start -->
*Max/average pooling & local gradients*
<!-- bilingual-en:end -->

跟踪一个实际池化窗口的输出和反向路由，比较最大池化、平均池化与并列最大值。

> [!question] 先预测，再操作
> 池化之后，能从输出恢复窗口中每个像素吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/pooling.mp4]]

约 40 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 40 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 从一个窗口，只留下一个摘要
- **0:07** 最大池化，让最大值留下来
- **0:14** 从输出往回问：哪个输入有影响？
- **0:23** 平均池化，让每个输入分担
- **0:30** 选择规则，决定梯度流向

<!-- bilingual-en:start -->
- **0:00** A pooling window produces one number.
- **0:07** Max pooling selects an input value.
- **0:14** Trace sensitivity back to the selected entry.
- **0:23** Mean pooling shares both value and sensitivity.
- **0:30** The reduction rule determines the backward route.
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
id: pooling
height: 980
```

## 观察路线

1. 观察当前窗口哪些像素决定所选输出。
2. 在Max与Mean之间切换，比较第三图的梯度去向。
3. 选全部相同的矩阵，说明Max此时没有唯一普通梯度。
4. 改变窗口与步幅，解释局部信息与位置信息为何不可逆地丢失。

## 公式与适用条件

$$
y_{ij}=\max_{(r,c)\in G_{ij}}x_{rc}\quad\text{or}\quad y_{ij}=\frac1{|G_{ij}|}\sum_{(r,c)\in G_{ij}}x_{rc}
$$

- 6×6合成矩阵，无填充；输出尺寸按有效窗口计算。
- 第三图只反传当前所选的一个输出，不是所有池化输出梯度的总和。
- 并列最大值时按行优先选第一个位置，展示一个合法次梯度约定。

## 回到知识点

这个主题尚未连接已有知识原子；定义、假设与来源可在本页和实验内阅读。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/convolution|卷积窗口如何逐项生成响应]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]] · [[07_Programme/可视化实验室/实验/residual|残差捷径怎样改变梯度乘积]]

## 来源与核验

- [Dive into Deep Learning](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html) — §7.5.1 Maximum Pooling and Average Pooling。支持：窗口最大/平均聚合与对局部位移的敏感度；不声称任意平移不变。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
