<!-- visual-lab:generated:start -->
# 卷积窗口如何逐项生成响应

<!-- bilingual-en:start -->
*Spatial cross-correlation, padding & stride*
<!-- bilingual-en:end -->

同步查看补零输入、核、输出和选定窗口的九项乘积，让每个响应都可手算。

> [!question] 先预测，再操作
> 核里有负数，为什么一条边缘的响应方向会改变？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/convolution.mp4]]

约 61 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 61 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一个输出格，是怎样算出来的？
- **0:10** 相同位置相乘，九项再相加
- **0:33** 把这个和，放回它的输出位置
- **0:51** 现在再读卷积的求和式

<!-- bilingual-en:start -->
- **0:00** Follow one receptive field into one output.
- **0:10** Multiply corresponding entries, then add.
- **0:33** The sum belongs to one output position.
- **0:51** The indices now refer to visible objects.
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
id: convolution
height: 980
```

## 观察路线

1. 先看框出的输入窗口，再把对应位置与核逐项相乘。
2. 将第四图九个数相加，核对当前输出指标。
3. 固定核和图案，只改变步幅与填充，手算新的输出尺寸。
4. 使用非对称左右差分核，区分互相关与真正翻转核后的数学卷积。

## 公式与适用条件

$$
Y_{ij}=\sum_{r=0}^{2}\sum_{c=0}^{2}K_{rc}\,X^{\mathrm{pad}}_{is+r,js+c},\quad n_{out}=\left\lfloor\frac{7+2p-3}{s}\right\rfloor+1
$$

- 合成7×7单通道图案，单个固定3×3核；没有训练或多通道求和。
- 补零是真实参与计算的边界约定，边缘处可引入与内部不同的响应。
- 检查行列限制0到2，保证所有可选输出形状下该位置有效。

## 回到知识点

这个主题尚未连接已有知识原子；定义、假设与来源可在本页和实验内阅读。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/pooling|池化留下什么，又丢掉什么]] · [[07_Programme/可视化实验室/实验/linear|矩阵怎样改变空间]] · [[07_Programme/可视化实验室/实验/residual|残差捷径怎样改变梯度乘积]]

## 来源与核验

- [Dive into Deep Learning](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html) — §7.2.1 The Cross-Correlation Operation; §7.2.6 Convolution。支持：核与窗口逐项乘加；深度学习通常实现互相关，严格卷积需反转核。
- [Dive into Deep Learning](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html) — §7.3.1 Padding; §7.3.2 Stride。支持：填充、步幅和输出空间尺寸；本实验只含单通道3×3核。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
