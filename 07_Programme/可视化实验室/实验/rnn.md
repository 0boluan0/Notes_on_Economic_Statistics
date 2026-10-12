<!-- visual-lab:generated:start -->
# RNN怎样在时间中保留状态与梯度

<!-- bilingual-en:start -->
*Scalar recurrence & full backpropagation through time*
<!-- bilingual-en:end -->

实际展开一个共享参数的标量递归，查看时间路径乘积以及共享权重的梯度贡献之和。

> [!question] 先预测，再操作
> 同一个权重在16步中重复出现，它的梯度为什么不能只看最后一步？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/rnn.mp4]]

约 45 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 45 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 把同一个单元，沿时间展开
- **0:17** 旧输入的影响，要穿过后续每一步
- **0:23** 共享参数，收到的是各次使用的总贡献
- **0:35** 前向递推，反向连乘，参数贡献相加

<!-- bilingual-en:start -->
- **0:00** One recurrent cell is reused over time.
- **0:17** Sensitivity travels through every intervening step.
- **0:23** A shared parameter receives a sum across its uses.
- **0:35** Recurrence, products along paths, sums into shared parameters.
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
id: rnn
height: 980
```

## 观察路线

1. 先逐步播放脉冲后的隐藏状态，区分有输入的首步与后续递归。
2. 从第二图的最右端向左读，预测早期状态对末态还有多大影响。
3. 切换线性激活，以a的幂核对梯度。
4. 观察第三图各步贡献的符号，确认指标等于这些贡献的和。

## 公式与适用条件

$$
h_t=\phi(ah_{t-1}+u x_t),\quad L=\tfrac12(h_T-y)^2,\quad \frac{\partial L}{\partial a}=\sum_{t=1}^T\frac{\partial L}{\partial h_t}\phi'(z_t)h_{t-1}
$$

- 单个标量状态、无偏置；权重固定，没有训练。
- 只有末时刻平方损失，完整反传全部T步；未采用截断BPTT或梯度裁剪。
- 敏感度对数图显示绝对值；符号必须结合数据和第三图读取。

## 回到知识点

- [[00_Knowledge/多元链式法则|多元链式法则]]：共享参数的多个时间路径求导后求和；状态敏感度由复合导数给出。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/lstm|LSTM记忆路径不等于完整梯度]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]] · [[07_Programme/可视化实验室/实验/residual|残差捷径怎样改变梯度乘积]] · [[07_Programme/可视化实验室/实验/initialization|初始化如何改变信号与梯度尺度]]

## 来源与核验

- [Dive into Deep Learning](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html) — §9.4.2 Recurrent Networks with Hidden States。支持：共享参数、隐藏状态递归；本页为手工指定的标量特例。
- [Dive into Deep Learning](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html) — §9.7.1，式(9.7.7)；§9.7.2。支持：时间路径梯度求和、完整反传与截断的区别。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
