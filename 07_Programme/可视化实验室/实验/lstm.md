<!-- visual-lab:generated:start -->
# LSTM记忆路径不等于完整梯度

<!-- bilingual-en:start -->
*Gated cell memory & total derivatives*
<!-- bilingual-en:end -->

固定三个门值，实际递归细胞和隐藏状态；再打开候选反馈，比较直通路径与完整导数。

> [!question] 先预测，再操作
> 遗忘门接近1，是否就能断言完整梯度等于1？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/lstm.mp4]]

约 47 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 47 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 记忆状态，和可见输出，是两条线
- **0:17** 输出门控制读出，不直接改写记忆
- **0:24** 只沿记忆直通路径，梯度怎么走？
- **0:37** 直通路径的导数，不总是完整导数

<!-- bilingual-en:start -->
- **0:00** Cell memory and hidden output are distinct states.
- **0:17** The output gate controls exposure of the cell.
- **0:24** Follow the direct cell-state shortcut.
- **0:37** The direct-path derivative need not be the full derivative.
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
id: lstm
height: 980
```

## 观察路线

1. 默认全零输入且r=0，核对c_t=f^t c₀。
2. 将f设1且i设0，同时比较细胞状态、隐藏输出和两种导数。
3. 打开r，使过去的h能经候选值回流到c，观察两条细胞梯度分离。
4. 将o设0，说明“没有对外输出”与“内部没有记忆”不同。

## 公式与适用条件

$$
g_t=\tanh(x_t+r h_{t-1}),\quad c_t=f c_{t-1}+i g_t,\quad h_t=o\tanh(c_t),\quad \left.\frac{\partial c_T}{\partial c_0}\right|_{\text{direct path}}=f^T
$$

- 标量、三门固定、无训练；门0与1为Sigmoid的理想极限值。
- h₀定义为o tanh(c₀)，对c₀求导时包含这一步。
- 图2使用保符号log₁₀(1+绝对值)，不是原始梯度刻度。

## 回到知识点

- [[00_Knowledge/链式法则|链式法则]]：细胞到隐藏输出经过tanh和输出门，保存细胞不等于输出梯度为1。
- [[00_Knowledge/多元链式法则|多元链式法则]]：候选反馈增加依赖路径，完整导数不能只保留细胞直通路径。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/rnn|RNN怎样在时间中保留状态与梯度]] · [[07_Programme/可视化实验室/实验/residual|残差捷径怎样改变梯度乘积]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]]

## 来源与核验

- [Dive into Deep Learning](https://d2l.ai/chapter_recurrent-modern/lstm.html) — §10.1.1.2–10.1.1.5，式(10.1.1)–(10.1.4)。支持：遗忘/输入/输出门、候选、细胞与隐藏状态；本页固定门值以分开检验路径和完整导数。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
