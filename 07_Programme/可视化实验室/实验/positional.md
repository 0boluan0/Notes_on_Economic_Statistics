<!-- visual-lab:generated:start -->
# 位置编码如何表达相对距离

<!-- bilingual-en:start -->
*Sinusoidal positions & relative geometry*
<!-- bilingual-en:end -->

从一对圆上的相位到多频率位置向量，核对固定范数、相对内积及非单调相似度。

> [!question] 先预测，再操作
> 移动两个位置的共同起点，为什么它们的内积可以保持不变？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/positional.mp4]]

约 39 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 39 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一对正弦余弦，把位置变成方向
- **0:08** 位置增加，方向按固定角速度旋转
- **0:15** 多对频率，提供多个位置尺度
- **0:30** 位置编码提供结构，不替模型理解顺序

<!-- bilingual-en:start -->
- **0:00** A sine-cosine pair represents position as an angle.
- **0:08** A position shift adds a fixed angular increment.
- **0:15** Several frequency pairs provide several scales.
- **0:30** The encoding supplies structure for learned use.
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
id: positional
height: 980
```

## 观察路线

1. 先把位置差设为零，核对平方范数和余弦相似度。
2. 固定 Δ，移动 p；第一频率的两根箭头会转动，但内积应保持不变。
3. 把频率对数降为 1，看相似度随距离反复升降。
4. 比较 B=10 与 10000；热图内不同维度的变化速度也应改变。

## 公式与适用条件

$$
\omega_j=B^{-j/m},\quad \mathrm{PE}_{p,2j}=\sin(p\omega_j),\quad\mathrm{PE}_{p,2j+1}=\cos(p\omega_j),\quad \frac{\mathrm{PE}_p^T\mathrm{PE}_{p+\Delta}}m=\frac1m\sum_{j=0}^{m-1}\cos(\Delta\omega_j)
$$

- 维度 d=2m 为偶数，频率按 B^(−j/m) 定义。B=10000 才是原论文使用的底数。
- 热图显示绝对位置 0–31；第二根箭头允许位置 p+Δ 超出热图窗口。
- 圆图只画第一频率；其余频率仍参与整体余弦相似度。

## 回到知识点

- [[00_Knowledge/正弦位置编码|正弦位置编码]]：同一 sin/cos 频率定义；实验进一步核对共同平移下的内积不变与单频周期性。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/embeddings|嵌入向量的角度、长度与相似度]] · [[07_Programme/可视化实验室/实验/attention|注意力从分数到读取向量]] · [[07_Programme/可视化实验室/实验/linear|矩阵怎样改变空间]]

## 来源与核验

- [Vaswani et al. · Attention Is All You Need](https://arxiv.org/html/1706.03762v7) — §3.5 Positional Encoding。支持：固定 sin/cos 频率及位置平移可由线性映射表示；本页从公式另行推导范数和相对内积。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
