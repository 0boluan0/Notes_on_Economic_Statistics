<!-- visual-lab:generated:start -->
# 注意力从分数到读取向量

<!-- bilingual-en:start -->
*Scaled dot-product attention & causal masking*
<!-- bilingual-en:end -->

查看四个手工token的QK分数、逐行权重与Value向量，分清匹配、可见性和实际读取内容。

> [!question] 先预测，再操作
> 注意力权重不变时，输出还能改变吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/attention.mp4]]

约 65 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 65 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一个 Query，先逐个比较 Key
- **0:13** 把分数变成共享总量的权重
- **0:26** 每个权重，都作用在一个 Value 上
- **0:42** 第 3 个 Query，不能读第 4 个位置
- **0:56** 可见范围改变，加权读取也随之改变

<!-- bilingual-en:start -->
- **0:00** A query scores each key before reading any value.
- **0:13** Normalize across the same query row.
- **0:26** The same weights mix the value vectors.
- **0:42** Mask future scores before normalization.
- **0:56** Visibility changes the weighted read.
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
id: attention
height: 980
```

## 观察路线

1. 先确认热图行代表Query、列代表Key/Value，再选择一行读取。
2. 打开因果掩码，核对被屏蔽位置权重为0、允许位置重新归一化。
3. 把强度设为0，预测每行在允许集合上的均匀权重。
4. 仅改变Value尺度，验证分数与权重不变，而二维输出按比例变化。

## 公式与适用条件

$$
S=\frac{\alpha QK^T}{T\sqrt2},\quad A_{ij}=\frac{\mathbf1[j\in J_i]e^{S_{ij}}}{\sum_{k\in J_i}e^{S_{ik}}},\quad O=AV
$$

- 固定四个二维手工向量，单头且没有训练、输出投影、残差或语言语义。
- 因果掩码保留对角线，每行至少一个可见Key。
- 原始缩放点积公式对应T=1；本实验额外引入温度和分数强度供观察。

## 回到知识点

- [[00_Knowledge/缩放点积注意力|缩放点积注意力]]：QKᵀ/√d、沿Key维归一化及Value加权读取。
- [[00_Knowledge/因果注意力掩码|因果注意力掩码]]：按查询行保留自身和过去位置，屏蔽未来位置。
- [[00_Knowledge/注意力掩码归一化|注意力掩码归一化]]：先确定可见Key集合，再在该集合做Softmax；每行集合非空。
- [[00_Knowledge/注意力权重解释边界|注意力权重解释边界]]：改变Value尺度不改变权重，权重不能独立说明输出贡献。
- [[00_Knowledge/Softmax|Softmax]]：平移稳定的指数归一化与温度控制；输出总和1不是校准证明。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/normalization|BatchNorm与LayerNorm沿哪个轴统计]] · [[07_Programme/可视化实验室/实验/residual|残差捷径怎样改变梯度乘积]] · [[07_Programme/可视化实验室/实验/positional|位置编码如何表达相对距离]] · [[07_Programme/可视化实验室/实验/embeddings|嵌入向量的角度、长度与相似度]]

## 来源与核验

- [Vaswani et al. · Attention Is All You Need (NeurIPS 2017)](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — §3.2.1 Eq. (1); §3.2.3 decoder masking。支持：QKᵀ/√dₖ、对key轴softmax、Value加权；解码器屏蔽未来位置。温度是本实验显式扩展参数。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
