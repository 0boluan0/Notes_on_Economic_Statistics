<!-- visual-lab:generated:start -->
# 图神经网络的邻居聚合

<!-- bilingual-en:start -->
*Graph aggregation & repeated propagation*
<!-- bilingual-en:end -->

在链和星形图上逐层传播一个节点脉冲，比较平均归一化、对称归一化、自环与残差。

> [!question] 先预测，再操作
> 反复聚合会让所有节点变成同一个数吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/gnn.mp4]]

约 36 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 36 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一层消息传递，只跨一条边
- **0:04** 第 1 层：邻居先发送，再一起汇总
- **0:11** 第 2 层：邻居先发送，再一起汇总
- **0:18** 第 3 层：邻居先发送，再一起汇总
- **0:26** 传播范围，随层数逐步扩大

<!-- bilingual-en:start -->
- **0:00** One layer carries information across one edge.
- **0:04** Layer 1: send, then aggregate.
- **0:11** Layer 2: send, then aggregate.
- **0:18** Layer 3: send, then aggregate.
- **0:26** The receptive field grows one hop per layer.
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
id: gnn
height: 980
```

## 观察路线

1. 从层数 0 开始确认只有节点 0 为 1，再逐层增加，观察信息最多沿一条边传播一层。
2. 切换星形图与两种归一化，比较中心节点和叶节点。
3. 去掉自环、设残差为零，交替查看偶数层和奇数层。
4. 加回自环或残差，再讨论长期行为所需的连通性与非周期性。

## 公式与适用条件

$$
H^{(k+1)}=rH^{(k)}+(1-r)PH^{(k)},\quad P_{\mathrm{mean}}=\widetilde D^{-1}\widetilde A,\quad P_{\mathrm{sym}}=\widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2}
$$

- 六节点无向图，初始 H=(1,0,0,0,0,0)ᵀ；没有孤立节点。
- 固定单位特征变换与恒等激活，没有监督标签和训练过程。
- r∈[0,0.9]；自环在计算度数前加入。

## 回到知识点

- [[00_Knowledge/邻接矩阵幂计数游走|邻接矩阵幂计数游走]]：矩阵乘法沿图逐层传播的组合基础；实验加入度归一化后权重不再是原始游走条数。
- [[00_Knowledge/Markov矩阵幂收敛判据|Markov矩阵幂收敛判据]]：邻居平均为行随机矩阵；无自环二部图的奇偶振荡说明单位圆谱边界不可省略。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/linear|矩阵怎样改变空间]] · [[07_Programme/可视化实验室/实验/pca|PCA、尺度与重构]] · [[07_Programme/可视化实验室/实验/attention|注意力从分数到读取向量]]

## 来源与核验

- [Kipf & Welling · Semi-Supervised Classification with GCNs](https://arxiv.org/pdf/1609.02907) — §2, Eq. (2); §2.2。支持：加自环与对称度归一化的层传播；本页明确取单位权重、恒等激活，另作邻居平均和残差比较。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
