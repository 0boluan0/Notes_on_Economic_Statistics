<!-- visual-lab:generated:start -->
# 决策树怎样划分空间

<!-- bilingual-en:start -->
*Greedy CART partitions*
<!-- bilingual-en:end -->

真实搜索 Gini 最优划分，观察轴对齐区域、叶内概率，以及局部贪心为何会停住。

> [!question] 先预测，再操作
> 最大深度允许很大，为什么树仍可能只有一个叶节点？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/tree.mp4]]

约 41 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 41 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一棵树，先问一个能分开样本的问题
- **0:04** 根节点先切一刀
- **0:10** 只在自己的子区域内继续切分
- **0:16** 只在自己的子区域内继续切分
- **0:23** 到达叶子，就使用叶子内的频率
- **0:31** 路径上的问题，共同定义一个区域

<!-- bilingual-en:start -->
- **0:00** A tree begins with one axis-aligned question.
- **0:04** Split the root.
- **0:10** A child splits only its own region.
- **0:16** A child splits only its own region.
- **0:23** A leaf predicts using the observations that reached it.
- **0:31** A decision path defines an intersection of regions.
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
id: tree
height: 980
```

## 观察路线

1. 先看根节点候选增益图，预测算法会选哪一轴、哪一个阈值。
2. 增加深度，观察新增边界只切割相应父节点区域。
3. 固定深度增加 minleaf，说明哪些候选不再可行。
4. 用挑战中的完全对称 XOR 例子解释“有复杂规则”与“第一步有正增益”的区别。

## 公式与适用条件

$$
G(Q)=2p_Q(1-p_Q),\quad\Delta G=G(Q)-\frac{n_L}{n}G(Q_L)-\frac{n_R}{n}G(Q_R)
$$

- 默认生成规则 x₁>0.2 且 x₂>−0.4；可切换随机 XOR。二类 Gini、二叉轴对齐划分；只接受严格正增益。
- 候选阈值取排序后相邻不同值的中点，两侧均须满足 minleaf。
- 严格对称 XOR 按四个对称点成组，实际数量 4⌊n/4⌋，此模式固定无噪声；每次轴划分均保持两类等比例，根增益为零。没有剪枝或独立验证。

## 回到知识点

这个主题尚未连接已有知识原子；定义、假设与来源可在本页和实验内阅读。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/knn|最近邻如何决定分类]] · [[07_Programme/可视化实验室/实验/boosting|提升算法怎样逐轮修正残差]] · [[07_Programme/可视化实验室/实验/ensemble|平均集成为何受相关性限制]] · [[07_Programme/可视化实验室/实验/modelselection|验证集怎样被选择过程过拟合]]

## 来源与核验

- [scikit-learn · Decision Trees](https://scikit-learn.org/stable/modules/tree.html) — §1.10.7 Mathematical formulation; §1.10.7.1 Classification criteria。支持：加权节点不纯度、递归贪心划分、Gini 与叶内类别比例。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
