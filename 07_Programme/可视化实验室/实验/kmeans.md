<!-- visual-lab:generated:start -->
# K-means 的分配与中心更新

<!-- bilingual-en:start -->
*Lloyd iterations & local optima*
<!-- bilingual-en:end -->

逐轮执行实际 Lloyd 算法，显示簇分配、中心移动与簇内平方距离下降。

> [!question] 先预测，再操作
> 目标值每轮下降，能否说明找到了唯一正确的聚类？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/kmeans.mp4]]

约 41 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 41 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先放中心，还没有决定谁属于谁
- **0:04** 先按最近中心分组
- **0:10** 再把中心移到组内平均位置
- **0:17** 先按最近中心分组
- **0:24** 再把中心移到组内平均位置
- **0:32** 两步交替，可能停在不同的局部解

<!-- bilingual-en:start -->
- **0:00** Start with centers before assigning observations.
- **0:04** Assign each observation to its nearest center.
- **0:10** Move each center to its assigned cluster mean.
- **0:17** Assign each observation to its nearest center.
- **0:24** Move each center to its assigned cluster mean.
- **0:32** Alternating improvements need not find a global optimum.
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
id: kmeans
height: 980
```

## 观察路线

1. 从 0 轮开始：先预测每个中心下一步会向哪个方向移动。
2. 逐轮增加 steps，核对中心是所属点的均值，SSE 不增加。
3. 保持数据，切换初始化，比较收敛路径和最终 SSE。
4. 改变 k，解释为什么较小 SSE 不足以单独确定合理簇数。

## 公式与适用条件

$$
J=\sum_i\min_j\|x_i-\mu_j\|^2,\quad c_i=\arg\min_j\|x_i-\mu_j\|^2,\quad\mu_j=\frac1{n_j}\sum_{i:c_i=j}x_i
$$

- 二维 Euclidean 平方距离，簇数由用户指定，均匀样本权重。
- 生成数据来自三个圆形高斯团；算法不读取生成团标签。
- 空簇保留旧中心并记录；spread 是最远点启发式，不是概率加权 k-means++。

## 回到知识点

- [[00_Knowledge/K-means目标|K-means目标]]：真实计算样本到所属中心的平方距离和。
- [[00_Knowledge/Lloyd算法|Lloyd算法]]：控件逐轮执行均值更新和最近中心重分配，显示实际中心轨迹。
- [[00_Knowledge/K-means初始化|K-means初始化]]：固定同一数据比较分散起点与集中起点。
- [[00_Knowledge/Lloyd局部收敛|Lloyd局部收敛]]：SSE 轨迹不增加；不同初始化可以落在不同结果，未宣称全局最优。
- [[00_Knowledge/K-means空簇|K-means空簇]]：独立边界测试触发空簇并验证保留旧中心的明确实现策略；默认画面不保证触发。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/gmm|EM 怎样做软分配]] · [[07_Programme/可视化实验室/实验/pca|PCA、尺度与重构]] · [[07_Programme/可视化实验室/实验/kde|带宽怎样改变密度估计]]

## 来源与核验

- [scikit-learn · K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means) — §2.3.2 K-means; Lloyd algorithm。支持：最近中心分配、均值更新与簇内平方距离目标；初始化影响局部结果。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
