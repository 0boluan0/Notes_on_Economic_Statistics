<!-- visual-lab:generated:start -->
# PCA、尺度与重构

<!-- bilingual-en:start -->
*PCA, scaling & reconstruction*
<!-- bilingual-en:end -->

把二维数据压到一条线，连接最大方差、特征向量和重构误差，并检验单位缩放的影响。

> [!question] 先预测，再操作
> 仅把一个变量换单位，第一主成分为什么就可能转向？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/pca.mp4]]

约 33 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 33 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 把点云压到一条线上，会丢掉什么？
- **0:11** 在线上的位置，只需要一个坐标
- **0:17** 保留方差，也是在减少重建误差
- **0:24** 尺度选择，会改变你认为重要的方向

<!-- bilingual-en:start -->
- **0:00** Project the centered cloud onto one direction.
- **0:11** A score is a coordinate along that direction.
- **0:17** Retained variance and reconstruction error are complementary.
- **0:24** Feature scaling changes the PCA problem.
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
id: pca
height: 980
```

## 观察路线

1. 固定种子、scale=1，增加标准差比，预测第一主成分解释率。
2. 只改变 scale，比较协方差 PCA 的方向；再启用标准化，重复这个比较。
3. 选择 ratio=1、scale=1，换几个种子观察方向：总体没有首选方向，但有限样本通常会有轻微差异。
4. 用导出的得分与重构，验证投影残差正交，以及平方重构误差与第二特征值的关系。

## 公式与适用条件

$$
S=\frac{X_c^TX_c}{n-1},\quad Sv_j=\lambda_jv_j,\quad z=X_cv_1,\quad \hat X_c=zv_1^T,\quad \mathrm{EVR}_1=\frac{\lambda_1}{\lambda_1+\lambda_2}
$$

- 先生成独立高斯变量，标准差比为 ratio:1，旋转后再对第二个观测坐标乘 scale。
- 样本协方差以 n−1 为分母；图中方向与指标基于样本，不等于生成角度或总体特征向量。
- 标准化开启时，方差、得分和重构误差针对标准化坐标计算；若回到原坐标展示，应恢复每列标准差与均值。

## 回到知识点

- [[00_Knowledge/主成分分析|主成分分析]]：中心化、主轴、得分和解释方差的整体对应。
- [[00_Knowledge/PCA低秩重构|PCA低秩重构]]：一维投影与丢弃特征值对应的平方误差。
- [[00_Knowledge/PCA尺度选择|PCA尺度选择]]：协方差与相关矩阵 PCA 的度量差异、正单位缩放不变性。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/linear|矩阵怎样改变空间]] · [[07_Programme/可视化实验室/实验/projection|投影、残差与最短距离]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]]

## 来源与核验

- [Cosma Shalizi · CMU Principal Components Analysis](https://stat.cmu.edu/~cshalizi/dm/19/lectures/05/lecture-05.html) — Lecture 5; Finding the principal component; Low-rank approximation; Backup: Gory details。支持：最大投影方差、最小重构误差和特征向量；本实验用样本协方差 n−1 约定。
- [NIST/SEMATECH · Properties of Principal Components](https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc551.htm) — §6.5.5.1; Roadmap to solve the V matrix。支持：标准化数据的协方差对应相关矩阵，特征分解给出方向和解释方差。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
