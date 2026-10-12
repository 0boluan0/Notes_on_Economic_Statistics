<!-- visual-lab:generated:start -->
# 投影、残差与最短距离

<!-- bilingual-en:start -->
*Orthogonal projection & residuals*
<!-- bilingual-en:end -->

把一个向量分成沿指定方向的分量和正交残差，建立最小二乘与 PCA 的几何基础。

> [!question] 先预测，再操作
> 沿直线选一个点，为什么垂直落点离原向量最近？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/projection.mp4]]

约 33 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 33 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 在一条直线上，找离向量最近的点
- **0:11** 最短残差，正好垂直于这条线
- **0:18** 把原向量分成两部分
- **0:24** 垂直性，让平方长度可以相加

<!-- bilingual-en:start -->
- **0:00** Find the closest point on a one-dimensional subspace.
- **0:11** At the minimum, the residual is orthogonal.
- **0:18** The vector is projection plus residual.
- **0:24** Orthogonality gives the Pythagorean identity.
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
id: projection
height: 980
```

## 观察路线

1. 固定 v=(2,1)，旋转方向；预测哪个角度让残差最短。
2. 使用垂直预设，解释为什么投影长度为零但原向量不为零。
3. 把 v 改为零，检查所有范数与残差；零向量没有唯一方向。
4. 把投影看成一个一元最小二乘问题，完成第三道挑战。

## 公式与适用条件

$$
u=(\cos\theta,\sin\theta)^T,\quad p=(u^Tv)u,\quad r=v-p,\quad u^Tr=0,\quad \|v\|^2=\|p\|^2+\|r\|^2
$$

- u 由三角函数构造，长度为 1；对一般非单位向量 a，应使用 a(aᵀv)/(aᵀa)。
- 最短距离采用欧氏范数；改变距离定义可能改变最优投影。
- 方向角相差 180° 对应同一条直线，投影不随基向量符号改变。

## 回到知识点

- [[00_Knowledge/正交投影|正交投影]]：子空间分量和正交残差的定义。
- [[00_Knowledge/最佳逼近与正交残差|最佳逼近与正交残差]]：通过移动方向观察最近点，并用勾股分解核验唯一性。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/linear|矩阵怎样改变空间]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]] · [[07_Programme/可视化实验室/实验/pca|PCA、尺度与重构]]

## 来源与核验

- [Georgia Tech · Interactive Linear Algebra](https://textbooks.math.gatech.edu/ila/projections.html) — §6.3.1 Orthogonal Decomposition; Recipe: Orthogonal projection onto a line。支持：正交分解、最近点及投影公式；勾股恒等式由正交性展开。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
