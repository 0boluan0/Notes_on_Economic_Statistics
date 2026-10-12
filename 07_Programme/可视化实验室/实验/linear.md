<!-- visual-lab:generated:start -->
# 矩阵怎样改变空间

<!-- bilingual-en:start -->
*Linear maps & signed area*
<!-- bilingual-en:end -->

观察基向量、网格和单位正方形如何被同一个矩阵变换，读懂行列式的面积与方向。

> [!question] 先预测，再操作
> 面积不变，图形就一定只是旋转吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/linear.mp4]]

约 37 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 37 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先问两个基向量去了哪里
- **0:12** 同样的线性组合，也决定其他点的去向
- **0:19** 一个单位方格，变成一个平行四边形
- **0:27** 矩阵不是一张孤立的数字表

<!-- bilingual-en:start -->
- **0:00** Two basis vectors determine the linear map.
- **0:12** The same coefficients combine the transformed basis.
- **0:19** The unit square becomes a parallelogram.
- **0:27** Columns specify actions on directions.
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
id: linear
height: 980
```

## 观察路线

1. 先看矩阵的两列，预测 e₁ 与 e₂ 各自的终点，再对照图。
2. 比较“剪切”和“旋转 90°”：行列式相同，但长度与角度的变化不同。
3. 使用反射预设，沿单位正方形顶点的固定次序观察方向翻转。
4. 使用秩一预设；尝试找两个不同输入，使它们产生同一输出。

## 公式与适用条件

$$
A=\begin{pmatrix}a&b\\c&d\end{pmatrix},\quad Ae_1=(a,c)^T,\quad Ae_2=(b,d)^T,\quad \det A=ad-bc,\quad \mathrm{Area}=|\det A|
$$

- 这是过原点的二维线性变换，不含平移。
- 面积结论用于二维面积；行列式为零不表示所有向量都被映到零。
- 几何图使用相同的坐标尺度；面积由行列式计算，不靠屏幕像素估计。

## 回到知识点

- [[00_Knowledge/行列式体积与取向|行列式体积与取向]]：面积倍率、取向翻转与奇异压缩；剪切预设检验面积保持不等于长度保持。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/projection|投影、残差与最短距离]] · [[07_Programme/可视化实验室/实验/pca|PCA、尺度与重构]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]]

## 来源与核验

- [Georgia Tech · Interactive Linear Algebra](https://textbooks.math.gatech.edu/ila/determinants-volumes.html) — §4.3 Determinants and Volumes。支持：行列式绝对值给出面积缩放；二维公式的符号记录有向面积。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
