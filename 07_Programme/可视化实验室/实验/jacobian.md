<!-- visual-lab:generated:start -->
# 非线性变形怎样被局部Jacobian近似

<!-- bilingual-en:start -->
*Local Jacobian maps & Taylor remainder*
<!-- bilingual-en:end -->

把同一小网格分别送进真实非线性函数和局部仿射映射，再检验随着观察范围缩小的误差。

> [!question] 先预测，再操作
> Jacobian作用在坐标本身，还是作用在相对展开点的扰动上？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/jacobian.mp4]]

约 35 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 35 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 非线性地图，在一点附近像什么？
- **0:14** 小网格的真实弯曲，与线性预测对照
- **0:25** 缩小邻域，才能检验局部近似

<!-- bilingual-en:start -->
- **0:00** A nonlinear map has a local linear action.
- **0:14** Compare the actual curved grid with its tangent grid.
- **0:25** Shrink the neighborhood to test a local approximation.
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
id: jacobian
height: 980
```

## 观察路线

1. 先固定中心，把输入网格与输出网格的同一条线对应起来。
2. 比较真实实线和仿射虚线在中心附近与边缘的误差。
3. 缩小ρ，观察它们靠拢，并查看单位方向误差的二阶上界。
4. 使用det J=0预设，辨别“导数压扁”与“整个非线性图像压扁”。

## 公式与适用条件

$$
F(x,y)=\begin{pmatrix}x+\alpha\sin y\\ y+\beta x^2\end{pmatrix},\quad J(a)=\begin{pmatrix}1&\alpha\cos a_y\\2\beta a_x&1\end{pmatrix},\quad F(a+\Delta)\approx F(a)+J(a)\Delta
$$

- 固定光滑二维函数，无神经网络拟合或数值微分替代真实映射。
- 网格ρ是各坐标半宽；误差图h是沿单位对角方向的欧氏位移长度，二者含义不同。
- 对数图把小于10⁻¹⁶的数显示在下限，原始0保留在导出数据。

## 回到知识点

- [[00_Knowledge/Jacobian矩阵|Jacobian矩阵]]：导数把相对中心的输入扰动映为一阶输出扰动；矩阵行列对应输出/输入。
- [[00_Knowledge/Jacobian行列式|Jacobian行列式]]：局部一阶面积和取向；奇异导数不能单独推断原非线性映射的全局性质。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/linear|矩阵怎样改变空间]] · [[07_Programme/可视化实验室/实验/tangent|割线如何变成切线]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]] · [[07_Programme/可视化实验室/实验/flow|可逆变换如何搬运概率密度]]

## 来源与核验

- [MIT 18.S096 · Matrix Calculus](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/pages/lecture-notes-and-readings/) — Lecture 1 Part 2；Lecture 2 Part 1；Lecture 3 Part 2。支持：导数作为线性映射、Jacobian形状、局部扰动与有限差分；具体映射和二阶上界独立推导。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
