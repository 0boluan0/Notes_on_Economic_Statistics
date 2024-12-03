
# 多变量分析Multivariate analysis

# 多元方法的目标

1. 数据锁紧或结构简化
2. 排序或分组
3.  变量间相关性考察
4. 预测
5. 假设构建和检验

# 数据的组织

## data 

对多个变量或特征进行的测量

## array

数组：每当调查人员寻求理解社会或物理现象时，选择数量 p ≥ 1 的变量或字符进行记录，就会出现多变量。

# 描述性统计Descriptive Statistics

## 样本均值 

 计算过程略

##  样本方差

对于样本 ${x_1, x_2, \dots, x_n}$，样本方差的计算公式为：

$$

S^2_x = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2

$$

其中：

• $n$ 是样本数。

• $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$ 是样本均值。

• $S^2_x$ 是样本方差。
## 样本协方差

对于样本 ${(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)}$，样本协方差的计算公式为：

$$

S_{xy} = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})

$$

其中：

• $\bar{x}$ 和 $\bar{y}$ 分别是 $x$ 和 $y$ 的样本均值。
• $S_{xy}$ 是样本协方差。

## 样本相关系数

$$

r_{xy} = \frac{S_{xy}}{S_x S_y}

$$

其中：

• $S_{xy}$ 是样本协方差。
• $S_x$ 和 $S_y$ 分别是 $x$ 和 $y$ 的样本标准差，$S_x = \sqrt{S^2_x}, S_y = \sqrt{S^2_y}$。

# 基本描述统计矩阵

## 样本均值向量

形状略

## 样本方差和协方差矩阵

假设数据的形状为:
$$

X =

\begin{bmatrix}

x_{11} & x_{12} & \cdots & x_{1p} \\

x_{21} & x_{22} & \cdots & x_{2p} \\

\vdots & \vdots & \ddots & \vdots \\

x_{n1} & x_{n2} & \cdots & x_{np} \\   

\end{bmatrix}

$$

其中，$x_{ij}$ 表示第 $i$ 个样本的第 $j$ 个变量值。

协方差矩阵 $\Sigma$ 的定义为：
$$

\Sigma = \frac{1}{n-1} X^\top X

$$

具体为：

$$

\Sigma =

\begin{bmatrix}

S^2_{x_1} & S_{x_1 x_2} & \cdots & S_{x_1 x_p} \\

S_{x_2 x_1} & S^2_{x_2} & \cdots & S_{x_2 x_p} \\ 

\vdots & \vdots & \ddots & \vdots \\

S_{x_p x_1} & S_{x_p x_2} & \cdots & S^2_{x_p}

\end{bmatrix}

$$

• $S^2_{x_i}$ 是变量 $x_i$ 的样本方差。

• $S_{x_i x_j}$ 是变量 $x_i$ 和 $x_j$ 的样本协方差。

## 样本相关系数矩阵

相关系数矩阵 $R$ 的元素为：

$$

R_{ij} = \frac{S_{x_i x_j}}{\sqrt{S^2_{x_i} S^2_{x_j}}}

$$

• $R_{ij}$ 是变量 $x_i$ 和 $x_j$ 的样本相关系数，取值范围为 $[-1, 1]$。

• 如果 $i = j$，则 $R_{ij} = 1$（变量与自身完全正相关）。

相关系数矩阵的结构为：

$$

R =

\begin{bmatrix}

1 & r_{12} & \cdots & r_{1p} \\
r_{21} & 1 & \cdots & r_{2p} \\

\vdots & \vdots & \ddots & \vdots \\

r_{p1} & r_{p2} & \cdots & 1

\end{bmatrix}

$$

# 图形技术

## 散点图

## 箱线图

## 直方图

## 折线图

# 距离Distance



