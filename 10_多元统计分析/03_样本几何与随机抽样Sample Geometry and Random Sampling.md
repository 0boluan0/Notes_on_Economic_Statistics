
# Chapter 3: Sample Geometry and Random Sampling

## 1. The Geometry of the Sample
- **Data Matrix Representation**
  - Structure and notation of the data matrix.
- **Mean-Corrected Vector and Deviation Vector**
  - Definitions and mathematical representations.
- **Angles Between Deviation Vectors**
  - Relationship between cosine of angles and correlation coefficients.

## 2. Random Sample and the Expected Values of the Sample Mean and Covariance Matrix
- **Definition of the Sample Covariance Matrix**
  - Formula and its components.
- **Generalized Variance**
  - Interpretation using determinant of the covariance matrix.
  - Trace of the covariance matrix as a measure of total variability.

## 3. Geometric Interpretation of Generalized Variance
- **Area and Volume**
  - Calculation of areas and volumes in low-dimensional and high-dimensional spaces.
- **High-Dimensional Covariance Characteristics**
  - Implications when the sample size ($n$) is less than the number of variables ($p$).

## 4. Matrix Representation of the Sample Covariance Matrix
- **Matrix Representation**
  - Deviation matrix and its role in computing the covariance matrix.
- **Determinant and Eigenvalues**
  - Use of eigenvalues to interpret data variability in $p$-space.
- **Rank-Deficient Cases**
  - Handling cases where the covariance matrix determinant is zero.

## 5. Standardized Covariance Matrix and Correlation Matrix
- **Standardization**
  - Transforming variables to obtain the correlation matrix.
- **Correlation Matrix Representation**
  - Mathematical formulation of the sample correlation matrix.
- **Relationship Between Covariance and Correlation Matrices**
  - Conversion formulas using diagonal scaling matrices.

-----

# 1. The Geometry of the Sample 样本的几何结构

## 1.1 Data Matrix Representation 数据矩阵的表示

数据矩阵（Data Matrix）$X_{n \times p}$ 用于表示 $n$ 个观测样本在 $p$ 个变量上的取值，其结构为：
$$
X_{n \times p} =
\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}
$$
其中每一行表示一个样本（Sample），每一列表示一个变量（Variable）。通过数据矩阵，我们可以对多变量样本进行统一表示。

## 1.2 Mean-Corrected Vector and Deviation Vector 均值修正向量与偏差向量

- 对于第 $i$ 个变量，其观测值组成向量 $y_i$，表示为：
  $$
  y_i =
  \begin{pmatrix}
  x_{1i} \\
  x_{2i} \\
  \vdots \\
  x_{ni}
  \end{pmatrix}
  $$
  其中，$x_{ji}$ 表示第 $j$ 个样本在第 $i$ 个变量上的取值。

- 对 $i$ 变量的均值修正向量（Mean-Corrected Vector），也称为偏差向量（Deviation Vector）$d_i$，定义为：
  $$
  d_i = y_i - \bar{x}_i \mathbf{1} =
  \begin{pmatrix}
  x_{1i} - \bar{x}_i \\
  x_{2i} - \bar{x}_i \\
  \vdots \\
  x_{ni} - \bar{x}_i
  \end{pmatrix}
  $$
  其中 $\bar{x}_i$ 是第 $i$ 个变量的均值，$\mathbf{1}$ 是 $n$ 维单位列向量。

- 均值修正向量的作用是将数据中心化，使每个观测值相对于变量均值的偏差得以量化。

---

## 1.3 Sum of Squared Deviations 偏差平方和

变量的总体变异性可以通过偏差平方和（Sum of Squared Deviations）来表示，其定义为：
$$
L^2(d_i) = \sum_{j=1}^n (x_{ji} - \bar{x}_i)^2
$$
这一指标反映了变量值相对于其均值的总偏离程度，数值越大，说明变量的分散程度越高。

---

## 1.4 Angles Between Deviation Vectors 偏差向量之间的夹角

- 偏差向量 $d_i$ 和 $d_k$ 之间的内积（Inner Product）定义为：
  $$
  d_i' d_k = \sum_{j=1}^n (x_{ji} - \bar{x}_i)(x_{jk} - \bar{x}_k)
  $$
  该内积反映了两个变量的联合偏差特性。

- 偏差向量之间的夹角（Angle Between Vectors）$\theta_{ik}$ 的余弦值（Cosine Value）表示为：
  $$
  \cos(\theta_{ik}) = \frac{d_i' d_k}{L(d_i) L(d_k)} = \frac{s_{ik}}{\sqrt{s_{ii}s_{kk}}} = r_{ik}
  $$
  其中：
  - $s_{ik}$ 是样本协方差（Sample Covariance）。
  - $s_{ii}$ 和 $s_{kk}$ 分别是 $d_i$ 和 $d_k$ 的样本方差（Sample Variance）。
  - $r_{ik}$ 是样本相关系数（Sample Correlation Coefficient）。

- 这一结果的几何意义是：相关系数是偏差向量夹角余弦值的标准化形式。当 $r_{ik}=1$ 时，两个变量完全正相关；当 $r_{ik}=-1$ 时，完全负相关；当 $r_{ik}=0$ 时，变量无关。

---

# 2. Random Sample and the Expected Values of the Sample Mean and Covariance Matrix

随机样本与样本均值和协方差矩阵的期望值

## 2.1 Sample Covariance Matrix 样本协方差矩阵

样本协方差矩阵（Sample Covariance Matrix）用于衡量变量之间的线性关系，其定义为：
$$
S = ((s_{ik})),
$$
其中协方差 $s_{ik}$ 的计算公式为：
$$
s_{ik} = \frac{1}{n - 1} \sum_{j=1}^n (x_{ji} - \bar{x}_i)(x_{jk} - \bar{x}_k) = \frac{1}{n - 1} d_i' d_k,
$$
其中：
- $n$ 是样本数量。
- $\bar{x}_i$ 和 $\bar{x}_k$ 分别是变量 $i$ 和变量 $k$ 的样本均值。
- $d_i$ 和 $d_k$ 是均值修正向量（Mean-Corrected Vectors）。

样本协方差矩阵 $S$ 包含 $p$ 个变量的样本方差（对角线元素）和 $\binom{p}{2} = \frac{p(p - 1)}{2}$ 个不同的样本协方差（非对角线元素）。

==这个做法及其麻烦,横竖看不懂.看下面的这个就行==

$$
S = \frac{1}{n-1} D^T D.
$$

D是Deviation Matrix(偏差矩阵)
		           
---

## 2.2 Generalized Variance 推广方差

为了用一个数值总结协方差矩阵所表达的变异性，可以使用以下两种方法：

### 2.2.1 Determinant of $S$ 矩阵行列式

样本协方差矩阵 $S$ 的行列式被称为**推广样本方差**（Generalized Sample Variance）：
$$
|S| = \text{det}(S).
$$
- 几何意义：当 $p=2$ 时，$|S|$ 是协方差矩阵所表示的椭圆面积的平方；当 $p>2$ 时，$|S|$ 是高维超体积的平方。

### 2.2.2 Trace of $S$ 矩阵迹

另一种方法是使用协方差矩阵的迹（Trace），即对角线元素之和，表示总样本方差（Total Sample Variance）：
$$
\text{Trace}(S) = s_{11} + s_{22} + \cdots + s_{pp}.
$$
- 这一值等于每个变量样本方差的总和。

---

## 2.3 Properties of Generalized Variance 推广方差的性质

- **高维情况**：当样本数量 $n$ 小于或等于变量数量 $p$ 时，$|S| = 0$。此时协方差矩阵的秩不足，称为高维退化（High-Dimensional Degeneracy）。
- **变量选择的影响**：当 $|S|=0$ 时，说明至少一个变量的偏差向量是其他变量偏差向量的线性组合。此时需要移除部分变量以确保矩阵为满秩。

---

## 2.4 Geometric Interpretation of Generalized Variance 推广方差的几何解释

- **面积与体积**：对于二维（$p=2$）情形，推广方差与椭圆面积的平方成正比：
  $$
  \text{Area} = (n-1) \sqrt{s_{11}s_{22}(1 - r_{12}^2)}.
  $$
- 对于高维情形（$p>2$），推广方差与高维空间中偏差向量生成的体积（Volume）的平方成正比：
  $$
  |S| = (\text{Volume})^2 \cdot (n - 1)^{-p}.
  $$

---

# 3. Geometric Interpretation of Generalized Variance推广方差的几何解释

推广方差（Generalized Variance）通过几何视角揭示了样本协方差矩阵中变异性的分布特性，特别是在二维和高维空间中。行列式（Determinant）和矩阵迹（Trace）的几何意义分别对应面积、体积和总体变异性的量化。

---

## 3.1 Generalized Variance in 2D: Elliptical Area in Two Dimensions

### 椭圆面积（二维情形）

当变量数 $p = 2$ 时，样本协方差矩阵 $S$ 的行列式 $|S|$ 表示的几何意义为变量生成的椭圆面积平方：
$$
\text{Area}^2 = |S| \cdot (n - 1)^{-2},
$$
其中：
- $s_{11}$ 和 $s_{22}$ 是两个变量的样本方差，分别对应椭圆的长轴和短轴方向上的伸展程度。
- $r_{12}$ 是变量间的相关系数，反映了椭圆的倾斜角度和变量的线性相关性。

椭圆面积可表示为：
$$
\text{Area} = (n-1)\sqrt{s_{11}s_{22}(1 - r_{12}^2)}.
$$

### 几何解释：

1. **正交变量**：当 $r_{12} = 0$ 时，两个变量独立，椭圆为对称圆，面积最大。
2. **完全线性相关**：当 $|r_{12}| = 1$ 时，两个变量共线，椭圆退化为直线，面积为零。

**示例**：
假设两个变量的协方差矩阵为：
$$
S =
\begin{pmatrix}
4 & 2 \\
2 & 3
\end{pmatrix}.
$$
行列式计算为：
$$
|S| = s_{11}s_{22}(1 - r_{12}^2) = 4 \cdot 3 - 2^2 = 8.
$$
椭圆面积为：
$$
\text{Area} = \sqrt{|S|} = \sqrt{8}.
$$

---

## 3.2 Generalized Variance in Higher Dimensions: Hypervolume

### 超体积（高维情形）

当 $p > 2$ 时，推广方差描述的是偏差向量 $d_1, d_2, \dots, d_p$ 在 $p$ 维空间中生成的高维超体积（Hypervolume）。超体积的平方与样本协方差矩阵的行列式相关：
$$
|S| = (\text{Volume})^2 \cdot (n-1)^{-p}.
$$

### 几何特性：

1. **变量的独立性**：当所有变量完全独立（偏差向量两两正交）时，超体积达到最大值，行列式 $|S|$ 也取最大值。
2. **变量的相关性**：当部分变量之间高度相关或完全相关时，偏差向量共线或共面，导致高维体积减小，甚至为零。
3. **退化情况**：当样本数 $n \leq p$ 时，协方差矩阵的秩不足，高维体积退化为零（$|S| = 0$）。

**示例**：
假设三变量的协方差矩阵为：
$$
S =
\begin{pmatrix}
4 & 1 & 2 \\
1 & 3 & 1 \\
2 & 1 & 5
\end{pmatrix}.
$$
行列式为：
$$
|S| = 4(3 \cdot 5 - 1 \cdot 1) - 1(1 \cdot 5 - 2 \cdot 1) + 2(1 \cdot 1 - 3 \cdot 2) = 60 - 5 - 10 = 45.
$$
高维体积平方为：
$$
\text{Volume}^2 = |S| \cdot (n-1)^{-3}.
$$

---

## 3.3 Trace as Total Variability: An Alternative Measure

矩阵迹（Trace）是另一种量化协方差矩阵总变异性的方法，其定义为对角线元素之和：
$$
\text{Trace}(S) = s_{11} + s_{22} + \cdots + s_{pp}.
$$
- 几何意义：Trace 反映了所有变量在各自方向上方差的总和。
- 特点：与行列式相比，Trace 不考虑变量之间的相关性。

**对比**：
1. 行列式（$|S|$）强调变量的交互关系，通过面积或体积体现相关性。
2. 矩阵迹（Trace）仅反映各变量单独的变异性，不受变量相关性影响。

---

## 3.4 Special Cases and Practical Considerations
### 特殊情形：

1. **行列式为零（$|S| = 0$）**：
   - 表示至少有一个变量是其他变量的线性组合。
   - 在这种情况下，需移除冗余变量使矩阵恢复为满秩。
2. **变量标准化**：
   - 为避免变量量纲的影响，可使用相关矩阵代替协方差矩阵。
   - 相关矩阵的行列式可提供标准化后的推广方差。

#### 实际意义：
推广方差在统计学中有广泛应用，包括主成分分析（PCA）、判别分析（Discriminant Analysis）等。通过行列式或迹值，可以评估变量之间的整体相关性与独立性。

---
# 4. Matrix Representation of the Sample Covariance Matrix

样本协方差矩阵的矩阵表示

样本协方差矩阵不仅可以用基本公式定义，还可以通过矩阵形式统一表示，从而揭示其在统计计算与几何解释中的核心作用。

---

## 4.1 Matrix Representation 矩阵表示

### 偏差矩阵（Deviation Matrix）的定义

假设数据矩阵 $X_{n \times p}$ 表示 $n$ 个样本在 $p$ 个变量上的观测值，其均值矩阵 $X̄$ 表示为：
$$
X̄ = \frac{1}{n} \mathbf{1}_{n \times 1} \cdot \mathbf{1}_{1 \times p} X,
$$
其中 $\mathbf{1}_{n \times 1}$ 是 $n$ 维列向量，$\mathbf{1}_{1 \times p}$ 是 $p$ 维行向量。

偏差矩阵（Deviation Matrix）定义为：
$$
D = X - \mathbf{1}_{n \times 1} X̄.
$$
矩阵 $D$ 的每一列是一个变量的均值修正向量（Mean-Corrected Vector），用于计算变量的偏差。

### 样本协方差矩阵的矩阵形式

样本协方差矩阵 $S$ 的计算公式可以用偏差矩阵表示为：
$$
S = \frac{1}{n-1} D^T D,
$$
其中：
- $D^T$ 是偏差矩阵的转置，表示变量之间的偏差向量。
- $D^T D$ 是偏差向量之间的内积矩阵。

这一表示形式显示了协方差矩阵是变量偏差的二次型（Quadratic Form），反映了变量之间的线性相关性。

---

## 4.2 Determinant and Eigenvalues 行列式与特征值
### 协方差矩阵行列式的几何意义

样本协方差矩阵的行列式（Determinant）$|S|$ 反映了数据在 $p$ 维空间中生成的高维超体积的平方：
$$
|S| = \prod_{i=1}^p \lambda_i,
$$
其中 $\lambda_1, \lambda_2, \dots, \lambda_p$ 是协方差矩阵的特征值（Eigenvalues）。

### 特征值与数据变异性的关系

- 每个特征值 $\lambda_i$ 表示数据在对应特征向量方向上的变异性。
- 特征值的总和等于协方差矩阵的迹（Trace）：
  $$
  \text{Trace}(S) = \sum_{i=1}^p \lambda_i.
  $$
- 特征值的积等于行列式 $|S|$：
  $$
  |S| = \prod_{i=1}^p \lambda_i.
  $$

### 几何直观

- 如果一个特征值为零，说明数据在该方向上没有变异性，变量间存在完全的线性相关。
- 特征值的大小反映了不同方向上的变异性分布。例如，在主成分分析（PCA）中，特征值越大，对应的主成分方向越重要。

---

## 4.3 Rank-Deficient Cases 协方差矩阵秩不足的情况
### 秩不足（Rank Deficiency）的定义

协方差矩阵的秩（Rank）表示变量间独立方向的个数。当矩阵秩不足（Rank-Deficient）时，行列式 $|S| = 0$。这通常发生在以下情况：
1. **样本数小于变量数（$n < p$）**：
   - 当样本数不足以覆盖变量维度时，协方差矩阵无法生成满秩矩阵。
   - 高维数据中常见这种情况（称为 "curse of dimensionality"）。
2. **变量间完全线性相关**：
   - 存在变量可以表示为其他变量的线性组合。

### 处理秩不足的方法

1. **移除冗余变量**：
   - 如果变量间存在完全线性相关性，可以通过主成分分析（PCA）或变量选择方法移除冗余变量。
2. **正则化协方差矩阵**：
   - 在高维情况下，可以对协方差矩阵进行正则化（Regularization），例如加上小的正对角元素：
     $$
     S_{\text{reg}} = S + \epsilon I,
     $$
     其中 $\epsilon$ 是正则化参数，$I$ 是单位矩阵。
3. **使用相关矩阵（Correlation Matrix）替代**：
   - 标准化变量后使用相关矩阵代替协方差矩阵，避免变量量纲差异的影响。

---

## 5. Standardized Covariance Matrix and Correlation Matrix标准化协方差矩阵与相关矩阵

协方差矩阵描述了变量间的线性关系，但其数值受到变量量纲的影响。通过标准化，可以消除量纲差异，使变量间的关系更清晰。

---

## 5.1 Standardization 标准化
### 标准化的定义

为了消除变量的量纲影响，可以对变量进行标准化（Standardization），即将每个变量的均值调整为 $0$，标准差调整为 $1$。对于变量 $X_j$，其标准化形式为：
$$
Z_j = \frac{X_j - \bar{X}_j}{\sqrt{s_{jj}}},
$$
其中：
- $\bar{X}_j$ 是变量 $X_j$ 的均值。
- $s_{jj}$ 是变量 $X_j$ 的样本方差。

标准化后得到的变量称为标准变量（Standardized Variables），其均值为 $0$，方差为 $1$。

### 标准化后协方差矩阵的变化

通过对数据矩阵 $X$ 标准化，生成的协方差矩阵成为相关矩阵（Correlation Matrix），其值范围固定为 $[-1, 1]$。

---

## 5.2 Correlation Matrix Representation 相关矩阵的表示
### 相关矩阵的定义

相关矩阵（Correlation Matrix）$R$ 是标准化后的协方差矩阵，表示变量之间的线性关系。其公式为：
$$
R = \frac{1}{n-1} Z^T Z,
$$
其中 $Z$ 是标准化后的数据矩阵。

相关矩阵的元素 $r_{ij}$ 为变量 $X_i$ 和 $X_j$ 的样本相关系数，计算公式为：
$$
r_{ij} = \frac{s_{ij}}{\sqrt{s_{ii} s_{jj}}},
$$
其中：
- $s_{ij}$ 是 $X_i$ 和 $X_j$ 的协方差。
- $s_{ii}, s_{jj}$ 分别是 $X_i$ 和 $X_j$ 的方差。

### 相关矩阵的性质

1. **对角线元素**：相关矩阵的对角线元素为 $1$，即 $r_{ii} = 1$，因为每个变量与自身完全相关。
2. **对称性**：相关矩阵是对称矩阵，即 $r_{ij} = r_{ji}$。
3. **取值范围**：相关系数 $r_{ij}$ 的取值范围为 $[-1, 1]$。

---

## 5.3 Relationship Between Covariance and Correlation Matrices 协方差矩阵与相关矩阵的关系

### 转换公式

协方差矩阵 $S$ 和相关矩阵 $R$ 通过对角标准差矩阵 $D^{1/2}$ 相互转换：
1. 从协方差矩阵 $S$ 得到相关矩阵 $R$：
   $$
   R = D^{-1/2} S D^{-1/2},
   $$
   其中 $D^{1/2}$ 是协方差矩阵对角线上方差的平方根矩阵：
   $$
   D^{1/2} =
   \begin{pmatrix}
   \sqrt{s_{11}} & 0 & \cdots & 0 \\
   0 & \sqrt{s_{22}} & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & \sqrt{s_{pp}}
   \end{pmatrix}.
   $$

2. 从相关矩阵 $R$ 得到协方差矩阵 $S$：
   $$
   S = D^{1/2} R D^{1/2}.
   $$

### 几何意义

- 协方差矩阵描述了变量的原始关系，其大小取决于变量的量纲。
- 相关矩阵消除了量纲的影响，直接反映变量间的线性相关性。

---
