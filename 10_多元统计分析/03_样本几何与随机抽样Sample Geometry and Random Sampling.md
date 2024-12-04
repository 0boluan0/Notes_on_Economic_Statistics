
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

## 6. Summary and Applications
- **Importance in Statistical Analysis**
  - Role of covariance and correlation matrices in understanding data relationships.
- **Considerations for High-Dimensional Data**
  - Challenges and strategies for analyzing data when $p > n$.

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

---

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
L_2(d_i) = \sum_{j=1}^n (x_{ji} - \bar{x}_i)^2
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
  \cos(\theta_{ik}) = \frac{d_i' d_k}{L_2(d_i) L_2(d_k)} = \frac{s_{ik}}{\sqrt{s_{ii}s_{kk}}} = r_{ik}
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
S = (s_{ik}),
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
