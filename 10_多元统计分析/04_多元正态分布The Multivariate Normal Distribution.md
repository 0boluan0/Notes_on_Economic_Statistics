
==证明不要求掌握.但是要知道本章提到过的各个性质,会在选择题中出现== 

# Chapter 4: The Multivariate Normal Distribution

## 1. Introduction
- **1.1 Overview**: Importance and applications of the multivariate normal distribution.
- **1.2 Use Cases**: Assumptions and scenarios where the multivariate normal distribution applies.

## 2. The Multivariate Normal Density and Its Properties
- **2.1 Univariate Normal Distribution**: Definition and probability density function.
- **2.2 Multivariate Normal Distribution**:
  - Density function formula and interpretation.
  - Special case: Bivariate normal distribution.
- **2.3 Equi-probability Density Contours**: Definition and elliptical properties.

## 3. Properties of Multivariate Normal Distribution
- **3.1 Linear Combinations**: Distribution properties and reverse deduction.
- **3.2 Affine Transformations**: Behavior under linear transformations.
- **3.3 Translation**: Impact of constant vector shifts.
- **3.4 Conditional Distributions**: Conditional mean and covariance formulas.

## 4. Quadratic Forms and Related Distributions
- **4.1 Quadratic Forms**: $\\chi^2$ distribution and related properties.
- **4.2 Standardization**: Properties of the standardized normal distribution.

## 5. Sampling from Multivariate Normal Distribution
- **5.1 Joint Density Function**: Formula for the joint distribution of samples.
- **5.2 Maximum Likelihood Estimation**: Estimation of sample mean and covariance matrix.

## 6. Wishart Distribution
- **6.1 Definition**: Distribution of the sample covariance matrix.
- **6.2 Properties**:
  - Independence and summation properties.
  - Transformations of matrices.

## 7. Large Sample Properties
- **7.1 Central Limit Theorem**: Asymptotic properties for large samples.

## 8. Assessing the Assumption of Normality
- **8.1 Univariate Normality Tests**: Using histograms and probability plots.
- **8.2 Multivariate Normality Tests**: Chi-square distances and quantile comparisons.

-----------------------
# 1. Introduction

## 1.1 概述：多元正态分布的重要性和应用
- **多元正态分布 (Multivariate Normal Distribution)** 是单变量正态分布的扩展，适用于高维数据。
- 它因其数学性质和可解释性，在多元统计分析中占据重要地位。
- 许多统计和机器学习技术，例如 **主成分分析 (Principal Component Analysis, PCA)**、**线性判别分析 (Linear Discriminant Analysis, LDA)** 和 **高斯混合模型 (Gaussian Mixture Models, GMM)** 都假设数据遵循多元正态分布。
- 即使真实数据并不完全服从正态分布，根据 **中心极限定理 (Central Limit Theorem, CLT)**，独立随机变量的和趋近正态分布，因此多元正态分布通常是实际分布的合理近似。

---

# 2. The Multivariate Normal Density and Its Properties

## 2.1 单变量正态分布 (Univariate Normal Distribution)

- **定义 (Definition)**：
  - 单变量正态分布描述的是一个随机变量 $x$ 的概率分布。
  - 若 $x$ 服从正态分布，记为：
    $$
    x \sim N(\mu, \sigma^2)
    $$
    其中：
    - $\mu$ 是均值 (Mean)，表示分布的中心。
    - $\sigma^2$ 是方差 (Variance)，描述分布的宽度。
- **概率密度函数 (Probability Density Function, PDF)**：
  - 单变量正态分布的 PDF 表达式为：
    $$
    f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{1}{2} \left( \frac{x-\mu}{\sigma} \right)^2}, \quad -\infty < x < \infty
    $$
    该函数具有以下性质：
    1. 对称性：以 $\mu$ 为中心对称。
    2. 范围：概率密度函数的定义域为 $(-\infty, \infty)$。
    3. 面积：曲线下的总面积为 1。

## 2.2 多元正态分布 (Multivariate Normal Distribution)
### 2.2.1 密度函数公式 (Density Function Formula)

- **定义**：
  - 多元正态分布描述的是随机向量 $X = (X_1, X_2, \dots, X_p)'$ 的概率分布。
  - 若 $X$ 服从多元正态分布，记为：
    $$
    X \sim N_p(\mu, \Sigma)
    $$
    其中：
    - $\mu$ 是 $p \times 1$ 的均值向量 (Mean Vector)，表示分布的中心。
    - $\Sigma$ 是 $p \times p$ 的协方差矩阵 (Covariance Matrix)，描述变量之间的相关性。
- **概率密度函数 (PDF)**：
  - 多元正态分布的 PDF 表达式为：
    $$
    f(X) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} e^{-\frac{1}{2} (X-\mu)'\Sigma^{-1}(X-\mu)}
    $$
    其中：
    - $|\Sigma|$ 是协方差矩阵的行列式 (Determinant)。
    - $\Sigma^{-1}$ 是协方差矩阵的逆矩阵 (Inverse Matrix)。
  - **性质**：
    1. 协方差矩阵 $\Sigma$ 是正定矩阵 (Positive Definite Matrix)
    2. 分布中心由均值向量 $\mu$ 确定。

假设有两个随机变量，$X$ 和 $Y$，它们服从二维正态分布，均值向量和协方差矩阵设定如下：

1. 均值向量
$$
\boldsymbol{\mu} =
\begin{pmatrix}
\mu_X \\
\mu_Y
\end{pmatrix}
=
\begin{pmatrix}
2 \\
3
\end{pmatrix}.
$$

2. **协方差矩阵**

**为了体现相关性，设相关系数为 $\rho = 0.5$；再设 $X$ 的标准差为 $\sigma_X = 1$，$Y$ 的标准差为 $\sigma_Y = 2$，则**

$$
\Sigma
\begin{pmatrix}
\sigma_X^2 & \rho,\sigma_X \sigma_Y \\
\rho,\sigma_X \sigma_Y & \sigma_Y^2
\end{pmatrix}
\begin{pmatrix}
1^2 & 0.5 \times 1 \times 2 \\
0.5 \times 1 \times 2 & 2^2
\end{pmatrix}
=
\begin{pmatrix}
1 & 1 \\
1 & 4
\end{pmatrix}.
$$

这样，$(X, Y)$ 的二维正态分布可记为

$$
(X, Y) \sim \mathcal{N}\bigl(\boldsymbol{\mu}, \Sigma\bigr).
$$


### 2.2.2 特例：二维正态分布 (Bivariate Normal Distribution)

- 当 $p = 2$ 时，$X = (X_1, X_2)'$ 的多元正态分布公式可以写为：
  $$
  f(X_1, X_2) = \frac{1}{2\pi|\Sigma|^{1/2}} e^{-\frac{1}{2} \begin{pmatrix} X_1 - \mu_1 \\ X_2 - \mu_2 \end{pmatrix}' \Sigma^{-1} \begin{pmatrix} X_1 - \mu_1 \\ X_2 - \mu_2 \end{pmatrix}}
  $$
- 其中协方差矩阵：
  $$
  \Sigma =\begin{pmatrix} \sigma_{11} & \sigma_{12} \\ \sigma_{12} & \sigma_{22} \end{pmatrix}=\begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix}
  $$
  这里：
  - $\rho$ 是两个随机变量之间的相关系数 (Correlation Coefficient)。
  - $\sigma_1$ 和 $\sigma_2$ 是 $X_1$ 和 $X_2$ 的标准差 (Standard Deviation)。

写作:$y\sim BN(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho_{12})$

二元正态分布的概率密度函数公式如下：

$$
f(x, y) = \frac{1}{2\pi \sigma_x \sigma_y \sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_x)^2}{\sigma_x^2} + \frac{(y-\mu_y)^2}{\sigma_y^2} - \frac{2\rho(x-\mu_x)(y-\mu_y)}{\sigma_x \sigma_y}\right]\right)
$$

对于二元正态分布：

协方差矩阵 $\Sigma$ 的行列式为：

$$
|\Sigma| = \begin{vmatrix}
\sigma_x^2 & \rho \sigma_x \sigma_y \\
\rho \sigma_x \sigma_y & \sigma_y^2
\end{vmatrix}
= \sigma_x^2 \sigma_y^2 (1 - \rho^2)
$$

协方差矩阵 $\Sigma$ 的逆矩阵为：

$$
\Sigma^{-1} = \frac{1}{|\Sigma|} \begin{bmatrix}
\sigma_y^2 & -\rho \sigma_x \sigma_y \\
-\rho \sigma_x \sigma_y & \sigma_x^2
\end{bmatrix}
= \frac{1}{\sigma_x^2 \sigma_y^2 (1-\rho^2)} \begin{bmatrix}
\sigma_y^2 & -\rho \sigma_x \sigma_y \\
-\rho \sigma_x \sigma_y & \sigma_x^2
\end{bmatrix}
$$

将这些代入多元正态分布的公式：

1. $\mathbf{X} - \mu = \begin{bmatrix} x - \mu_x \\ y - \mu_y \end{bmatrix}$；

2. 二次型 $(\mathbf{X} - \mu)^\top \Sigma^{-1} (\mathbf{X} - \mu)$ 展开为：

$$
(\mathbf{X} - \mu)^\top \Sigma^{-1} (\mathbf{X} - \mu)
= \frac{1}{1-\rho^2} \left[
\frac{(x - \mu_x)^2}{\sigma_x^2} +
\frac{(y - \mu_y)^2}{\sigma_y^2} -
\frac{2\rho (x - \mu_x)(y - \mu_y)}{\sigma_x \sigma_y}
\right]
$$

最终得到二元正态分布的概率密度函数：

$$
f(x, y) = \frac{1}{2\pi \sigma_x \sigma_y \sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[
\frac{(x - \mu_x)^2}{\sigma_x^2} +
\frac{(y - \mu_y)^2}{\sigma_y^2} -
\frac{2\rho (x - \mu_x)(y - \mu_y)}{\sigma_x \sigma_y}
\right]\right)
$$

## 2.3 等概率密度曲线 (Equi-probability Density Contours)

- **定义 (Definition)**：

“**多元正态分布的等概率密度曲线（或曲面）**”是由满足

$$
(X - \mu)’\Sigma^{-1}(X - \mu) = c^2
$$

的所有点 $X$ 组成的。这里：
• $X$ 表示一个 $p \times 1$ 的向量（即 $p$ 维随机向量）。
• $\mu$ 是它的均值向量（同为 $p \times 1$）。
• $\Sigma$ 是它的协方差矩阵（$p \times p$）。
• $\Sigma^{-1}$ 是协方差矩阵的逆矩阵。
• $c$ 是一个常数。

在多元正态分布的概率密度函数中，指数部分的“核心”就是

$$

-\frac12 (X - \mu)’,\Sigma^{-1},(X - \mu).

$$

当我们固定这个二次型 等于某个常数 时，就得到了一条（或一族）“等概率密度曲线”（在高维时是“等概率密度超曲面”）。几何上，这些集合通常是**以 $\mu$ 为中心的椭球体**（二维时是椭圆、三维是椭球，更高维则是超椭球）。
之所以是椭圆或椭球，是因为该方程实际上定义了一个**加权距离**：

$$

(X - \mu)’,\Sigma^{-1},(X - \mu) = \text{常数},

$$

它描述了点 $X$ 到中心 $\mu$ 的“二次距离”，并由协方差矩阵 $\Sigma$ 来决定各个方向的伸缩和倾斜。在二维情况下，我们看到的等高线是一组围绕均值点的椭圆；在三维或更高维中则是椭球或更复杂的超曲面。

- **椭圆性质 (Elliptical Properties)**：
  1. 曲线的形状是椭圆 (Ellipse)，其中心位于均值向量 $\mu$。
  2. 椭圆的方向和大小由协方差矩阵 $\Sigma$ 决定：
     - 主轴方向对应 $\Sigma$ 的特征向量 (Eigenvectors)。
     - 主轴长度与特征值 (Eigenvalues) 成正比。
  3. 当协方差矩阵为对角矩阵时，等概率密度曲线为圆形。

--------

# 3. Properties of Multivariate Normal Distribution

**前置示例设定**

我们先给出一个**二维**（即 $p=2$）的随机向量 $X$：

$$
X
= \begin{pmatrix} X_1 \ X_2 \end{pmatrix}
\sim N_2\Bigl(
\begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix},
\Sigma
\Bigr),
$$
其中

$$
\mu_1 = 2,\quad \mu_2 = 3,
$$

$$
\Sigma
= \begin{pmatrix}
1 & 0.5 \\
0.5 & 2
\end{pmatrix}.
$$

这样，$X$ 的均值向量是$\begin{pmatrix}2\\3\end{pmatrix},$

协方差矩阵是
$$
\begin{pmatrix}
1 & 0.5\\
0.5 & 2
\end{pmatrix}.
$$

## 3.1 线性组合 (Linear Combinations)

- 如果 $X \sim N_p(\mu, \Sigma)$ 且 $a$ 是 $p \times 1$ 的常数向量，则 $a'X$ 的分布为：
  $$
  a'X \sim N(a'\mu, a'\Sigma a)
  $$
- **反向推导**：
  - 如果对于任意向量 $a$，$a'X \sim N(a'\mu, a'\Sigma a)$ 都成立，那么可以证明：
    $$
    X \sim N_p(\mu, \Sigma)
    $$

**数值示例**

1. 令

$$
a
= \begin{pmatrix}2\\1\end{pmatrix}.
$$

2. 则线性组合为

$$

a’X

= 2X_1 + X_2.

$$

3. 根据结论，有

$$

a’X \sim N\bigl(a’\mu,; a’\Sigma a\bigr).

$$

先计算 $a’\mu$：

$$

a’\mu

= \begin{pmatrix}2 & 1\end{pmatrix}

\begin{pmatrix}2\\3\end{pmatrix}

= 2 \times 2 + 1 \times 3

= 7.

$$

  

再计算 $a’\Sigma a$：

先做 $\Sigma a$：

$$
\Sigma a
= \begin{pmatrix}
1 & 0.5\\
0.5 & 2
\end{pmatrix}
\begin{pmatrix}
2 \\
1
\end{pmatrix}
= \begin{pmatrix}
1 \times 2 + 0.5 \times 1\\[4pt]
0.5 \times 2 + 2 \times 1
\end{pmatrix}
= \begin{pmatrix}
2.5\\
3
\end{pmatrix}.
$$

然后
$$
a’\Sigma a
= \begin{pmatrix}2 & 1\end{pmatrix}
\begin{pmatrix}
2.5\\
3
\end{pmatrix}
= 2 \times 2.5 + 1 \times 3
= 5 + 3
= 8.
$$
因此
$$
a’X \sim N(7,8).
$$


## 3.2 仿射变换 (Affine Transformations)
- 若 $X \sim N_p(\mu, \Sigma)$，$A$ 是 $q \times p$ 的矩阵，则仿射变换 $AX$ 的分布为：
  $$
  AX \sim N_q(A\mu, A\Sigma A')
  $$

**数值示例**

令
$$
A
= \begin{pmatrix}
1 & 2\\
0 & 1
\end{pmatrix}.
$$
则
$$
AX
= \begin{pmatrix}
1 & 2\\
0 & 1
\end{pmatrix}
\begin{pmatrix}
X_1\\
X_2
\end{pmatrix}
= \begin{pmatrix}
X_1 + 2X_2\\
X_2
\end{pmatrix}.
$$

根据结论：
$$
AX \sim N_2(A\mu,A\Sigma A’).
$$
• 均值向量：

$$
A\mu
= \begin{pmatrix}
1 & 2\\
0 & 1
\end{pmatrix}
\begin{pmatrix}
2\\
3
\end{pmatrix}
= \begin{pmatrix}
2 + 2 \times 3\\
3
\end{pmatrix}
= \begin{pmatrix}
8\\
3
\end{pmatrix}.
$$

• 协方差矩阵 $A\Sigma A’$ 可以照常做矩阵乘法即可，结果依然是一个 $2\times2$ 的对称、正定矩阵。

## 3.3 平移 (Translation)
- 若 $X \sim N_p(\mu, \Sigma)$ 且 $d$ 是 $p \times 1$ 的常量向量，则平移后的分布为：
  $$
  X + d \sim N_p(\mu + d, \Sigma)
  $$

## 3.4 条件分布 (Conditional Distributions)

==考试的时候不考==

**分块向量与协方差的分块**

设随机向量

$$
X
= \begin{pmatrix}
X_1\\
X_2
\end{pmatrix}
\sim
N_p\Bigl(
\begin{pmatrix}
\mu_1\\
\mu_2
\end{pmatrix},
\begin{pmatrix}
\Sigma_{11} & \Sigma_{12}\\
\Sigma_{21} & \Sigma_{22}
\end{pmatrix}
\Bigr),
$$

其中：
• $X$ 是一个 $p\times1$ 的向量，被拆分成两个子向量 $X_1$ 和 $X_2$；
• $X_1$ 维度为 $q\times1$，$X_2$ 维度为 $(p-q)\times1$；
• $\mu_1$、$\mu_2$ 分别是 $X_1$、$X_2$ 的均值向量；
• 协方差矩阵分成 4 块：
$$
\Sigma_{11} \text{ 是 } q\times q,
\quad
\Sigma_{22} \text{ 是 }(p-q)\times(p-q),
\quad
\Sigma_{12} = \Sigma_{21}’ \text{ 是 }q\times(p-q).

$$

**结论** 涉及到两件事：

1. **独立性**
• 对于多元正态分布，$X_1$ 与 $X_2$ 独立的充要条件是
$$\Sigma_{12} = 0.$$
• 也就是说，在多元正态情形下，“零协方差” 等价于 “彼此独立”。（非正态情况下，这种等价一般不成立。）

2. **条件分布**

• $X_1$ 在已知 $X_2=x_2$ 的条件下，依然服从正态分布，其参数有一个**非常整齐**的公式：

$$
X_1 \mid X_2 = x_2
\sim
N\Bigl(
\mu_1 +\Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2),
\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\Bigr).
$$

**几何/直观理解**
• “$X_1$ 的条件均值” 会根据 “$X_2$ 的观测值” 做线性调整；
• “$X_1$ 的条件协方差” 则变小了（相比原先的 $\Sigma_{11}$ 少了一个正定项），说明在知道 $X_2$ 之后，$X_1$ 的不确定性降低。

**进一步剖析——为何有此公式**

在多元正态中，联合密度函数可以写成指数里带有

$(X-\mu)’\Sigma^{-1}(X-\mu)$ 的二次型。将 $X$ 拆分成 $(X_1, X_2)$ 后，对应地可以对该二次型做块状分解。

通过完成平方的技巧，或者直接引用“多元正态分布的条件分布公式”，就能得到上面所列的结果。其核心原因是：

1. 多元正态的“线性-高斯”结构使得条件分布必定仍是正态；
2. 条件均值和条件方差（或协方差）可以通过“块矩阵逆”或“舒尔补( Schur complement )”之类的方法推导出来。

**数值示例：二维情形 (最简单的分块)**

为了让内容更直观，我们使用 $p=2$、$q=1$ 的场景，把 $X_1$ 和 $X_2$ 都当作**标量**来举例。此时协方差矩阵分块就很简单：

1. $X_1$ 是 $1\times1$（即一个标量），$X_2$ 也是 $1\times1$（另一个标量）。
2. $\Sigma_{11}$、$\Sigma_{22}$、$\Sigma_{12}=\Sigma_{21}$ 都是实数。

**设定具体数值**

令

$$
X
= \begin{pmatrix}
X_1\\
X_2
\end{pmatrix}
\sim N_2\Bigl(
\begin{pmatrix}
2\\
3
\end{pmatrix},
\begin{pmatrix}
1 & 0.5\\
0.5 & 2
\end{pmatrix}
\Bigr).
$$

• 这里，$X_1$ 与 $X_2$ 的均值分别是 $2$ 和 $3$；

• 协方差矩阵中，
$$
\Sigma_{11} = 1,\quad
\Sigma_{22} = 2,\quad
\Sigma_{12} = \Sigma_{21} = 0.5.
$$

**1. 独立性检查**

• 因为 $\Sigma_{12} = 0.5 \neq 0$，所以根据结论可知：$X_1$ 和 $X_2$ **并不独立**。
• 若我们把 $0.5$ 改成 $0$，那就变成对角矩阵

$$
\begin{pmatrix}
1 & 0\\
0 & 2
\end{pmatrix},
$$

这时才表示 $X_1$ 与 $X_2$ **独立且都正态**。

**2. 条件分布公式**

• 已知 $X_2 = x_2$ 时，$X_1$ 服从一维正态，参数如下：

$$
X_1 \mid X_2 = x_2
\sim
N!\Bigl(
\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2),
;\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\Bigr).
$$

• 在本例中，所有符号都是标量，因此计算相当直接：
• $\Sigma_{22}^{-1} = \frac{1}{2}$；
• $\Sigma_{12} = 0.5$；
• $\mu_1=2,;\mu_2=3,;\Sigma_{11}=1$。
因此

$$
\text{条件均值}
= 2 + 0.5 \times \frac{1}{2}(x_2 - 3)
= 2 + 0.25(x_2 - 3)
$$

$$
\text{条件方差}
= 1 - 0.5 \times \frac{1}{2} \times 0.5
= 1 - 0.125
= 0.875.
$$

于是

$$
X_1\mid X_2=x_2
\sim
N\Bigl(2 + 0.25(x_2 - 3);0.875\Bigr).
$$

• 这就说明：当 $X_2$ 取值越大时（比均值 3 更大），$X_1$ 的条件均值会相应**上调**一些（因为两者是正相关）；反之亦然。

**一句话概括**

• **在多元正态分布中**，“零协方差” 等价于 “子向量之间独立”；
• 即使不独立，子向量的**条件分布**依然是正态，其均值和协方差具有明晰的线性修正公式。

---------------

# 4. Quadratic Forms and Related Distributions

## 4.1 平方形式的分布 (Distribution of Quadratic Forms)

==很重要的内容.是后面很多知识点的基础==

若有随机向量

$$
X \sim N_p(\mu, \Sigma),
$$
则称
$$
D^2 = (X - \mu)’\Sigma^{-1}(X - \mu)
$$
为它的「**（平方）马哈拉诺比斯距离**」。

2. 这个 $D^2$ 服从自由度为 $p$ 的卡方分布（记作 $\chi^2_p$）。

为什么会有这样的结论?关键步骤是利用线性变换把 $X$ 转成一个标准正态向量 $Z$：

• 定义
$$
Z =\Sigma^{-\tfrac12}(X-\mu).
$$
若 $X \sim N_p(\mu,\Sigma)$，则可以证明
$$
Z \sim N_p\bigl(0,I_p\bigr),
$$

也就是 $p$ 维标准正态分布（协方差矩阵为单位阵 $I_p$）。

• 接着，$(X - \mu)’,\Sigma^{-1},(X - \mu)$ 可以写成

$$
(\Sigma^{-\tfrac12}(X-\mu))’(\Sigma^{-\tfrac12}(X-\mu))
=Z’Z
= Z_1^2 + Z_2^2 + \cdots + Z_p^2.
$$

因为在 $Z$ 中，每个分量 $Z_i$ 都是服从 $N(0,1)$ 的独立随机变量，所以它们的平方和自然就是 $\chi^2_p$ 分布。

  
• **马哈拉诺比斯距离**：在多元正态中，$(X - \mu)’ \Sigma^{-1} (X - \mu)$ 可以看作点 $X$ 到均值 $\mu$ 的“加权距离”——其中 $\Sigma$ 决定了不同坐标方向上的尺度和相关性。

• **卡方分布**：在一维正态中，标准正态随机变量 $Z \sim N(0,1)$ 的平方 $Z^2$ 服从 $\chi^2_1$；多维情况相当于把若干个独立的标准正态平方加起来，就得到 $\chi^2_p$。

• **几何或统计推断应用**：该结论常用于统计检验、异常值检测、构造置信椭球等场景。例如，如果我们希望检验 $X$ 是否“偏离” $\mu$ 太多，就可以看它的马哈拉诺比斯距离是否过大，从而判断是否落在置信椭球外面。

总之，给定 $X \sim N_p(\mu, \Sigma)$，经过合适的线性变换（$\Sigma^{-1/2}$）可以把它「标准化」成 $Z \sim N_p(0,I_p)$，而由此推导出 $(X-\mu)’,\Sigma^{-1},(X-\mu)$ 的分布是 $\chi^2_p$。


## 4.2 标准化 (Standardization)

- **定义**：
  - 通过以下变换对 $X$ 进行标准化：
    $$
    Z = \Sigma^{-1/2}(X - \mu)
    $$
    其中 $\Sigma^{-1/2}$ 是协方差矩阵 $\Sigma$ 的平方根逆矩阵。
    
- **标准化后的性质**：
  1. $Z$ 服从标准正态分布 (Standard Normal Distribution)：
     $$
     Z \sim N_p(0, I)
     $$
     其中 $I$ 是 $p \times p$ 的单位矩阵。
  2. 标准化后的变量之间独立。

---

# 5. Sampling from Multivariate Normal Distribution

## 5.1 联合密度函数 (Joint Density Function)

- 假设从多元正态分布 $N_p(\mu, \Sigma)$ 中抽取 $n$ 个独立样本 $X_1, X_2, \dots, X_n$，则这些样本的联合概率密度函数为：
  $$
  f(X_1, X_2, \dots, X_n) = \prod_{j=1}^n \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} e^{-\frac{1}{2} (X_j - \mu)'\Sigma^{-1}(X_j - \mu)}
  $$
  或者等价地表示为：
  $$
  f(X_1, \dots, X_n) = \frac{1}{(2\pi)^{np/2} |\Sigma|^{n/2}} e^{-\frac{1}{2} \sum_{j=1}^n (X_j - \mu)'\Sigma^{-1}(X_j - \mu)}
  $$
- **解释**：
  - 联合密度函数描述了 $n$ 个样本的联合概率。
  - 总体均值为 $\mu$，协方差矩阵为 $\Sigma$。

## 5.2 最大似然估计 (Maximum Likelihood Estimation)

==考试不会考证明==

### 5.2.1 多元正态分布的最大似然估计

多元正态分布（Multivariate Normal）的似然函数（Likelihood）。如果我们假设有 $n$ 个观测向量 $X_1, X_2, \dots, X_n$，它们都是从 $N_p(\mu, \Sigma)$（维度为 $p$ 的多元正态分布，均值为 $p \times 1$ 向量 $\mu$，协方差矩阵为 $p \times p$ 的对称正定矩阵 $\Sigma$）中独立同分布采样而来，则它们的联合概率密度函数写成一个乘积形式。具体来说：

  $$
\prod_{j=1}^n {
\frac{1}{(2\pi)^{p/2} , |\Sigma|^{1/2}} \exp(-\frac{1}{2} (X_j - \mu)^\prime \Sigma^{-1} (X_j - \mu) )
}=
\frac{1}{(2\pi)^{\frac{np}{2}} , |\Sigma|^{\frac{n}{2}}}
\exp ( -\frac{1}{2} \sum_{j=1}^n (X_j - \mu)^\prime \Sigma^{-1} (X_j - \mu) ).
$$

• 其中，$(X_j - \mu)^\prime \Sigma^{-1} (X_j - \mu)$ 就是“偏离均值向量”的二次型，它衡量了 $X_j$ 相对于均值 $\mu$ 的“距离”（由协方差矩阵 $\Sigma$ 来度量）。
• 似然函数就是把单个样本点的概率密度相乘得到的结果，因为样本独立同分布。

可以对均值向量和协方差矩阵进行最大似然估计 (MLE)：
  - **均值向量的估计**：
    $$
    \hat{\mu} = \bar{X} = \frac{1}{n} \sum_{j=1}^n X_j
    $$
    其中 $\bar{X}$ 是样本均值。
  - **协方差矩阵的估计**：
    $$
    \hat{\Sigma} = \frac{1}{n} \sum_{j=1}^n (X_j - \bar{X})(X_j - \bar{X})'=\frac{n-1}{n}S
    $$

- **样本协方差矩阵的性质**：
  1. 如果 $n > p$，样本协方差矩阵是非奇异的。 
  2. 如果 $n \leq p$，样本协方差矩阵可能是奇异的，无法直接求逆。

在估计时用到了一个常见的矩阵恒等式，说明了向量-矩阵-向量的二次型 $x^\prime A x$ 可以用迹 (trace) 的形式来表示：
$$
x^\prime A x= \mathrm{tr} \bigl(x^\prime A x\bigr) = \mathrm{tr}\bigl(A  x x^\prime\bigr),
$$
这里 $A$ 是 $k \times k$ 的对称矩阵，$x$ 是 $k \times 1$ 的向量。
• 为啥可以写成迹？因为标量（即 $1 \times 1$ 矩阵）的迹就是它本身，而且使用迹可以帮助在推导、化简中做“循环换位”，比如 $\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB)$，经常用来简化带有二次型的式子。

**一个简单例子：**

假设在一维情形（即 $p=1$），那么 $X_j - \mu$ 就是一个数字，$\Sigma$ 也就是一个方差 $\sigma^2$；此时 $(X_j - \mu)^\prime \Sigma^{-1} (X_j - \mu)$ 其实就是
$$
\frac{(X_j - \mu)^2}{\sigma^2}.
$$

多维情形下，这个概念会变成向量和矩阵，写成 $(X_j - \mu)^\prime \Sigma^{-1} (X_j - \mu)$。如果我们希望用迹来写这个“距离”，也可以用
$$
\mathrm{tr}\bigl(\Sigma^{-1} (X_j - \mu)(X_j - \mu)^\prime\bigr),
$$

因为上面提到的 $x^\prime A x = \mathrm{tr}(Axx^\prime)$。

### **5.2.2. 不变性 (Invariance property)**

**结论**：如果$\hat{\theta}$是$\theta$的最大似然估计（MLE），并且$h(\theta)$是关于参数$\theta$的某个函数，那么
$$
h(\hat{\theta})
$$

就是$h(\theta)$的最大似然估计。

换句话说，**先估计参数$\theta$，再对估计结果$\hat{\theta}$做函数变换$h(\hat{\theta})$**，就等价于**先对$\theta$做函数变换$h(\theta)$，再对$h(\theta)$做最大似然估计**。这就是MLE的不变性原理。

一个简单的一维例子是：

• 若我们有单参数正态分布$N(\mu,\sigma^2)$，其均值$\mu$的最大似然估计为
$$
\hat{\mu} = \overline{X}.
$$
• 若我们关心$e^\mu$，根据不变性，$e^\mu$的MLE就是
$$
e^{\hat{\mu}}.
$$
在多元正态分布中也是同理：只要将$\hat{\theta}$代入$h(\theta)$即可得到$h(\theta)$的MLE。

### **5.2.3. 多元正态分布中的一些 MLE 举例**

1. MLE of $\mu\Sigma^{-1}\mu$

对于多元正态分布$N(\mu,\Sigma)$，假设已知

• 均值向量的MLE：$\hat{\mu}$
• 协方差矩阵的MLE：$\hat{\Sigma}$

那么对于函数
$$
\mu \Sigma^{-1} \mu,
$$
根据不变性原理，它的MLE即为
$$
\hat{\mu},\hat{\Sigma}^{-1},\hat{\mu}.
$$
2. MLE of $\sqrt{\sigma_{ii}}$

如果我们只关心协方差矩阵中的对角线元素$\sigma_{ii}$，其最大似然估计（在某些定义下）是

$$
\hat{\sigma}_{ii}= \frac{1}{n} \sum_{j=1}^n \bigl(X_{ji} - \overline{X}_i\bigr)^2.

$$

那么要估计$\sqrt{\sigma_{ii}}$，根据不变性原理，直接取
$$
\sqrt{\hat{\sigma}_{ii}}.
$$
### **5.2.4. 充足性 (Sufficiency)**

**结论**：在多元正态分布$N(\mu,\Sigma)$中，样本均值向量$\overline{X}$和样本协方差矩阵（或与之成比例的$(n-1)S$）是关于$\mu$和$\Sigma$的**充分统计量**。

直观理解：给定充分统计量$T(X)$后，原始样本数据所包含的关于参数的信息就“浓缩”在$T(X)$里；在多元正态情形下：

• 关于均值向量$\mu$的全部信息都在样本均值$\overline{X}$中，
• 关于协方差矩阵$\Sigma$的全部信息都在样本协方差矩阵$S$中。

因此，无论样本容量$n$如何变化，只要知道了$\overline{X}$与$S$（或$(n-1)S$），就已经掌握了样本中关于$\mu$、$\Sigma$的全部信息  

**示例**

假设有二维正态分布$N(\mu,\Sigma)$，其中
$$
\mu =
\begin{pmatrix}
\mu_1 \\
\mu_2
\end{pmatrix},
\quad
\Sigma =
\begin{pmatrix}
\sigma_{11} & \sigma_{12} \\
\sigma_{21} & \sigma_{22}
\end{pmatrix}.
$$
从中抽样得到$n$个观测值
$$
X_1, X_2, \ldots, X_n \quad (\text{每个 }X_i \in \mathbb{R}^2).
$$
• 样本均值向量为
$$
\overline{X}
= \frac{1}{n} \sum_{i=1}^n X_i.
$$
• 若按MLE的定义（分母取$n$）得到的样本协方差矩阵为
$$
\hat{\Sigma}
= \frac{1}{n} \sum_{i=1}^n (X_i - \overline{X})(X_i - \overline{X})^\top.
$$
此时，任何与$\mu$、$\Sigma$相关的量（例如$\mu \Sigma^{-1}\mu$或$\sqrt{\sigma_{ii}}$等）的估计，都可以只依赖$\overline{X}$和$\hat{\Sigma}$来完成，而不再需要回到原始样本点。

----------------------

# 6. Wishart Distribution

==如果总体是正态分布,那么样本的方差协方差矩阵就是Wishart 分布==

## 6.1 定义 (Definition)

**1. 样本协方差矩阵的分布及Wishart分布**

当我们从$N_p(\mu,\Sigma)$（$p$维正态分布）中抽取样本$X_1, X_2, \dots, X_n$（每个$X_i\in\mathbb{R}^p$），定义样本协方差矩阵$S$为
$$
S = \frac{1}{n-1}\sum_{i=1}^n\bigl(X_i - \overline{X}\bigr)\bigl(X_i - \overline{X}\bigr)^\top,
$$
其中$\overline{X}$为样本均值向量
$$
\overline{X} = \frac{1}{n}\sum_{i=1}^n X_i.
$$

在这里，我们有以下结论：

1. $(n-1)S$服从Wishart分布(Wishart distribution)，记为

$$

(n-1)S \sim W_p\bigl(n-1, \Sigma\bigr),

$$

其中$(n-1)$是Wishart分布的自由度(degrees of freedom)，$\Sigma$是底层正态分布的真协方差矩阵。

2. 由多元正态分布的性质可知，$\overline{X}$与$S$相互独立(independent)。且
$$
\overline{X} \sim N_p\Bigl(\mu,;\frac{1}{n}\Sigma\Bigr).
$$

这意味着在抽样得到$X_1,\ldots,X_n$后，**样本均值向量**与**样本协方差矩阵**分别具有各自的分布，并且二者独立。

**2. Wishart分布的形成原理**

Wishart分布可以看作以下过程的矩阵形式：假设$Z_1,\dots,Z_m$是相互独立的$N_p(0,\Sigma)$随机向量（每个$Z_j\in\mathbb{R}^p$），那么

  

$$

\sum_{j=1}^m Z_j Z_j^\top

$$

  

就服从一个$W_p(m,\Sigma)$（即具有$m$自由度的Wishart分布，以$\Sigma$为参数）。在抽样均值为0的情形下，可以将$n$个$X_i-\mu$视作相互独立同分布的$N_p(0,\Sigma)$随机向量，从而得到$(n-1)S$的Wishart分布性质。

**注意**：如果原分布不是中心化的（$\mu\neq 0$），我们可以先将样本向量减去其均值$\overline{X}$，再乘以$\Sigma^{-1/2}$，在一定变换下把问题转化为中心分布的形式，这就揭示了$(n-1)S$是Wishart分布的来龙去脉  

**3. 主要结论与应用**

1. **$(n-1)S$的分布**

当样本来自$N_p(\mu,\Sigma)$时，样本协方差矩阵的无缩放版本$(n-1)S$服从Wishart分布，记为
$$
(n-1)S \sim W_p\bigl(n-1, \Sigma\bigr).
$$

2. **$\overline{X}$的分布**

样本均值$\overline{X}$服从$N_p\Bigl(\mu,\tfrac{1}{n}\Sigma\Bigr)$。并且它与$S$相互独立。

3. **独立性**

在多元正态分布下，$\overline{X}$与$S$的相互独立是一个非常重要且常被使用的结论。它在构造区间估计、假设检验以及各种推断场景（如Hotelling$T^2$检验）时都起到关键作用。

**4. 一个简单的直观认识**

• **在一维情况下**（$p=1$时），Wishart分布退化为$\chi^2$分布（卡方分布）。因为此时$(n-1)S$就是样本方差乘以$(n-1)$，而它正好符合$\chi^2_{n-1}$分布。

• **在多维情况下**（$p>1$），$(n-1)S$不仅是一个标量，而是一个正定对称随机矩阵；Wishart分布可以被视为多维情形下的“卡方分布矩阵版”。

## 6.2 性质

**1. 互不依赖的Wishart随机矩阵之和**

**性质**：如果$A_1 \sim W_{m_1}(\Sigma)$、$A_2 \sim W_{m_2}(\Sigma)$，并且$A_1$和$A_2$相互独立，那么

$$
A_1 + A_2 \sim W_{m_1 + m_2}\bigl(\Sigma\bigr).
$$
**解读**：

• $W_{m}(\Sigma)$表示一个服从自由度为$m$、协方差矩阵为$\Sigma$的Wishart分布的随机矩阵。
• 若有两个独立的Wishart随机矩阵$A_1$、$A_2$，它们的自由度分别为$m_1$和$m_2$，且都基于同一个$\Sigma$，则它们的**矩阵之和**依旧是Wishart分布，且新自由度为$m_1+m_2$。
• 这个性质与一维情形下的“若$X_1 \sim \chi^2_{m_1}$、$X_2 \sim \chi^2_{m_2}$且相互独立，则$X_1 + X_2 \sim \chi^2_{m_1+m_2}$”非常类似，可以视作在多维情形下的推广。

**2. 矩阵相似变换保持Wishart分布形式**

**性质**：如果$A \sim W_m(\Sigma)$，那么对于任何可逆矩阵$C$（通常需要是可逆的实矩阵），都有
$$
CAC’ \sim W_m\bigl(C\Sigma C’\bigr).
$$

**解读**：

• 这条性质说明了Wishart分布在**相似变换**（或者说双边变换）下的协方差矩阵相应发生了$C,\Sigma,C’$的变换，而自由度$m$保持不变。
• 这与中心对称正态随机向量做线性变换后仍然是高斯随机向量的原理相似：若我们对一个Wishart随机矩阵$A$施加一个双边线性变换$CAC’$，新矩阵依旧遵循某个Wishart分布，只不过底层的“协方差”矩阵换成了$C\Sigma C’$。
• 一般在多元统计中的**变换不变性**、**协方差矩阵估计在不同度量下的表达**等场景会用到这一性质。

---

# 7. Large Sample Properties

## 7.1 中心极限定理 (Central Limit Theorem)

**1. 大数定理与中心极限定理的多维情形**

令$X_1, X_2, \dots, X_n$是来自同一个总体的独立观测，假设该总体具有均值向量$\mu$和有限的协方差矩阵$\Sigma$。课件中给出的结论表明，当样本容量$n$较大且$n > p$（其中$p$是向量的维度）时，有

$$

\sqrt{n}\bigl(\overline{X} - \mu\bigr) \Rightarrow N_p\bigl(0,\Sigma\bigr).

$$

• 这里$\overline{X} = \tfrac{1}{n}\sum_{i=1}^n X_i$是样本均值向量。
• $\Rightarrow$表示“收敛于分布上”(convergence in distribution)，也称渐近分布。
• 这实际上是**中心极限定理**在多维情形下的体现：样本均值的分布会在大样本极限下趋近于多元正态分布。

**2. 当$\overline{X}$来自正态总体时，$n(\overline{X}-\mu)’\Sigma^{-1}(\overline{X}-\mu)$服从$\chi^2_p$**

如果观测$X_i$本身就是从$N_p(\mu,\Sigma)$（多元正态分布）抽样得到的，那么我们可以精确地推导出下面这个结论：

$$
n\bigl(\overline{X}-\mu\bigr)’\Sigma^{-1}\bigl(\overline{X}-\mu\bigr)
\sim\chi^2_p.
$$

• 这里$\chi^2_p$表示自由度为$p$的卡方分布。
• 这是**精确分布**的结果，而不仅仅是渐近近似。
• 该结论也可以从**多元正态分布的性质**推导：$\overline{X}\sim N_p\bigl(\mu,\tfrac{1}{n}\Sigma\bigr)$，将其中心化并做标准化后得到一个$N_p(0,I_p)$向量，而其平方和则服从$\chi^2_p$。

当$n$足够大、$n > p$时，这个结论完全成立（即使$n$并不算“大”，只要$X_i$真正来自正态分布，也有精确分布）。

**3. 用样本协方差矩阵$S$替代$\Sigma$时的卡方近似**

在实际情况中，我们常常并不知道$\Sigma$，只能用样本协方差$S$来估计$\Sigma$。记

$$
S = \frac{1}{n-1} \sum_{i=1}^n \bigl(X_i - \overline{X}\bigr)\bigl(X_i - \overline{X}\bigr)’.
$$

于是，我们有
$$
n\bigl(\overline{X}-\mu\bigr)’S^{-1}\bigl(\overline{X}-\mu\bigr)\sim\chi^2_p (\text{approx}),
$$

当样本量$n$较大，并且$(n - p)$也较大时，这个量会**近似**服从$\chi^2_p$分布。这里的“approx”表示它是一个渐近分布或近似分布，而并非精确分布。

• 为什么是近似？因为$S$本身就是估计而非真实协方差$\Sigma$，会带来额外的随机性。
• 当$n-p$大时，$(n-1)S$的Wishart分布会越来越集中于$(n-1)\Sigma$附近，所以$S^{-1}$也能较好地近似$\Sigma^{-1}$。

**4. 整体逻辑**

1. **中心极限定理(多元)**：$\sqrt{n}(\overline{X} - \mu)$趋于$N_p(0,\Sigma)$。
2. **精确结果(正态总体)**：如果$X_i$真来自$N_p(\mu,\Sigma)$，则
$$
n(\overline{X}-\mu)’\Sigma^{-1}(\overline{X}-\mu)\sim\chi^2_p.
$$
3. **用$S$替代$\Sigma$时的卡方近似**：
$$
n(\overline{X}-\mu)’S^{-1}(\overline{X}-\mu)\sim\chi^2_p(\text{approx}),
$$

在$n$和$(n-p)$都很大时成立

这些结果在多元统计推断中非常常用：例如构造关于$\mu$的置信区间、多元假设检验(Hotelling$T^2$)等，都要用到上述卡方分布或近似。

**5. 总结**

• **当总体分布已知为多元正态**：能得到精确的分布结论；$n(\overline{X}-\mu)’\Sigma^{-1}(\overline{X}-\mu)$服从$\chi^2_p$。
• **当不确定总体是否正态，且仅知$\mu, \Sigma$有限**：可以使用大样本下的中心极限定理（多维版），得到渐近正态以及卡方近似的结论。
• **用$S$替代$\Sigma$会带来估计误差，但在大样本下依然能使用卡方分布做近似。

## 7.2 样本协方差矩阵的渐近分布 (Asymptotic Distribution of Sample Covariance Matrix)

- 假设样本协方差矩阵为 $S = \frac{1}{n-1} \sum_{j=1}^n (X_j - \bar{X})(X_j - \bar{X})'$，当 $n$ 很大时：
  - $(n-1)S$ 的分布趋于 **Wishart 分布 (Wishart Distribution)**：
    $$
    (n-1)S \sim W_p(\Sigma, n-1)
    $$

## 7.3 卡方分布性质 (Chi-square Distribution Approximation)

- 当样本量较大时，样本均值的平方形式：
  $$
  n(\bar{X} - \mu)' \Sigma^{-1} (\bar{X} - \mu) \sim \chi^2_p
  $$
  - 自由度为 $p$。
- 如果使用样本协方差矩阵 $S$ 代替总体协方差矩阵 $\\Sigma$：
  $$
  n(\bar{X} - \mu)' S^{-1} (\bar{X} - \mu) \sim \chi^2_p \ (\text{approximate for large } n)
  $$

---

# 8. Assessing the Assumption of Normality

## 8.1 单变量正态性检验 (Univariate Normality Tests)

- **目标**：
  - 检查每个变量单独是否服从正态分布。
- **方法**：
  1. **绘制直方图 (Histogram)**：
     - 绘制每个变量的频率直方图，检查分布是否对称且呈钟形。
  2. **概率图 (Probability Plot)**：
     - 生成正态概率图 (Normal Probability Plot)，查看数据点是否接近直线。
  3. **偏度与峰度检验 (Skewness and Kurtosis Test)**：
     - 偏度和峰度的统计量接近 0 表明接近正态分布。
  4. **Shapiro-Wilk 或 Kolmogorov-Smirnov 检验**：
     - 使用统计检验量化正态性。

## 8.2 多变量正态性检验 (Multivariate Normality Tests)

==了解一下就行==

**1. 为什么需要多元正态性的检验**

• **单变量正态检验不足**：我们可以对每个变量$X_1, X_2, \dots, X_p$分别做直方图或正态概率图(probability plot)，来判断它们各自的边际分布是否“看上去”接近正态。然而，这并不能保证它们“合在一起”就一定满足多元正态分布。
• **多元正态性**要求不仅各分量边际分布是正态，更要求它们的线性组合也呈正态，以及存在某种特定的协方差结构。因此需要额外的检验方法来判断是否符合**联合正态性**(joint normality)。

2. **方法**：
  1. **卡方距离 (Mahalanobis Distance)**：
     - 定义每个样本的卡方距离：
       $$
       d_j^2 = (X_j - \bar{X})'S^{-1}(X_j - \bar{X})
       $$
       其中：
       - $\bar{X}$ 是样本均值。
       - $S$ 是样本协方差矩阵。
     - $d_j^2$ 应近似服从自由度为 $p$ 的 $\chi^2_p$ 分布。
  2. **绘制卡方分布图 (Chi-square Plot)**：
     - 将卡方距离从小到大排序为 $d_{(1)}^2, d_{(2)}^2, \dots, d_{(n)}^2$。
     - 绘制卡方距离 $\sqrt{d_{(j)}^2}$ 与理论 $\chi^2_p$ 分布的分位数 $\sqrt{q_j}$ 的散点图，检查是否接近直线。

**2. Mahalanobis距离与$\chi^2_p$图**

为检验多元正态性，一种常用且直观的方法是绘制**$\chi^2$图**(也被称为**QQ图**或**chi-square plot**)。该方法基于**Mahalanobis距离**：

$$
d_j^2 ;=;(X_j - \overline{X})’S^{-1}(X_j - \overline{X}),
\quad j=1,2,\dots,n.
$$
其中
• $X_j$是第$j$个观测的$p$维向量，
• $\overline{X}$是样本均值向量，
• $S$是样本协方差矩阵(用$\tfrac{1}{n-1}$或$\tfrac{1}{n}$等因子计算，具体取决于课程/书上定义)。

如果${X_j}$真的来自$N_p(\mu,\Sigma)$，则对于足够大的样本量$n$，这些$ d_j^2 $应当大致服从$\chi^2_p$分布(这是一个近似结论；若完全正态且知道$\Sigma$，则更准确)。由此可通过比较$ d_j^2 $和$\chi^2_p$分位数之间的关系，来判断多元正态性。

  

**3. 绘制$\chi^2$图的步骤**

1. **计算$d_j^2$：**
对每一个观测$X_j$，计算
$$
d_j^2
=(X_j - \overline{X})’S^{-1}(X_j - \overline{X}),
\quad j=1,\dots,n.

$$

2. **排序**：
将这$n$个距离从小到大进行排序，得到
$$
d_{(1)}^2 \le d_{(2)}^2 \le\dots\le d_{(n)}^2.
$$

3. **计算理论分位数$q_j$**：
对$\chi^2_p$分布的分位数进行相应排序。一般会取
$$
q_j = \chi_{p,\alpha_j}^2
\quad\text{其中}\alpha_j = \frac{j - 1/2}{,n,},
$$

或者在课件中也可能定义
$$

q_j = 100 \bigl(\frac{j - 1/2}{n}\bigr)\%\text{  quantile of }\chi^2_p.

$$

这意味着$q_j$是$\chi^2_p$分布在概率点$\tfrac{j - 1/2}{n}$处的分位数。

4. **绘图**：

• **方法1**：绘制$\sqrt{,d_{(j)}^2},$对$\sqrt{,q_j},$的散点图。若$X_1,\dots,X_p$确实服从多元正态分布，则点云应大致在一条直线附近。

• **方法2**：也可以直接绘制$d_{(j)}^2$对$q_j$看是否接近对角线。

5. **判断线性程度**：

若散点图呈现接近“线性”(即在$x=y$直线或某条接近的拟合线附近)的分布，则说明数据对$\chi^2_p$分布的拟合较好，从而支持多元正态性假设；若严重偏离直线，则说明数据可能并不满足多元正态性。

**4. 常见的实践要点**

• 有时即使单变量检验都近似正态，仍有可能存在“组合效应”导致联合分布并非正态，这就是为什么要做多元$\chi^2$图。
• 当$n$和$(n-p)$比较大时，这种QQ图更具稳定性和可解释性；样本量过小或维度$p$过大时，结果会有较大变动。
• 和一维正态概率图类似，$\chi^2$图是一个**可视化**的诊断工具，它并不是万能的检验，如有必要，还可以结合其他统计检验(如Mardia’s test)或基于偏度、峰度的统计量来检验多元正态性。

**5. 总结**

• **单变量正态检验**只能说明各分量边际近似正态，无法保证**联合正态**。
• **Mahalanobis距离**$d_j^2$若数据来自多元正态，则应服从$\chi^2_p$分布(确切或近似)。
• **$\chi^2$图/QQ图**：将$\sqrt{d_{(j)}^2}$与$\sqrt{q_j}$配对作图(或直接$d_{(j)}^2$对$q_j$)，观察是否呈线性分布。若越接近直线，则多元正态性越可信。
