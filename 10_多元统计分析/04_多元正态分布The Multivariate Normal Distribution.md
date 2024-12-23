
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
(X - \mu)’,\Sigma^{-1},(X - \mu) = c^2
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









**(d) 分块向量、条件分布与独立性**

  











--------

# 3. Properties of Multivariate Normal Distribution

**前置示例设定**

我们先给出一个**二维**（即 $p=2$）的随机向量 $X$：

$$
X
= \begin{pmatrix} X_1 \ X_2 \end{pmatrix}
\sim N_2!\Bigl(
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
a’X \sim N(7,;8).
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
AX \sim N_2(A\mu,; A\Sigma A’).
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
N_p!\Bigl(
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

**结论 ** 涉及到两件事：

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

Z \sim N_p\bigl(0,;I_p\bigr),

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

----------------------

# 6. Wishart Distribution

==如果总体是正态分布,那么样本就是Wishart 分布==

## 6.1 定义 (Definition)
- **Wishart 分布** 是样本协方差矩阵的分布。
- 假设 $X_1, X_2, \dots, X_n$ 是从 $N_p(\mu, \Sigma)$ 独立抽取的样本，定义样本协方差矩阵：
  $$
  S = \frac{1}{n} \sum_{j=1}^n (X_j - \bar{X})(X_j - \bar{X})'
  $$
  如果定义：
  $$
  W = nS = \sum_{j=1}^n (X_j - \bar{X})(X_j - \bar{X})'
  $$
  则 $W$ 服从 **Wishart 分布**：
  $$
  W \sim W_p(\Sigma, n-1)
  $$
  其中：
  - $W_p(\Sigma, m)$ 表示自由度为 $m$ 且尺度矩阵为 $\\Sigma$ 的 Wishart 分布。

## 6.2 性质 (Properties)
1. **加和性质**：
   - 如果 $W_1 \sim W_p(\Sigma, m_1)$ 且 $W_2 \sim W_p(\Sigma, m_2)$，且 $W_1$ 和 $W_2$ 独立，则：
     $$
     W_1 + W_2 \sim W_p(\Sigma, m_1 + m_2)
     $$

2. **线性变换性质**：
   - 如果 $W \sim W_p(\Sigma, m)$，且 $C$ 是 $q \times p$ 的非奇异矩阵，则：
     $$
     CWC' \sim W_q(C\Sigma C', m)
     $$

3. **样本协方差矩阵的性质**：
   - 样本协方差矩阵 $S = \frac{1}{n-1}W$，其分布可以表示为：
     $$
     S \sim \frac{1}{n-1} W_p(\Sigma, n-1)
     $$
     
---

# 7. Large Sample Properties

## 7.1 中心极限定理 (Central Limit Theorem)

- **描述**：
  - 中心极限定理 (CLT) 描述了大样本情况下样本均值的分布特性。
  - 假设 $X_1, X_2, \dots, X_n$ 是从任意分布中独立抽取的随机样本，其总体均值为 $\\mu$，协方差矩阵为 $\\Sigma$。
- **性质**：
  - 当样本量 $n$ 很大时，样本均值的分布趋于正态：
    $$
    \sqrt{n}(\bar{X} - \mu) \xrightarrow{d} N_p(0, \Sigma)
    $$
    其中：
    - $\\bar{X}$ 是样本均值。
    - $\\xrightarrow{d}$ 表示渐近分布。

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

## 7.4 应用 (Applications)
1. **假设检验**：
   - 检验样本均值是否等于某一假设值。
   - 检验多元正态性假设。
2. **参数估计的置信区间**：
   - 利用渐近正态分布特性构造均值和协方差矩阵的置信区间。

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

- **目标**：
  - 检验整个数据集是否服从联合多元正态分布。
- **方法**：
  1. **卡方距离 (Mahalanobis Distance)**：
     - 定义每个样本的卡方距离：
       $$
       d_j^2 = (X_j - \bar{X})'S^{-1}(X_j - \bar{X})
       $$
       其中：
       - $\\bar{X}$ 是样本均值。
       - $S$ 是样本协方差矩阵。
     - $d_j^2$ 应近似服从自由度为 $p$ 的 $\\chi^2_p$ 分布。
  2. **绘制卡方分布图 (Chi-square Plot)**：
     - 将卡方距离从小到大排序为 $d_{(1)}^2, d_{(2)}^2, \dots, d_{(n)}^2$。
     - 绘制卡方距离 $\sqrt{d_{(j)}^2}$ 与理论 $\chi^2_p$ 分布的分位数 $\sqrt{q_j}$ 的散点图，检查是否接近直线。
## 8.3 注意事项 (Cautions)

1. **单变量正态性不保证联合正态性**：
   - 即使每个变量单独服从正态分布，它们的联合分布可能不是多元正态分布。
2. **样本量影响**：
   - 小样本可能导致正态性检验结果的不稳定。
   - 大样本中，即使存在偏差，数据也可能被误认为正态分布。
