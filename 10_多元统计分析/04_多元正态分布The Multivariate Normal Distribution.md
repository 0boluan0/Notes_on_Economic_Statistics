
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

## 1.2 使用场景：多元正态分布的假设和适用情境
- **统计建模 (Statistical Modeling)**：
  - 多元正态分布广泛用于建模包含多个相关变量的数据。
  - 例如，在金融领域，资产收益率 (Asset Returns) 的建模通常涉及资产间的相关性 (Correlation)。

- **降维 (Dimensionality Reduction)**：
  - 像 **主成分分析 (PCA)** 这样的技术，假设数据可以投影到低维空间，同时保留其结构，这通常基于多元正态分布的假设。

- **假设检验 (Hypothesis Testing)**：
  - 许多多元假设检验方法，例如 **多元方差分析 (Multivariate Analysis of Variance, MANOVA)**，要求数据遵循多元正态分布。

- **预测建模 (Predictive Modeling)**：
  - 概率模型，如 **高斯过程 (Gaussian Processes)** 或 **贝叶斯网络 (Bayesian Networks)**，经常使用多元正态分布来实现计算上的可行性和解释性。

- **实际案例 (Real-world Example)**：
  - 在环境研究中，不同地点的污染物浓度可能遵循多元正态分布，因为它们共享污染源并受大气混合的影响。

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
    - $\\mu$ 是均值 (Mean)，表示分布的中心。
    - $\\sigma^2$ 是方差 (Variance)，描述分布的宽度。
- **概率密度函数 (Probability Density Function, PDF)**：
  - 单变量正态分布的 PDF 表达式为：
    $$
    f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{1}{2} \left( \frac{x-\mu}{\sigma} \right)^2}, \quad -\infty < x < \infty
    $$
    该函数具有以下性质：
    1. 对称性：以 $\\mu$ 为中心对称。
    2. 范围：概率密度函数的定义域为 $(-\\infty, \\infty)$。
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
    1. 如果协方差矩阵 $\Sigma$ 是正定矩阵 (Positive Definite Matrix)，则 PDF 为正常定义。
    2. 分布中心由均值向量 $\\mu$ 确定。

### 2.2.2 特例：二维正态分布 (Special Case: Bivariate Normal Distribution)

- 当 $p = 2$ 时，$X = (X_1, X_2)'$ 的多元正态分布公式可以写为：
  $$
  f(X_1, X_2) = \frac{1}{2\pi|\Sigma|^{1/2}} e^{-\frac{1}{2} \begin{pmatrix} X_1 - \mu_1 \\ X_2 - \mu_2 \end{pmatrix}' \Sigma^{-1} \begin{pmatrix} X_1 - \mu_1 \\ X_2 - \mu_2 \end{pmatrix}}
  $$
- 其中协方差矩阵：
  $$
  \Sigma = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix}
  $$
  这里：
  - $\rho$ 是两个随机变量之间的相关系数 (Correlation Coefficient)。
  - $\sigma_1$ 和 $\sigma_2$ 是 $X_1$ 和 $X_2$ 的标准差 (Standard Deviation)。

## 2.3 等概率密度曲线 (Equi-probability Density Contours)

- **定义 (Definition)**：
  - 等概率密度曲线是多元正态分布概率密度函数为常数的路径，由以下方程定义：
    $$
    (X - \mu)'\Sigma^{-1}(X - \mu) = c^2
    $$
  - 该方程表示的是随机向量 $X$ 的所有可能取值中具有相同密度值的集合。
- **椭圆性质 (Elliptical Properties)**：
  1. 曲线的形状是椭圆 (Ellipse)，其中心位于均值向量 $\\mu$。
  2. 椭圆的方向和大小由协方差矩阵 $\\Sigma$ 决定：
     - 主轴方向对应 $\\Sigma$ 的特征向量 (Eigenvectors)。
     - 主轴长度与特征值 (Eigenvalues) 成正比。
  3. 当协方差矩阵为对角矩阵时，等概率密度曲线为圆形。

--------

# 3. Properties of Multivariate Normal Distribution

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

## 3.2 仿射变换 (Affine Transformations)
- 若 $X \sim N_p(\mu, \Sigma)$，$A$ 是 $q \times p$ 的矩阵，则仿射变换 $AX$ 的分布为：
  $$
  AX \sim N_q(A\mu, A\Sigma A')
  $$

## 3.3 平移 (Translation)
- 若 $X \sim N_p(\mu, \Sigma)$ 且 $d$ 是 $p \times 1$ 的常量向量，则平移后的分布为：
  $$
  X + d \sim N_p(\mu + d, \Sigma)
  $$

## 3.4 条件分布 (Conditional Distributions)
### 3.4.1 条件分布定义
- 若随机向量 $X$ 可以分解为 $X = \begin{pmatrix} X_1 \\ X_2 \end{pmatrix}$，其中：
  - $X_1$ 是 $q \times 1$ 的随机向量，
  - $X_2$ 是 $(p-q) \times 1$ 的随机向量，
  - 均值向量 $\\mu$ 和协方差矩阵 $\\Sigma$ 分别为：
    $$
    \mu = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \quad
    \Sigma = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}
    $$
- 则 $X_1 | X_2 \sim N_q(\mu_{1|2}, \Sigma_{1|2})$，其中：
  - 条件均值 (Conditional Mean)：
    $$
    \mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(X_2 - \mu_2)
    $$
  - 条件协方差 (Conditional Covariance)：
    $$
    \Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
    $$

### 3.4.2 条件独立性
- 如果 $\\Sigma_{12} = 0$，则 $X_1$ 和 $X_2$ 是独立的，且：
  $$
  X_1 \perp X_2 \implies X_1 \sim N_q(\mu_1, \Sigma_{11}), \quad X_2 \sim N_{p-q}(\mu_2, \Sigma_{22})
  $$

---------------

# 4. Quadratic Forms and Related Distributions

## 4.1 平方形式的分布 (Distribution of Quadratic Forms)

- 若随机向量 $X \sim N_p(\mu, \Sigma)$，定义平方形式：
  $$
  D^2 = (X - \mu)'\Sigma^{-1}(X - \mu)
  $$
  则 $D^2$ 服从 **卡方分布 (Chi-square Distribution)**，其自由度为 $p$：
  $$
  D^2 \sim \chi^2_p
  $$

- **推导**：
  - 通过将 $X$ 标准化，即令 $Z = \Sigma^{-1/2}(X - \mu)$，则：
    $$
    Z \sim N_p(0, I)
    $$
  - 平方形式可以写为：
    $$
    D^2 = Z'Z = Z_1^2 + Z_2^2 + \dots + Z_p^2
    $$
    其中每个 $Z_i \sim N(0, 1)$。

## 4.2 标准化 (Standardization)

- **定义**：
  - 通过以下变换对 $X$ 进行标准化：
    $$
    Z = \Sigma^{-1/2}(X - \mu)
    $$
    其中 $\\Sigma^{-1/2}$ 是协方差矩阵 $\\Sigma$ 的平方根逆矩阵。
- **标准化后的性质**：
  1. $Z$ 服从标准正态分布 (Standard Normal Distribution)：
     $$
     Z \sim N_p(0, I)
     $$
     其中 $I$ 是 $p \\times p$ 的单位矩阵。
  2. 标准化后的变量之间独立。

- **应用**：
  - 标准化是验证多元正态分布假设和简化计算的重要步骤。

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
  - 总体均值为 $\\mu$，协方差矩阵为 $\\Sigma$。

## 5.2 最大似然估计 (Maximum Likelihood Estimation)

- 假设样本 $X_1, X_2, \dots, X_n$ 来自多元正态分布 $N_p(\mu, \Sigma)$，可以对均值向量和协方差矩阵进行最大似然估计 (MLE)：
  - **均值向量的估计**：
    $$
    \hat{\mu} = \bar{X} = \frac{1}{n} \sum_{j=1}^n X_j
    $$
    其中 $\\bar{X}$ 是样本均值。
  - **协方差矩阵的估计**：
    $$
    \hat{\Sigma} = \frac{1}{n} \sum_{j=1}^n (X_j - \bar{X})(X_j - \bar{X})'
    $$

- **样本协方差矩阵的性质**：
  1. 如果 $n > p$，样本协方差矩阵是非奇异的。
  2. 如果 $n \leq p$，样本协方差矩阵可能是奇异的，无法直接求逆。

- **总结**：
  最大似然估计提供了多元正态分布参数的有效估计，对后续的统计分析和建模至关重要。

----------------------

# 6. Wishart Distribution

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
     
## 6.3 应用 (Applications)
- **多元正态分布假设检验**：
  - Wishart 分布用于检验样本是否来源于多元正态分布。
- **多元回归分析**：
  - 在多元回归中，协方差矩阵的分布依赖于 Wishart 分布。
- **贝叶斯分析**：
  - Wishart 分布常被用作协方差矩阵的先验分布 (Prior Distribution)。

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

- 假设样本协方差矩阵为 $S = \\frac{1}{n-1} \\sum_{j=1}^n (X_j - \\bar{X})(X_j - \\bar{X})'$，当 $n$ 很大时：
  - $(n-1)S$ 的分布趋于 **Wishart 分布 (Wishart Distribution)**：
    $$
    (n-1)S \sim W_p(\\Sigma, n-1)
    $$

## 7.3 卡方分布性质 (Chi-square Distribution Approximation)

- 当样本量较大时，样本均值的平方形式：
  $$
  n(\bar{X} - \mu)' \\Sigma^{-1} (\bar{X} - \mu) \sim \chi^2_p
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
     - 绘制卡方距离 $\\sqrt{d_{(j)}^2}$ 与理论 $\\chi^2_p$ 分布的分位数 $\\sqrt{q_j}$ 的散点图，检查是否接近直线。
## 8.3 注意事项 (Cautions)

1. **单变量正态性不保证联合正态性**：
   - 即使每个变量单独服从正态分布，它们的联合分布可能不是多元正态分布。
2. **样本量影响**：
   - 小样本可能导致正态性检验结果的不稳定。
   - 大样本中，即使存在偏差，数据也可能被误认为正态分布。
