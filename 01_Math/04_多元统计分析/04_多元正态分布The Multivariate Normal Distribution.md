# 1. 第4章：多元正态分布（The Multivariate Normal Distribution）
<!-- bilingual-en:start -->
*Chapter 4: The Multivariate Normal Distribution*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 多元正态分布是后续 Hotelling $T^2$、Wishart 分布、判别分析和多元正态性检验的基础。复习时抓住三个对象：均值向量 $\mu$、协方差矩阵 $\Sigma$、二次型 $(X-\mu)'\Sigma^{-1}(X-\mu)$。
> <!-- bilingual-en:start -->
> The multivariate normal distribution underpins Hotelling's $T^2$, the Wishart distribution, discriminant analysis, and tests of multivariate normality. For review, focus on three objects: the mean vector $\mu$, the covariance matrix $\Sigma$, and the quadratic form $(X-\mu)'\Sigma^{-1}(X-\mu)$.
> <!-- bilingual-en:end -->

>[!note] 课堂提示
> 旧笔记标注“证明不要求掌握，但要知道关键性质”。因此本章整理以定义、性质、公式识别和应用为主。
> <!-- bilingual-en:start -->
> The original note says that proofs are not required, but the key properties must be understood. This chapter therefore emphasizes definitions, properties, recognition of formulas, and applications.
> <!-- bilingual-en:end -->

## 1.1. 引言
<!-- bilingual-en:start -->
*Introduction*
<!-- bilingual-en:end -->

多元正态分布是单变量正态分布的高维扩展，适用于描述多个连续变量的联合分布。
<!-- bilingual-en:start -->
The multivariate normal distribution is the higher-dimensional extension of the univariate normal distribution. It describes the joint distribution of several continuous variables.
<!-- bilingual-en:end -->

它重要的原因：

1. 线性变换后仍是正态。
2. 条件分布仍是正态。
3. 二次型与卡方分布相连。
4. 样本均值和样本协方差矩阵有清晰分布。
5. 很多多元推断方法以它为小样本精确理论基础。
<!-- bilingual-en:start -->
It matters because:

**1.** linear transformations remain normal;<br>
**2.** conditional distributions remain normal;<br>
**3.** quadratic forms are connected to the chi-square distribution;<br>
**4.** the sample mean and sample covariance matrix have tractable distributions;<br>
**5.** many multivariate inference procedures use it as their exact small-sample foundation.<br>
<!-- bilingual-en:end -->

## 1.2. 多元正态密度及等密度曲线
<!-- bilingual-en:start -->
*Multivariate Normal Density and Equal-Density Contours*
<!-- bilingual-en:end -->

### 1.2.1. 单变量正态回顾
<!-- bilingual-en:start -->
*Review of the Univariate Normal Distribution*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
If
<!-- bilingual-en:end -->
$$
X\sim N(\mu,\sigma^2),
$$
密度为
<!-- bilingual-en:start -->
then its density is
<!-- bilingual-en:end -->
$$
f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac12\left(\frac{x-\mu}{\sigma}\right)^2\right].
$$

### 1.2.2. 多元正态分布
<!-- bilingual-en:start -->
*The Multivariate Normal Distribution*
<!-- bilingual-en:end -->

若 $X$ 是 $p$ 维随机向量，
<!-- bilingual-en:start -->
If $X$ is a $p$-dimensional random vector,
<!-- bilingual-en:end -->
$$
X\sim N_p(\mu,\Sigma),
$$
其密度为
<!-- bilingual-en:start -->
its density is
<!-- bilingual-en:end -->
$$
f(x)=
\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}
\exp\left[
-\frac12(x-\mu)'\Sigma^{-1}(x-\mu)
\right].
$$

其中：

- $\mu$ 是 $p\times1$ 均值向量；
- $\Sigma$ 是 $p\times p$ 协方差矩阵；
- $|\Sigma|$ 控制整体体积；
- $\Sigma^{-1}$ 调整不同方向上的距离尺度。
<!-- bilingual-en:start -->
Here:

- $\mu$ is the $p\times1$ mean vector;
- $\Sigma$ is the $p\times p$ covariance matrix;
- $|\Sigma|$ controls the overall volume;
- $\Sigma^{-1}$ adjusts the distance scale in different directions.
<!-- bilingual-en:end -->

>[!attention] 条件
> 这里的密度公式要求 $\Sigma$ 正定，因此 $|\Sigma|>0$ 且 $\Sigma^{-1}$ 存在。
> <!-- bilingual-en:start -->
> This density formula requires $\Sigma$ to be positive definite, so $|\Sigma|>0$ and $\Sigma^{-1}$ exists.
> <!-- bilingual-en:end -->

### 1.2.3. 二元正态分布
<!-- bilingual-en:start -->
*The Bivariate Normal Distribution*
<!-- bilingual-en:end -->

当 $p=2$ 时，
<!-- bilingual-en:start -->
When $p=2$,
<!-- bilingual-en:end -->
$$
\Sigma=
\begin{bmatrix}
\sigma_X^2&\rho\sigma_X\sigma_Y\\
\rho\sigma_X\sigma_Y&\sigma_Y^2
\end{bmatrix}.
$$

二元正态密度可写作
<!-- bilingual-en:start -->
the bivariate normal density can be written as
<!-- bilingual-en:end -->
$$
f(x,y)=
\frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}}
\exp\left\{
-\frac{1}{2(1-\rho^2)}
\left[
\frac{(x-\mu_X)^2}{\sigma_X^2}
+\frac{(y-\mu_Y)^2}{\sigma_Y^2}
-\frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y}
\right]
\right\}.
$$

### 1.2.4. 等概率密度曲线
<!-- bilingual-en:start -->
*Equal-Density Contours*
<!-- bilingual-en:end -->

固定二次型
<!-- bilingual-en:start -->
Holding the quadratic form
<!-- bilingual-en:end -->
$$
(x-\mu)'\Sigma^{-1}(x-\mu)=c^2
$$
得到多元正态的等密度曲线或曲面。
<!-- bilingual-en:start -->
fixed gives an equal-density contour or surface of the multivariate normal distribution.
<!-- bilingual-en:end -->

几何上：

- 二维是椭圆；
- 三维是椭球；
- 高维是超椭球。
<!-- bilingual-en:start -->
Geometrically:

- in two dimensions, the contour is an ellipse;
- in three dimensions, it is an ellipsoid;
- in higher dimensions, it is a hyperellipsoid.
<!-- bilingual-en:end -->

椭球中心是 $\mu$，主轴方向由 $\Sigma$ 的特征向量决定，主轴长度与特征值平方根相关。
<!-- bilingual-en:start -->
The ellipsoid is centered at $\mu$. Its principal-axis directions are determined by the eigenvectors of $\Sigma$, and its axis lengths are related to the square roots of the corresponding eigenvalues.
<!-- bilingual-en:end -->

## 1.3. 多元正态分布的性质
<!-- bilingual-en:start -->
*Properties of the Multivariate Normal Distribution*
<!-- bilingual-en:end -->

### 1.3.1. 线性组合
<!-- bilingual-en:start -->
*Linear Combinations*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
If
<!-- bilingual-en:end -->
$$
X\sim N_p(\mu,\Sigma),
$$
且 $a$ 是 $p\times1$ 常数向量，则
<!-- bilingual-en:start -->
and $a$ is a constant $p\times1$ vector, then
<!-- bilingual-en:end -->
$$
a'X\sim N(a'\mu,a'\Sigma a).
$$

反过来，如果任意线性组合 $a'X$ 都是一元正态，则 $X$ 是多元正态。
<!-- bilingual-en:start -->
Conversely, if every linear combination $a'X$ is univariate normal, then $X$ is multivariate normal.
<!-- bilingual-en:end -->

### 1.3.2. 仿射变换
<!-- bilingual-en:start -->
*Affine Transformations*
<!-- bilingual-en:end -->

若 $A$ 是 $q\times p$ 常数矩阵，$b$ 是 $q\times1$ 常数向量，则
<!-- bilingual-en:start -->
If $A$ is a constant $q\times p$ matrix and $b$ is a constant $q\times1$ vector, then
<!-- bilingual-en:end -->
$$
AX+b\sim N_q(A\mu+b,A\Sigma A').
$$

### 1.3.3. 平移
<!-- bilingual-en:start -->
*Translations*
<!-- bilingual-en:end -->

若 $d$ 是常数向量，则
<!-- bilingual-en:start -->
If $d$ is a constant vector, then
<!-- bilingual-en:end -->
$$
X+d\sim N_p(\mu+d,\Sigma).
$$

平移改变均值，不改变协方差矩阵。
<!-- bilingual-en:start -->
A translation changes the mean but leaves the covariance matrix unchanged.
<!-- bilingual-en:end -->

### 1.3.4. 条件分布
<!-- bilingual-en:start -->
*Conditional Distributions*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
Suppose
<!-- bilingual-en:end -->
$$
\begin{bmatrix}
X_1\\X_2
\end{bmatrix}
\sim N\left(
\begin{bmatrix}
\mu_1\\\mu_2
\end{bmatrix},
\begin{bmatrix}
\Sigma_{11}&\Sigma_{12}\\
\Sigma_{21}&\Sigma_{22}
\end{bmatrix}
\right).
$$

则
<!-- bilingual-en:start -->
Then
<!-- bilingual-en:end -->
$$
X_1\mid X_2=x_2
\sim
N\left(
\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),
\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\right).
$$

>[!note] 记忆点
> 条件均值会随 $x_2-\mu_2$ 变化；条件协方差只由 $\Sigma$ 的分块决定。
> <!-- bilingual-en:start -->
> The conditional mean changes with $x_2-\mu_2$, whereas the conditional covariance depends only on the blocks of $\Sigma$.
> <!-- bilingual-en:end -->

### 1.3.5. 独立性与零协方差
<!-- bilingual-en:start -->
*Independence and Zero Covariance*
<!-- bilingual-en:end -->

在多元正态分布中：
<!-- bilingual-en:start -->
Within a multivariate normal distribution,
<!-- bilingual-en:end -->
$$
\Sigma_{12}=0
\quad\Longleftrightarrow\quad
X_1\text{ 与 }X_2\text{ 独立}.
$$

这是一条正态分布下的特殊性质。
<!-- bilingual-en:start -->
This equivalence is a special property of the normal distribution.
<!-- bilingual-en:end -->

## 1.4. 二次型与相关分布
<!-- bilingual-en:start -->
*Quadratic Forms and Related Distributions*
<!-- bilingual-en:end -->

### 1.4.1. Mahalanobis 距离
<!-- bilingual-en:start -->
*Mahalanobis Distance*
<!-- bilingual-en:end -->

多元正态密度中的核心二次型是
<!-- bilingual-en:start -->
The central quadratic form in the multivariate normal density is
<!-- bilingual-en:end -->
$$
D^2=(X-\mu)'\Sigma^{-1}(X-\mu).
$$

它是协方差调整后的距离，称为平方 Mahalanobis 距离。
<!-- bilingual-en:start -->
It is a covariance-adjusted distance known as the squared Mahalanobis distance.
<!-- bilingual-en:end -->

### 1.4.2. 卡方分布性质
<!-- bilingual-en:start -->
*Chi-Square Distribution Property*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
If
<!-- bilingual-en:end -->
$$
X\sim N_p(\mu,\Sigma),
$$
则
<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->
$$
(X-\mu)'\Sigma^{-1}(X-\mu)\sim\chi_p^2.
$$

推理直觉：令
<!-- bilingual-en:start -->
For intuition, define
<!-- bilingual-en:end -->
$$
Z=\Sigma^{-1/2}(X-\mu),
$$
则
<!-- bilingual-en:start -->
Then
<!-- bilingual-en:end -->
$$
Z\sim N_p(0,I),
$$
所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->
$$
D^2=Z'Z=\sum_{i=1}^p Z_i^2\sim\chi_p^2.
$$

## 1.5. 多元正态分布的抽样
<!-- bilingual-en:start -->
*Sampling from a Multivariate Normal Distribution*
<!-- bilingual-en:end -->

### 1.5.1. 联合密度
<!-- bilingual-en:start -->
*Joint Density*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
If
<!-- bilingual-en:end -->
$$
X_1,\ldots,X_n\overset{iid}{\sim}N_p(\mu,\Sigma),
$$
则联合密度是单个密度的乘积：
<!-- bilingual-en:start -->
then the joint density is the product of the individual densities:
<!-- bilingual-en:end -->
$$
L(\mu,\Sigma)=\prod_{j=1}^n f(x_j;\mu,\Sigma).
$$

### 1.5.2. 最大似然估计
<!-- bilingual-en:start -->
*Maximum Likelihood Estimation*
<!-- bilingual-en:end -->

多元正态下：
<!-- bilingual-en:start -->
Under multivariate normality,
<!-- bilingual-en:end -->
$$
\hat\mu=\bar X.
$$

协方差矩阵的 MLE 为
<!-- bilingual-en:start -->
the MLE of the covariance matrix is
<!-- bilingual-en:end -->
$$
\hat\Sigma_{\text{MLE}}
=\frac1n\sum_{j=1}^n(X_j-\bar X)(X_j-\bar X)'.
$$

样本协方差矩阵通常定义为
<!-- bilingual-en:start -->
whereas the sample covariance matrix is usually defined as
<!-- bilingual-en:end -->
$$
S=\frac1{n-1}\sum_{j=1}^n(X_j-\bar X)(X_j-\bar X)'.
$$

>[!attention] 分母区别
> MLE 用 $n$；无偏样本协方差矩阵用 $n-1$。考试中要看题目问的是 MLE 还是 sample covariance。
> <!-- bilingual-en:start -->
> The MLE uses $n$, whereas the unbiased sample covariance matrix uses $n-1$. In an exam, check whether the question asks for the MLE or the sample covariance.
> <!-- bilingual-en:end -->

### 1.5.3. 不变性
<!-- bilingual-en:start -->
*Invariance*
<!-- bilingual-en:end -->

若 $\hat\theta$ 是 $\theta$ 的 MLE，则 $h(\hat\theta)$ 是 $h(\theta)$ 的 MLE。
<!-- bilingual-en:start -->
If $\hat\theta$ is the MLE of $\theta$, then $h(\hat\theta)$ is the MLE of $h(\theta)$.
<!-- bilingual-en:end -->

## 1.6. Wishart 分布
<!-- bilingual-en:start -->
*The Wishart Distribution*
<!-- bilingual-en:end -->

### 1.6.1. 定义
<!-- bilingual-en:start -->
*Definition*
<!-- bilingual-en:end -->

若 $Z_1,\ldots,Z_m$ 相互独立且
<!-- bilingual-en:start -->
If $Z_1,\ldots,Z_m$ are mutually independent and
<!-- bilingual-en:end -->
$$
Z_j\sim N_p(0,\Sigma),
$$
则
<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->
$$
\sum_{j=1}^m Z_jZ_j'\sim W_p(m,\Sigma).
$$

在多元正态随机样本中：
<!-- bilingual-en:start -->
For a random sample from a multivariate normal distribution,
<!-- bilingual-en:end -->
$$
(n-1)S\sim W_p(n-1,\Sigma).
$$

### 1.6.2. 样本均值与协方差矩阵独立
<!-- bilingual-en:start -->
*Independence of the Sample Mean and Sample Covariance Matrix*
<!-- bilingual-en:end -->

正态总体下：
<!-- bilingual-en:start -->
For a normal population,
<!-- bilingual-en:end -->
$$
\bar X\ \text{与}\ S\ \text{独立}.
$$

同时
<!-- bilingual-en:start -->
and
<!-- bilingual-en:end -->
$$
\bar X\sim N_p\left(\mu,\frac1n\Sigma\right).
$$

### 1.6.3. 与卡方分布的关系
<!-- bilingual-en:start -->
*Relationship to the Chi-Square Distribution*
<!-- bilingual-en:end -->

当 $p=1$ 时，Wishart 分布退化为卡方分布：
<!-- bilingual-en:start -->
When $p=1$, the Wishart distribution reduces to a chi-square distribution:
<!-- bilingual-en:end -->
$$
(n-1)\frac{s^2}{\sigma^2}\sim\chi_{n-1}^2.
$$

## 1.7. 大样本性质
<!-- bilingual-en:start -->
*Large-Sample Properties*
<!-- bilingual-en:end -->

在一般条件下，样本均值有多元中心极限定理：
<!-- bilingual-en:start -->
Under general conditions, the sample mean satisfies a multivariate central limit theorem:
<!-- bilingual-en:end -->
$$
\sqrt n(\bar X-\mu)\overset{d}{\to}N_p(0,\Sigma).
$$

若总体本身为多元正态，则这是精确结果：
<!-- bilingual-en:start -->
If the population itself is multivariate normal, this becomes an exact result:
<!-- bilingual-en:end -->
$$
\bar X\sim N_p\left(\mu,\frac1n\Sigma\right).
$$

大样本下，Hotelling $T^2$ 常可近似为
<!-- bilingual-en:start -->
In large samples, Hotelling's $T^2$ can often be approximated by
<!-- bilingual-en:end -->
$$
T^2\approx\chi_p^2.
$$

## 1.8. 正态性假设检验
<!-- bilingual-en:start -->
*Assessing the Normality Assumption*
<!-- bilingual-en:end -->

### 1.8.1. 单变量检查
<!-- bilingual-en:start -->
*Univariate Checks*
<!-- bilingual-en:end -->

先对每个变量检查：

1. 直方图。
2. 箱线图。
3. 单变量 QQ 图。
4. 明显异常值。
<!-- bilingual-en:start -->
Begin by checking each variable with:

**1.** a histogram;<br>
**2.** a box plot;<br>
**3.** a univariate QQ plot;<br>
**4.** a check for conspicuous outliers.<br>
<!-- bilingual-en:end -->

### 1.8.2. 多变量正态性检查
<!-- bilingual-en:start -->
*Multivariate Normality Checks*
<!-- bilingual-en:end -->

计算每个观测的平方 Mahalanobis 距离：
<!-- bilingual-en:start -->
Compute the squared Mahalanobis distance for each observation:
<!-- bilingual-en:end -->
$$
d_j^2=(X_j-\bar X)'S^{-1}(X_j-\bar X).
$$

若多元正态近似成立，$d_j^2$ 应大致符合 $\chi_p^2$ 分布。
<!-- bilingual-en:start -->
If multivariate normality is a reasonable approximation, the $d_j^2$ values should approximately follow a $\chi_p^2$ distribution.
<!-- bilingual-en:end -->

常见做法：

1. 将 $d_j^2$ 从小到大排序。
2. 与 $\chi_p^2$ 的理论分位数作图。
3. 若点大致落在直线附近，则多元正态假设较合理。
<!-- bilingual-en:start -->
A common procedure is to:

**1.** sort the $d_j^2$ values from smallest to largest;<br>
**2.** plot them against the theoretical quantiles of $\chi_p^2$;<br>
**3.** regard multivariate normality as more plausible when the points lie approximately on a straight line.<br>
<!-- bilingual-en:end -->

>[!attention] 诊断边界
> 单变量正态不保证多元正态；多元异常点也可能不在任何单变量图里显得极端。
> <!-- bilingual-en:start -->
> Univariate normality does not guarantee multivariate normality, and a multivariate outlier may not look extreme in any individual variable plot.
> <!-- bilingual-en:end -->

## 1.9. 关联卡片
<!-- bilingual-en:start -->
*Related Cards*
<!-- bilingual-en:end -->

- [[多元正态分布#密度与椭球几何|Multivariate Normal Distribution]]
- [[多元正态分布#边际、条件与独立|Bivariate Normal Distribution]]
- [[多元正态分布#边际、条件与独立|Conditional Multivariate Normal Distribution]]
- [[多元数据、随机向量与样本协方差#距离与几何|Mahalanobis Distance]]
- [[Wishart 分布与样本协方差推断|Wishart Distribution]]
- [[多元正态分布#正态性诊断|Multivariate Normality Check]]
- [[多元正态分布#二次型与卡方|Chi-square Distribution]]
- [[对称矩阵与正定二次型#二次型与正定性|Matrix Square Root]]
