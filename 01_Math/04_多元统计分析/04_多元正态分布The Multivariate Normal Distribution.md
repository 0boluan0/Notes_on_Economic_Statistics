# 1. 第4章：多元正态分布（The Multivariate Normal Distribution）

>[!summary] 本章主线
> 多元正态分布是后续 Hotelling $T^2$、Wishart 分布、判别分析和多元正态性检验的基础。复习时抓住三个对象：均值向量 $\mu$、协方差矩阵 $\Sigma$、二次型 $(X-\mu)'\Sigma^{-1}(X-\mu)$。

>[!note] 课堂提示
> 旧笔记标注“证明不要求掌握，但要知道关键性质”。因此本章整理以定义、性质、公式识别和应用为主。

## 1.1. 引言

多元正态分布是单变量正态分布的高维扩展，适用于描述多个连续变量的联合分布。

它重要的原因：

1. 线性变换后仍是正态。
2. 条件分布仍是正态。
3. 二次型与卡方分布相连。
4. 样本均值和样本协方差矩阵有清晰分布。
5. 很多多元推断方法以它为小样本精确理论基础。

## 1.2. 多元正态密度及等密度曲线

### 1.2.1. 单变量正态回顾

若
$$
X\sim N(\mu,\sigma^2),
$$
密度为
$$
f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac12\left(\frac{x-\mu}{\sigma}\right)^2\right].
$$

### 1.2.2. 多元正态分布

若 $X$ 是 $p$ 维随机向量，
$$
X\sim N_p(\mu,\Sigma),
$$
其密度为
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

>[!warning] 条件
> 这里的密度公式要求 $\Sigma$ 正定，因此 $|\Sigma|>0$ 且 $\Sigma^{-1}$ 存在。

### 1.2.3. 二元正态分布

当 $p=2$ 时，
$$
\Sigma=
\begin{bmatrix}
\sigma_X^2&\rho\sigma_X\sigma_Y\\
\rho\sigma_X\sigma_Y&\sigma_Y^2
\end{bmatrix}.
$$

二元正态密度可写作
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

固定二次型
$$
(x-\mu)'\Sigma^{-1}(x-\mu)=c^2
$$
得到多元正态的等密度曲线或曲面。

几何上：

- 二维是椭圆；
- 三维是椭球；
- 高维是超椭球。

椭球中心是 $\mu$，主轴方向由 $\Sigma$ 的特征向量决定，主轴长度与特征值平方根相关。

## 1.3. 多元正态分布的性质

### 1.3.1. 线性组合

若
$$
X\sim N_p(\mu,\Sigma),
$$
且 $a$ 是 $p\times1$ 常数向量，则
$$
a'X\sim N(a'\mu,a'\Sigma a).
$$

反过来，如果任意线性组合 $a'X$ 都是一元正态，则 $X$ 是多元正态。

### 1.3.2. 仿射变换

若 $A$ 是 $q\times p$ 常数矩阵，$b$ 是 $q\times1$ 常数向量，则
$$
AX+b\sim N_q(A\mu+b,A\Sigma A').
$$

### 1.3.3. 平移

若 $d$ 是常数向量，则
$$
X+d\sim N_p(\mu+d,\Sigma).
$$

平移改变均值，不改变协方差矩阵。

### 1.3.4. 条件分布

设
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

### 1.3.5. 独立性与零协方差

在多元正态分布中：
$$
\Sigma_{12}=0
\quad\Longleftrightarrow\quad
X_1\text{ 与 }X_2\text{ 独立}.
$$

这是一条正态分布下的特殊性质。

## 1.4. 二次型与相关分布

### 1.4.1. Mahalanobis 距离

多元正态密度中的核心二次型是
$$
D^2=(X-\mu)'\Sigma^{-1}(X-\mu).
$$

它是协方差调整后的距离，称为平方 Mahalanobis 距离。

### 1.4.2. 卡方分布性质

若
$$
X\sim N_p(\mu,\Sigma),
$$
则
$$
(X-\mu)'\Sigma^{-1}(X-\mu)\sim\chi_p^2.
$$

推理直觉：令
$$
Z=\Sigma^{-1/2}(X-\mu),
$$
则
$$
Z\sim N_p(0,I),
$$
所以
$$
D^2=Z'Z=\sum_{i=1}^p Z_i^2\sim\chi_p^2.
$$

## 1.5. 多元正态分布的抽样

### 1.5.1. 联合密度

若
$$
X_1,\ldots,X_n\overset{iid}{\sim}N_p(\mu,\Sigma),
$$
则联合密度是单个密度的乘积：
$$
L(\mu,\Sigma)=\prod_{j=1}^n f(x_j;\mu,\Sigma).
$$

### 1.5.2. 最大似然估计

多元正态下：
$$
\hat\mu=\bar X.
$$

协方差矩阵的 MLE 为
$$
\hat\Sigma_{\text{MLE}}
=\frac1n\sum_{j=1}^n(X_j-\bar X)(X_j-\bar X)'.
$$

样本协方差矩阵通常定义为
$$
S=\frac1{n-1}\sum_{j=1}^n(X_j-\bar X)(X_j-\bar X)'.
$$

>[!warning] 分母区别
> MLE 用 $n$；无偏样本协方差矩阵用 $n-1$。考试中要看题目问的是 MLE 还是 sample covariance。

### 1.5.3. 不变性

若 $\hat\theta$ 是 $\theta$ 的 MLE，则 $h(\hat\theta)$ 是 $h(\theta)$ 的 MLE。

## 1.6. Wishart 分布

### 1.6.1. 定义

若 $Z_1,\ldots,Z_m$ 相互独立且
$$
Z_j\sim N_p(0,\Sigma),
$$
则
$$
\sum_{j=1}^m Z_jZ_j'\sim W_p(m,\Sigma).
$$

在多元正态随机样本中：
$$
(n-1)S\sim W_p(n-1,\Sigma).
$$

### 1.6.2. 样本均值与协方差矩阵独立

正态总体下：
$$
\bar X\ \text{与}\ S\ \text{独立}.
$$

同时
$$
\bar X\sim N_p\left(\mu,\frac1n\Sigma\right).
$$

### 1.6.3. 与卡方分布的关系

当 $p=1$ 时，Wishart 分布退化为卡方分布：
$$
(n-1)\frac{s^2}{\sigma^2}\sim\chi_{n-1}^2.
$$

## 1.7. 大样本性质

在一般条件下，样本均值有多元中心极限定理：
$$
\sqrt n(\bar X-\mu)\overset{d}{\to}N_p(0,\Sigma).
$$

若总体本身为多元正态，则这是精确结果：
$$
\bar X\sim N_p\left(\mu,\frac1n\Sigma\right).
$$

大样本下，Hotelling $T^2$ 常可近似为
$$
T^2\approx\chi_p^2.
$$

## 1.8. 正态性假设检验

### 1.8.1. 单变量检查

先对每个变量检查：

1. 直方图。
2. 箱线图。
3. 单变量 QQ 图。
4. 明显异常值。

### 1.8.2. 多变量正态性检查

计算每个观测的平方 Mahalanobis 距离：
$$
d_j^2=(X_j-\bar X)'S^{-1}(X_j-\bar X).
$$

若多元正态近似成立，$d_j^2$ 应大致符合 $\chi_p^2$ 分布。

常见做法：

1. 将 $d_j^2$ 从小到大排序。
2. 与 $\chi_p^2$ 的理论分位数作图。
3. 若点大致落在直线附近，则多元正态假设较合理。

>[!warning] 诊断边界
> 单变量正态不保证多元正态；多元异常点也可能不在任何单变量图里显得极端。

## 1.9. 关联卡片

- [[Multivariate Normal Distribution]]
- [[Bivariate Normal Distribution]]
- [[Conditional Multivariate Normal Distribution]]
- [[Mahalanobis Distance]]
- [[Wishart Distribution]]
- [[Multivariate Normality Check]]
- [[Chi-square Distribution]]
- [[Matrix Square Root]]
