---
aliases:
- 中心极限定理
- CLT
tags:
- proof
- 数学基础
- concept
---
# 中心极限定理（CLT）

## 定理内容

**林德伯格-列维（Lindberg-Lévy）中心极限定理**：

设 $\{X_n\}_{n=1}^{\infty}$ 为独立同分布（i.i.d.）随机变量序列，满足：
$$E[X_n] = \mu < \infty, \quad \text{Var}(X_n) = \sigma^2 < \infty$$

定义样本均值：
$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$$

则标准化样本均值依分布收敛到标准正态分布：
$$\sqrt{n}\left(\frac{\bar{X}_n - \mu}{\sigma}\right) \xrightarrow{d} N(0, 1)$$

等价地：
$$\bar{X}_n \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right)$$

**矩阵形式的 CLT**：

设 $\{X_i\}_{i=1}^n$ 为独立同分布 $k \times 1$ 随机向量序列，满足：
$$E[X_i] = \mu, \quad \text{Var}(X_i) = \Sigma \text{（有限正定矩阵）}$$

则：
$$\sqrt{n}\left(\frac{1}{n}\sum_{i=1}^n X_i - \mu\right) \xrightarrow{d} N(0, \Sigma)$$

## 证明思路

CLT 的证明比较复杂，通常使用**特征函数（Characteristic Function）**方法或**矩生成函数（MGF）**方法。核心思想是将独立随机变量之和的特征函数分解，利用泰勒展开和对数近似，证明极限特征函数对应正态分布。

## 证明过程（特征函数法）

### 步骤 1：定义标准化变量

设 $Y_i = \frac{X_i - \mu}{\sigma}$，则 $E[Y_i] = 0$，$\text{Var}(Y_i) = 1$。

定义 $S_n = \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i$，即标准化的样本均值。

### 步骤 2：特征函数的定义

随机变量 $Z$ 的特征函数为：
$$\phi_Z(t) = E[e^{itZ}]$$

### 步骤 3：计算 $S_n$ 的特征函数

由于 $Y_i$ 独立同分布，其特征函数相同：
$$\phi_{Y_i}(t) = \phi_Y(t)$$

$S_n$ 的特征函数：
$$
\begin{aligned}
\phi_{S_n}(t)
&= E\left[e^{it S_n}\right] \\
&= E\left[\exp\left(it \cdot \frac{1}{\sqrt{n}} \sum_{i=1}^n Y_i\right)\right] \\
&= \prod_{i=1}^n E\left[\exp\left(it \cdot \frac{1}{\sqrt{n}} Y_i\right)\right] \quad \text{（独立性）} \\
&= \left[\phi_Y\left(\frac{t}{\sqrt{n}}\right)\right]^n
\end{aligned}
$$

### 步骤 4：泰勒展开特征函数

对 $\phi_Y\left(\frac{t}{\sqrt{n}}\right)$ 在 $0$ 处进行二阶泰勒展开：
$$\phi_Y(s) \approx \phi_Y(0) + \phi_Y'(0)s + \frac{1}{2}\phi_Y''(0)s^2 + o(s^2)$$

其中：
- $\phi_Y(0) = E[e^{i \cdot 0 \cdot Y}] = 1$
- $\phi_Y'(0) = iE[Y] = 0$
- $\phi_Y''(0) = i^2 E[Y^2] = -E[Y^2] = -1$（因为 $\text{Var}(Y) = 1$ 且 $E[Y] = 0$）

因此：
$$\phi_Y\left(\frac{t}{\sqrt{n}}\right) \approx 1 + 0 \cdot \frac{t}{\sqrt{n}} + \frac{1}{2}(-1)\left(\frac{t}{\sqrt{n}}\right)^2 = 1 - \frac{t^2}{2n}$$

### 步骤 5：取对数并求极限

$$
\begin{aligned}
\ln \phi_{S_n}(t)
&= n \ln \phi_Y\left(\frac{t}{\sqrt{n}}\right) \\
&\approx n \ln\left(1 - \frac{t^2}{2n}\right) \\
&\approx n \left(-\frac{t^2}{2n}\right) \quad \text{（利用 $\ln(1+x) \approx x$ 当 $x$ 很小时）} \\
&= -\frac{t^2}{2}
\end{aligned}
$$

因此：
$$\phi_{S_n}(t) \approx \exp\left(-\frac{t^2}{2}\right)$$

### 步骤 6：识别极限分布

$\exp\left(-\frac{t^2}{2}\right)$ 正是**标准正态分布 $N(0, 1)$** 的特征函数。

由**连续映射定理**或特征函数的唯一性定理：
$$S_n \xrightarrow{d} N(0, 1)$$

### 步骤 7：回代到原始变量

$$S_n = \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i = \frac{1}{\sqrt{n}}\sum_{i=1}^n \frac{X_i - \mu}{\sigma} = \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}}$$

因此：
$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

等价于：
$$\sqrt{n}\left(\frac{\bar{X}_n - \mu}{\sigma}\right) \xrightarrow{d} N(0, 1)$$

## 结论

中心极限定理保证了大量独立随机变量之和（或均值）渐近服从正态分布，无论单个变量服从什么分布（只要有限均值和方差）。

**直观意义**：

1. **正态分布的普适性**：许多自然和社会现象之所以服从正态分布，是因为它们可以看作大量独立微小因素的叠加
2. **大样本推断**：即使总体分布未知，在大样本下仍可使用正态分布进行统计推断
3. **近似误差**：样本量 $n$ 越大，正态近似越精确

**实际应用**：

1. **均值检验**：$t$ 检验依赖于 CLT
2. **回归系数检验**：OLS 系数的渐近正态性
3. **置信区间构造**：大样本下使用正态分位数

**CLT 的条件**：

1. **独立性**：随机变量相互独立（或弱独立）
2. **同分布**：或至少方差有限
3. **有限矩**：$E[X^2] < \infty$

**非独立同分布的扩展**：

- **林德伯格条件**：处理非同分布情况
- **鞅差中心极限定理**：处理鞅差序列
- **平稳过程 CLT**：处理弱相关序列

## 相关概念
[[大数定律]]
[[OLS估计量的一致性]]
[[渐近理论]]
