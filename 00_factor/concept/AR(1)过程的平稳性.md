---
aliases:
  - AR(1) stationarity
  - AR(1) 平稳性条件
tags:
  - proof
  - 06_时间序列分析
---

# AR(1) 过程的平稳性

## 定理内容

对于 AR(1) 模型：
$$y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$$

其中 $\varepsilon_t$ 为白噪声，$E(\varepsilon_t) = 0$，$\text{Var}(\varepsilon_t) = \sigma^2$。

**平稳性充要条件**：$|a_1| < 1$

当且仅当 $|a_1| < 1$ 时，该过程是协方差平稳的，此时：
- 均值：$\mu = \frac{a_0}{1 - a_1}$
- 方差：$\text{Var}(y_t) = \frac{\sigma^2}{1 - a_1^2}$
- 自协方差：$\gamma_s = \frac{\sigma^2 a_1^s}{1 - a_1^2}$
- 自相关系数：$\rho_s = a_1^s$

## 证明思路

通过迭代求解 AR(1) 方程，将 $y_t$ 表示为过去白噪声的无限和，然后利用等比级数收敛的条件分析平稳性。

## 证明过程

### 步骤 1：迭代展开

从 $y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$ 开始，不断代入前一期：

$$
\begin{aligned}
y_t &= a_0 + a_1 y_{t-1} + \varepsilon_t \\
&= a_0 + a_1 (a_0 + a_1 y_{t-2} + \varepsilon_{t-1}) + \varepsilon_t \\
&= a_0(1 + a_1) + a_1^2 y_{t-2} + a_1 \varepsilon_{t-1} + \varepsilon_t \\
&= a_0(1 + a_1 + a_1^2) + a_1^3 y_{t-3} + a_1^2 \varepsilon_{t-2} + a_1 \varepsilon_{t-1} + \varepsilon_t \\
&\quad \vdots \\
&= a_0 \sum_{i=0}^{t-1} a_1^i + a_1^t y_0 + \sum_{i=0}^{t-1} a_1^i \varepsilon_{t-i}
\end{aligned}
$$

### 步骤 2：考虑稳定特解（忽略初始条件）

假设过程从无限远的过去开始，令 $t \to \infty$。此时 $a_1^t y_0$ 项取决于 $|a_1|$ 的值：

- **当 $|a_1| < 1$**：$a_1^t \to 0$，初始条件影响消失
- **当 $|a_1| \geq 1$**：$a_1^t$ 发散或为常数，初始条件持续影响

因此，当 $|a_1| < 1$ 时，得到稳定特解：
$$y_t = \frac{a_0}{1 - a_1} + \sum_{i=0}^{\infty} a_1^i \varepsilon_{t-i}$$

### 步骤 3：验证均值有限且为常数

对稳定特解取期望：
$$
\begin{aligned}
E[y_t] &= E\left[\frac{a_0}{1 - a_1} + \sum_{i=0}^{\infty} a_1^i \varepsilon_{t-i}\right] \\
&= \frac{a_0}{1 - a_1} + \sum_{i=0}^{\infty} a_1^i E[\varepsilon_{t-i}] \\
&= \frac{a_0}{1 - a_1}
\end{aligned}
$$

因为 $E[\varepsilon_t] = 0$，所以 $\mu = \frac{a_0}{1 - a_1}$ 为常数。

### 步骤 4：验证方差有限且为常数

计算方差：
$$
\begin{aligned}
\text{Var}(y_t) &= E[(y_t - \mu)^2] \\
&= E\left[\left(\sum_{i=0}^{\infty} a_1^i \varepsilon_{t-i}\right)^2\right] \\
&= E\left[\sum_{i=0}^{\infty} a_1^{2i} \varepsilon_{t-i}^2 + \sum_{i \neq j} a_1^{i+j} (k}\varepsilon_{t-i} \varepsilon_{t-j}\right]
\end{aligned}
$$

由于白噪声不相关，$\varepsilon_{t-i} \varepsilon_{t-j} = 0$ 当 $i \neq j$，只保留对角项：
$$
\begin{aligned}
\text{Var}(y_t) &= \sum_{i=0}^{\infty} a_1^{2i} E[\varepsilon_{t-i}^2] \\
&= \sigma^2 \sum_{i=0}^{\infty} a_1^{2i} \\
&= \frac{\sigma^2}{1 - a_1^2} \quad \text{（等比级数求和）}
\end{aligned}
$$

等比级数 $\sum_{i=0}^{\infty} a_1^{2i} = \frac{1}{1 - a_1^2}$ 收敛当且仅当 $|a_1| < 1$。

### 步骤 5：验证自协方差仅依赖于滞后

计算滞后 $s > 0$ 的自协方差：
$$
\begin{aligned}
\gamma_s &= E[(y_t - \mu)(y_{t-s} - \mu)] \\
&= E\left[\left(\sum_{i=0}^{\infty} a_1^i \varepsilon_{t-i}\right)\left(\sum_{j=0}^{\infty} a_1^j \varepsilon_{t-s-j}\right)\right]
\end{aligned}
$$

展开后，只有当 $t - i = t - s - j$，即 $i = j + s$ 时，$\varepsilon_{t-i} \varepsilon_{t-s-j} = \varepsilon_{t-s-j}^2$：
$$
\begin{aligned}
\gamma_s &= \sum_{j=0}^{\infty} a_1^{j+s} \cdot a_1^j E[\varepsilon_{t-s-j}^2] \\
&= \sigma^2 a_1^s \sum_{j=0}^{\infty} a_1^{2j} \\
&= \frac{\sigma^2 a_1^s}{1 - a_1^2}
\end{aligned}
$$

自协方差 $\gamma_s$ 只依赖于滞后 $s$，与 $t$ 无关。

### 步骤 6：计算自相关系数

$$\rho_s = \frac{\gamma_s}{\gamma_0} = \frac{\frac{\sigma^2 a_1^s}{1 - a_1^2}}{\frac{\sigma^2}{1 - a_1^2}} = a_1^s$$

## 结论

当且仅当 $|a_1| < 1$ 时，AR(1) 过程满足协方差平稳的三个条件：

1. **均值有限且为常数**：$\mu = \frac{a_0}{1 - a_1}$
2. **方差有限且为常数**：$\text{Var}(y_t) = \frac{\sigma^2}{1 - a_1^2}$
3. **自协方差仅依赖滞后**：$\gamma_s = \frac{\sigma^2 a_1^s}{1 - a_1^2}$

条件 $|a_1| < 1$ 的直观解释：
- $a_1$ 是一阶自回归系数，表示过去冲击对当前的影响权重
- $|a_1| < 1$ 确保过去冲击的影响随时间衰减（$a_1^i \to 0$）
- 这对应于特征方程 $1 - a_1 z = 0$ 的根 $z = 1/a_1$ 位于单位圆外

## 推广到 AR(p) 过程

对于一般 AR(p) 过程，平稳性条件为：**特征方程的所有根都在单位圆外**。

特征方程：$1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0$

## 相关概念
[[Yule-Walker方程]]
[[ARMA模型]]
[[差分方程]]
