---
aliases:
- Yule-Walker 方程
- Yule-Walker方程
- Yule
- Yule-Walker equations
tags:
- proof
- 06_时间序列分析
- concept
---
# Yule-Walker 方程

## 定理内容

对于平稳的 AR(p) 模型：
$$y_t = \mu + \sum_{i=1}^p \phi_i y_{t-i} + \varepsilon_t$$

其中 $\varepsilon_t$ 为白噪声，$E(\varepsilon_t) = 0$，$\text{Var}(\varepsilon_t) = \sigma^2$。自协方差函数 $\gamma_k = \text{Cov}(y_t, y_{t-k})$ 满足：
$$\gamma_k = \sum_{i=1}^p \phi_i \gamma_{k-i}, \quad k \geq 1$$

对应的自相关函数 $\rho_k$ 满足：
$$\rho_k = \sum_{i=1}^p \phi_i \rho_{k-i}, \quad k \geq 1$$

对于 $k = 0$：
$$\gamma_0 = \sum_{i=1}^p \phi_i \gamma_i + \sigma^2$$

## 证明思路

核心思想是将 AR(p) 方程两边同时乘以 $y_{t-k}$，然后取期望。关键在于利用白噪声 $\varepsilon_t$ 与过去观测值 $y_{t-k}$ 的不相关性。

## 证明过程

### 步骤 1：建立基本方程

从 AR(p) 模型出发，将方程两边同时乘以 $y_{t-k}$（$k \geq 0$）：
$$y_t \cdot y_{t-k} = \mu \cdot y_{t-k} + \sum_{i=1}^p \phi_i y_{t-i} \cdot y_{t-k} + \varepsilon_t \cdot y_{t-k}$$

### 步骤 2：取期望

对两边取期望：
$$E[y_t y_{t-k}] = \mu E[y_{t-k}] + \sum_{i=1}^p \phi_i E[y_{t-i} y_{t-k}] + E[\varepsilon_t y_{t-k}]$$

### 步骤 3：利用平稳性条件

由于过程平稳，均值为常数 $\mu = E[y_t]$，且自协方差仅依赖于滞后 $k$：

- $E[y_t y_{t-k}] = \gamma_k$
- $E[y_{t-i} y_{t-k}] = \gamma_{k-i}$
- $E[y_{t-k}] = \mu$

### 步骤 4：处理噪声项

关键步骤：分析 $E[\varepsilon_t y_{t-k}]$

当 $k = 0$ 时：
$$E[\varepsilon_t y_t] = E[\varepsilon_t (\mu + \sum_{i=1}^p \phi_i y_{t-i} + \varepsilon_t)] = E[\varepsilon_t^2] = \sigma^2$$

当 $k > 0$ 时：
由于 $\varepsilon_t$ 只影响 $y_t, y_{t+1}, \ldots$，而 $y_{t-k}$ 只依赖于 $\varepsilon_{t-k}, \varepsilon_{t-k-1}, \ldots$，两者不相关：
$$E[\varepsilon_t y_{t-k}] = 0$$

### 步骤 5：得到 Yule-Walker 方程

**对于 $k = 0$：**
$$\gamma_0 = \mu^2 + \sum_{i=1}^p \phi_i \gamma_i + \sigma^2$$

若已中心化为零均值（$\mu = 0$）：
$$\gamma_0 = \sum_{i=1}^p \phi_i \gamma_i + \sigma^2$$

**对于 $k \geq 1$：**
$$\gamma_k = \mu^2 + \sum_{i=1}^p \phi_i \gamma_{k-i}$$

若 $\mu = 0$：
$$\gamma_k = \sum_{i=1}^p \phi_i \gamma_{k-i}$$

### 步骤 6：转换为自相关系数

定义 $\rho_k = \frac{\gamma_k}{\gamma_0}$，将方差归一化：
$$\rho_k = \sum_{i=1}^p \phi_i \rho_{k-i}, \quad k = 1, 2, \ldots$$

## 结论

Yule-Walker 方程建立了 AR 参数 $\{\phi_i\}$ 与自相关系数 $\{\rho_k\}$ 之间的递推关系。这个关系可用于：

1. **已知 AR 参数求 ACF**：通过递推方程计算各阶自相关
2. **参数估计**：用样本自相关 $\hat{\rho}_k$ 替代理论 $\rho_k$，求解线性方程组得到参数估计
3. **模型识别**：根据 ACF 的递推模式判断 AR 阶数

## 关键条件

1. **平稳性**：过程必须平稳，否则自协方差定义不随时间平移
2. **零均值或中心化**：公式简化通常假设零均值，否则需要考虑均值项
3. **白噪声不相关**：$\varepsilon_t$ 与所有过去 $y_{t-k}$ 不相关

## 相关概念
[[00_factor/concept/ARMA|ARMA模型]]
[[00_factor/concept/Autocorrelation Function|自相关函数 ACF]]
[[00_factor/concept/White Noise|白噪声过程]]
