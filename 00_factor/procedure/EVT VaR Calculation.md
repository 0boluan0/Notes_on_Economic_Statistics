---
aliases:
- 极值理论VaR计算
- VaR
tags:
- procedure
- 07_金融机构与风险管理
- 经济
---
# 极值理论VaR计算

## 适用场景

当需要估计高置信度（如99.9%、99.99%）的VaR或ES，而历史数据不足以直接计算分位数时使用。适用于捕捉厚尾分布的极端风险。

## 所需数据/条件

- 损失序列（或收益率转换为损失）${L_t\}_{t=1}^T$
- 阈值 $u$（通常选择经验分布95%或97.5%分位数）
- 超过阈值的样本量 $n_u = |\{t: L_t > u\}|$

## 计算步骤

### 步骤 1：选择合适阈值

根据样本量选择阈值 $u$：
- 样本量 $T$ 较小时（如500），选择 $u$ 为95%分位数
- 样本量 $T$ 大时，可选择更高分位数

**注意点**：阈值需足够高以确保尾部样本充足，又不能过高导致尾部样本过少。

### 步骤 2：提取超额损失

收集所有超过阈值的损失：
$$
\{y_i\}_{i=1}^{n_u} = \{L_t - u : L_t > u, t=1,\dots,T\}
$$

其中 $y_i$ 为超额量，总是非负值。

**注意点**：使用损失定义为正，VaR为损失的正值。

### 步骤 3：估计GPD参数

使用最大似然估计GPD参数 $\xi$（形状）和 $\beta$（尺度）：

对数似然函数：
$$
\ell(\xi,\beta) = -n_u \ln \beta - (1 + 1/\xi) \sum_{i=1}^{n_u} \ln(1 + \xi y_i / \beta)
$$

一阶条件（对 $\beta$）：
$$
\frac{\partial \ell}{\partial \beta} = 0 \Rightarrow \hat{\beta} = \bar{y} \frac{\xi}{1 + \xi}
$$

需用数值方法求解 $\xi$。

**注意点**：GPD要求 $\xi > 0$（厚尾）和 $\beta > 0$。

### 步骤 4：计算尾部概率

计算超过阈值的经验概率：
$$
\hat{p} = P(L > u) \approx n_u / T
$$

**注意点**：这是尾部概率的一致估计。

### 步骤 5：计算VaR

给定置信水平 $\alpha$，VaR定义为：

若VaR > 阈值（即 $\alpha > 1 - \hat{p}$）：
$$
\text{VaR}_{\alpha} = u + \frac{\hat{\beta}}{\hat{\xi}} \left[ \left(\frac{1-\alpha}{\hat{p}}\right)^{-\hat{\xi}} - 1 \right]
$$

若VaR ≤ 阈值（即 $\alpha \leq 1 - \hat{p}$），用其他方法（如历史模拟）。

**注意点**：此公式利用GPD外推估计高置信分位数。

### 步骤 6：计算ES（预期损失）

ES为超过VaR的平均损失：

$$
\text{ES}_{\alpha} = \frac{\text{VaR}_{\alpha}}{1 - \hat{\xi}} + \frac{\hat{\beta} - \hat{\xi} u}{1 - \hat{\xi}}
$$

当 $\xi \to 0$（指数尾部）时：
$$
\text{ES}_{\alpha} = \text{VaR}_{\alpha} + \hat{\beta}
$$

**注意点**：ES总是大于或等于VaR。

## 关键公式

**GPD累积分布函数**：
$$
F_{\text{GPD}}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}, \quad y \ge 0
$$

**GPD密度函数**：
$$
f_{\text{GPD}}(y) = \frac{1}{\beta} \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi - 1}
$$

**VaR外推公式**：
$$
\text{VaR}_{\alpha} = u + \frac{\beta}{\xi} \left[ \left(\frac{1-\alpha}{\hat{p}}\right)^{-\xi} - 1 \right]
$$

**ES公式**：
$$
\text{ES}_{\alpha} = \frac{\text{VaR}_{\alpha}}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi}
$$

**形状参数含义**：
- $\xi > 0$：厚尾（重尾），极端事件概率高
- $\xi = 0$：指数尾部
- $\xi < 0$：有限尾部（有界）

## 常见问题

1. **阈值选择**：阈值过高导致尾部样本过少，估计不稳定。
2. **形状参数估计**：小样本下$\xi$估计可能不准确。
3. **外推风险**：过度外推到远超历史范围的置信度风险较大。
4. **时间变化**：极值模型假设尾部形状参数稳定，实际可能变化。
5. **与正态比较**：正态VaR通常低估极端风险，极值理论更保守。

## 相关概念
[[VaR]]
[[历史模拟法VaR计算]]
[[GPD]]
