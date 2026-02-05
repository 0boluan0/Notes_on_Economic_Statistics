---
aliases:
- 方差-协方差法VaR
- VaR参数法计算
- VaR Parameter Calculation
- VaR
- VaR Parametric Method
tags:
- procedure
- 07_金融机构与风险管理
---
# VaR参数法计算

## 适用场景

当资产或投资组合收益（或损失）服从正态分布，且组合主要由线性资产（股票、债券等）构成时使用。适合日常风险监控，计算快速简便。

## 所需数据/条件

- 资产/组合当前市值 $V$
- 资产/组合日波动率（标准差）$\sigma$
- 期望收益率 $\mu$（通常近似为0）
- 置信水平 $\alpha$（如95%、99%）
- 持有期 $T$（天数）
- 相关系数矩阵（多资产组合）

## 计算步骤

### 步骤 1：获取基础参数

收集所需的基础数据：
- 单一资产：市值 $V$、日波动率 $\sigma$
- 多资产组合：各资产市值 V_i、波动率 \sigma_i、相关系数 \rho_{ij}

**注意点**：短期常用 $\mu \approx 0$，长期需考虑期望收益。

### 步骤 2：计算组合标准差

**单一资产**：
$\sigma_p = \sigma \times V$

**多资产组合**：
$\sigma_p^2 = \sum_{i=1}^{n}\sum_{j=1}^{n} V_i V_j \sigma_i \sigma_j \rho_{ij}$
$\sigma_p = \sqrt{\sigma_p^2}$

**注意点**：确保协方差矩阵正定，避免完全多重共线性。

### 步骤 3：确定置信水平分位数

根据置信水平查标准正态分布表：
- $\alpha = 95\% \Rightarrow z_{0.95} \approx 1.645$
- $\alpha = 99\% \Rightarrow z_{0.99} \approx 2.33$
- $\alpha = 97.5\% \Rightarrow z_{0.975} \approx 1.96$

### 步骤 4：计算单日VaR

$\text{VaR}_{1天} = -(\mu + z_{\alpha}\sigma_p)$

当 $\mu \approx 0$ 时：
$ \text{VaR}_{1天} \approx z_{\alpha}\sigma_p $

**注意点**：VaR取正值，表示最大可能损失金额。

### 步骤 5：调整持有期

当持有期 $T > 1$ 天时：
$ \text{VaR}_{T天} = \text{VaR}_{1天} \times \sqrt{T} $

**注意点**：此公式假设每日收益独立同分布。若存在自相关，需修正：
$ \sigma_{T} = \sqrt{\sigma^2(T + 2\sum_{k=1}^{T-1}(T-k)\rho^k)} $

### 步骤 6：处理非线性资产（可选）

对于期权等衍生品，使用Delta近似：
$ \Delta P \approx \delta \times \Delta S $
将期权头寸等效为 $\delta$ 股标的资产，重复步骤2-5。

更精确可使用Delta-Gamma-Cornish Fisher展开。

## 关键公式

**单资产单日VaR**：
$ \text{VaR}_{\alpha} = z_{\alpha} \times \sigma \times V $

**双资产组合标准差**：
$\sigma_p = \sqrt{(V_1\sigma_1)^2 + (V_2\sigma_2)^2 + 2\rho V_1\sigma_1 V_2\sigma_2}$

**多日VaR（独立假设）**：
$\text{VaR}_{\alpha,T} = z_{\alpha} \times \sigma_p \times \sqrt{T}$

## 常见问题

1. **低估尾部风险**：正态分布假设会低估极端事件，实际收益常呈厚尾分布。
2. **非线性误差**：含期权的组合使用简单参数法误差较大。
3. **相关性变化**：危机时相关性可能剧变，历史相关性无法准确反映。
4. **均值忽略**：长期VaR计算忽略期望收益可能产生偏差。

## 相关概念
[[VaR]]
[[Historical Simulation Method|历史模拟法]]
[[Monte Carlo Simulation Method|蒙特卡罗模拟法]]
[[duration|久期]]
