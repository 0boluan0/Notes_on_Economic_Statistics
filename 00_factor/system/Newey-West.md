---
aliases:
- HAC标准误
- Newey-West Standard Errors
- Newey
- Newey-West
tags:
- system
- 计量经济学
---
# Newey-West 标准误

## 诊断目的

在异方差和自相关同时存在的情况下，提供异方差-自相关一致（HAC）协方差矩阵估计，确保t检验和F检验的渐近有效性。

## 计算方法

### Newey-West 估计量

$$Var(\hat{\beta})_{NW} = (X'X)^{-1} S \ (X'X)^{-1}$$

其中S是长期方差协方差矩阵估计：

$$S = \hat{\Omega}_0 + \sum_{j=1}^L \omega_j (\hat{\Omega}_j + \hat{\Omega}_j')$$

### 长期协方差分量

$$\hat{\Omega}_j = \frac{1}{n} \sum_{t=j+1}^n x_t x_{t-j}' e_t e_{t-j}$$

### Bartlett权重

$$\omega_j = 1 - \frac{j}{L+1}$$

其中L是最大滞后阶数（带宽）。

### 带宽选择

| 规则 | 带宽L | 适用情况 |
|------|--------|----------|
| Newey-West自动 | $\lfloor 4(n/100)^{2/9} \rfloor$ | 大样本 |
| 固定带宽 | 经验值（如4-6） | 小样本 |
| Andrews | 自动最优选择 | 精确要求高 |

## 适用场景

| 场景 | Newey-West适用性 |
|------|-----------------|
| 纯异方差 | White标准误更有效 |
| 纯自相关 | Cochrane-Orcutt更有效 |
| 异方差+自相关 | Newey-West是最佳选择 |
| 纵截面数据 | 通常不需要 |

## 判断标准

| 情况 | Newey-West vs White vs OLS | 含义 |
|------|--------------------------|------|
| NW > White > OLS | 自相关和异方差都存在 | OLS严重低估标准误 |
| NW > White ≈ OLS | 主要自相关问题 | 需处理自相关 |
| NW ≈ White > OLS | 主要异方差问题 | White标准误即可 |

## 常见问题与对策

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 带宽选择困难 | 样本量小 | 尝试多个带宽，进行敏感性分析 |
| 结果不稳定 | 带宽过大或过小 | 使用自动选择规则或Andrews方法 |
| 计算缓慢 | 样本量很大、带宽大 | 限制最大滞后阶数 |

## 相关概念
[[00_factor/system/White Robust Standard Errors|White稳健标准误]]
[[00_factor/system/Autocorrelation Diagnosis|自相关诊断]]
[[00_factor/system/Heteroscedasticity Diagnosis|异方差诊断]]
