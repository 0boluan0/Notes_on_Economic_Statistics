---
aliases:
- HC标准误
- White稳健标准误
- White
- 稳健标准误
- White Robust Standard Errors
- White Robust SE
tags:
- system
- 计量经济学
---
# White 稳健标准误

## 诊断目的

在异方差存在的情况下，提供一致的协方差矩阵估计，确保即使同方差假设不成立，t检验和F检验仍然渐近有效。

## 计算方法

### White 异方差一致协方差矩阵（HC0）

$Var(\hat{\beta})_{White} = (X'X)^{-1} \left[ \sum_{i=1}^n x_i x_i' e_i^2 \right] (X'X)^{-1}$

### 改进的White估计量

| 类型 | 公式 | 特点 |
|------|------|------|
| HC0 | $(X'X)^{-1} \left(\sum x_i x_i' e_i^2\right) (X'X)^{-1}$ | 基本White估计 |
| HC1 | $\frac{n}{n-k}(X'X)^{-1} \left(\sum x_i x_i' e_i^2\right) (X'X)^{-1}$ | 小样本调整 |
| HC2 | $(X'X)^{-1} \left(\sum \frac{n}{n-k}(1-h_{ii})^{-1} x_i x_i' e_i^2\right) (X'X)^{-1}$ | 杠杆值调整 |
| HC3 | $(X'X)^{-1} \left(\sum (1-h_{ii})^{-2} x_i x_i' e_i^2\right) (X'X)^{-1}$ | 更激进调整 |

其中 $h_{ii}$ 是帽子矩阵对角元素。

## 适用场景

| 场景 | White标准误适用性 |
|------|------------------|
| 已知异方差模式 | 更好的方法是WLS |
| 未知异方差模式 | White标准误是最佳选择 |
| 样本量小 | HC2或HC3更稳健 |
| 样本量大 | HC1即可 |

## 判断标准

比较OLS标准误与White标准误：

| 情况 | White标准误 vs OLS标准误 | 含义 |
|------|------------------------|------|
| 显著较大 | White > OLS | OLS低估标准误 |
| 接近 | White ≈ OLS | 异方差不严重 |
| 更小 | White < OLS | 异方差使OLS标准误过大 |

## 常见问题与对策

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| White标准误与OLS标准误差异大 | 严重异方差 | 报告两种结果、考虑模型变换 |
| 仍然发现问题 | 异方差伴随自相关 | 使用Newey-West HAC标准误 |
| 小样本下结果不稳定 | 样本量小 | 使用HC2或HC3、或自助法（Bootstrap） |

## 相关概念
[[Heteroscedasticity Diagnosis|异方差诊断]]
[[Newey-West]]
[[FGLS]]
