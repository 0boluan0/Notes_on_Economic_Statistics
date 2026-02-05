---
aliases:
- Hausman检验
- Hausman
- 豪斯曼检验
- Hausman Test
tags:
- 计量经济学
- 统计检验
- concept
---
Hausman检验用于检验变量是否内生，以及比较OLS和IV（或2SLS）估计量是否有显著差异。

## 基本原理

如果变量是外生的，则OLS和IV估计量都是一致的，且OLS更有效（方差更小）。如果变量是内生的，则OLS估计量有偏且不一致，而IV估计量是一致的。

通过比较OLS和IV估计量的差异来检验内生性。

## 检验统计量

$H = (\hat{\beta}_{IV} - \hat{\beta}_{OLS})' [\text{Var}(\hat{\beta}_{IV}) - \text{Var}(\hat{\beta}_{OLS})]^{-1} (\hat{\beta}_{IV} - \hat{\beta}_{OLS})$

在H₀（变量外生）下，H ~ χ²(k)

其中k是待检验的参数个数。

## 原假设和备择假设

- H₀：变量是外生的，OLS和IV估计量无显著差异
- H₁：变量是内生的，OLS和IV估计量有显著差异

## 检验步骤

1. 分别估计OLS和IV（或2SLS）模型
2. 计算估计量差异：$\hat{\beta}_{IV} - \hat{\beta}_{OLS}$
3. 计算方差差矩阵：$\text{Var}(\hat{\beta}_{IV}) - \text{Var}(\hat{\beta}_{OLS})$
4. 计算H统计量
5. 比较H统计量与χ²(k)分布的临界值

## 判断标准

- 若H统计量显著，拒绝H₀，认为变量是内生的，应使用IV估计
- 若H统计量不显著，无法拒绝H₀，认为变量是外生的，可以使用OLS估计

## 方案选择

根据检验结果：

1. **接受H₀**：使用OLS估计（更有效）
2. **拒绝H₀**：使用IV估计（一致）

## 扩展应用

Hausman检验也用于：

1. **固定效应vs随机效应**：在面板数据模型中检验
2. **模型设定检验**：比较不同估计方法
3. **过度识别检验**：当工具变量多于内生变量时

## 注意事项

1. 方差差矩阵$\text{Var}(\hat{\beta}_{IV}) - \text{Var}(\hat{\beta}_{OLS})$应为正定矩阵
2. 如果方差差矩阵非正定，检验可能无法进行
3. 检验结果对工具变量选择敏感

相关链接: [[2SLS]], [[Instrumental Variable|工具变量]], [[Endogeneity|内生性]], [[Panel Data Model|面板数据模型]]
