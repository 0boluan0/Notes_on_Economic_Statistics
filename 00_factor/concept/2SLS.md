---
aliases:
- 两阶段最小二乘法
- Two-Stage Least Squares
- SLS
- 2SLS
tags:
- 计量经济学
- 估计方法
- concept
---
2SLS（Two-Stage Least Squares，两阶段最小二乘法）是解决内生性问题的常用估计方法。

## 估计步骤

### 第一阶段

用工具变量Z对内生解释变量X回归：

$X = \gamma_0 + \gamma_1 Z + v$

得到预测值：$\hat{X} = \hat{\gamma}_0 + \hat{\gamma}_1 Z$

### 第二阶段

用预测值$\hat{X}$对被解释变量y回归：

$y = \beta_0 + \beta_1 \hat{X} + \varepsilon$

## 估计量公式

对于简单回归模型y = β_0 + β_1X + ε：

$\hat{\beta}_{2SLS} = \frac{\text{Cov}(\hat{X}, y)}{\text{Var}(\hat{X})}$

其中$\hat{X}$是第一阶段回归得到的预测值。

## 与IV估计的关系

2SLS是IV估计量的一种实现方式。

$\hat{\beta}_{IV} = \frac{\text{Cov}(Z, y)}{\text{Cov}(Z, X)}$

当工具变量和内生变量都是一维时，2SLS与IV等价。

## 矩阵形式

对于多变量模型：

$y = X\beta + \varepsilon$

第一阶段：
$X = Z\Pi + V$
$\hat{X} = P_Z X$

第二阶段：
$\hat{\beta}_{2SLS} = (X^T P_Z X)^{-1} X^T P_Z y$

$其中P_Z = Z(Z^T Z)^{-1} Z^T是投影矩阵。$

## 性质

1. **一致性**：在工具变量满足外生性和相关性条件下，2SLS估计量是一致的
2. **非有效性**：OLS估计量在无内生性时更有效（方差更小）
3. **渐近正态**：在大样本下，2SLS估计量服从正态分布

## 多工具变量的情况

当工具变量个数m > 内生变量个数k时：

第一阶段：每个内生变量用所有工具变量回归
第二阶段：使用所有内生变量的预测值

## 检验

1. **工具变量有效性检验**：过度识别检验（J检验）
2. **内生性检验**：Hausman检验
3. **弱工具变量检验**：第一阶段F检验

相关链接: [[Instrumental Variable|工具变量]], [[GMM]], [[Hausman Test|Hausman检验]], [[Endogeneity|内生性]]
