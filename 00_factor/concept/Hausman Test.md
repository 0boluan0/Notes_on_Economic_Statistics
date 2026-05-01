---
aliases:
- Hausman Test
- Hausman检验
- 豪斯曼检验
tags:
- concept
- econometrics
---
# Hausman Test

## 先记一句话

Hausman Test 比较两个估计量是否系统性不同，用来判断更有效但可能不一致的估计量是否可用。

## 它是什么

典型内生性场景中：

- 原假设：OLS 和 IV 都一致，OLS 更有效。
- 备择假设：OLS 不一致，IV 一致。

统计量基于估计量差异：

$$
H=(\hat\beta_A-\hat\beta_B)'[Var(\hat\beta_A)-Var(\hat\beta_B)]^{-1}(\hat\beta_A-\hat\beta_B)
$$

## 解决什么判断

它回答：“OLS 和 IV/FE/RE 的差异是否大到说明某个关键假设不成立？”

## 最小例子

比较 OLS 和 2SLS 的教育回报估计。如果差异显著，说明教育可能内生，OLS 不可信。

## 易混点

- Hausman Test 不是工具变量有效性的完整证明。
- 面板中也用它比较固定效应和随机效应。
- 方差差矩阵非正定时，软件可能给出替代实现或无法计算。

## 来自课程位置

- [[09_联立方程模型(内生性)]]
- [[13_面板数据模型]]

## 关联卡片

- [[Endogeneity Diagnosis]]
- [[2SLS]]
- [[Fixed Effects Model]]
- [[Random Effects Model]]
