---
aliases:
- Capital Holding Ratio
- 资本金持有率
- capital-to-asset ratio
tags:
- system
- banking
- risk-management
---
# Capital Holding Ratio

## 诊断目标

资本金持有率用资产分母衡量银行要持有多少资本，常用于简化题中判断“年末股权资本为正”的安全垫。

## 公式

$$
r=\frac{K}{A}
$$

其中 $K$ 是资本金，$A$ 是资产规模。

## 典型题型

若银行下一年度资产收益率服从正态分布，题目要求在 99% 或 99.9% 置信水平下年末资本为正，就用收益率分位数推最低资本率。

## 判断步骤

1. 写出资产收益率分布。
2. 找到目标置信水平下的左尾损失分位数。
3. 资本率至少覆盖该损失率。

## 易混点

- 资本金持有率使用总资产作分母；[[Basel Capital Adequacy Ratio]] 使用 [[Risk-Weighted Assets]] 作分母。
- 它是简化风险缓冲题，不等同于完整监管资本充足率。

## 来自课程位置

- [[01_引言]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[VaR]]
- [[Basel Capital Adequacy Ratio]]
- [[Leverage Ratio]]
