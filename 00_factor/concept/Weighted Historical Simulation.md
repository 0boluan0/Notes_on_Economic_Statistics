---
aliases:
- Weighted Historical Simulation
- Weighted Historical Simulation VaR
- Exponentially Weighted Historical Simulation
- 加权历史模拟法
- 加权历史模拟
tags:
- concept
- risk-management
- VaR
---
# Weighted Historical Simulation

## 一句话记忆

加权历史模拟给较新的历史收益更高权重，以使 VaR 对近期市场状态更敏感。

## 它是什么

对历史损失 $L_1,\ldots,L_n$ 赋予权重 $w_t$，再按加权经验分布的分位数估计 VaR。若 $t=1$ 表示最近期观测，常见指数权重为

$$
w_t=\frac{(1-\lambda)\lambda^{t-1}}{1-\lambda^n},\qquad 0<\lambda<1,
$$

归一化后满足 $\sum_t w_t=1$；若按从旧到新排序，指数次序需相应调整。

## 最小例子

若最近损失的累计权重已达到 99% VaR 的分位位置，就以该损失附近的加权经验分位数作为 VaR，而不是把每个历史观测等权处理。

## 易混点

- 它仍是经验分布方法，不等同于 [[EWMA]] 的条件波动率模型。
- $\lambda$ 越小，越强调近期，但有效样本量也可能下降。
- 权重不会自动修复异常值、结构突变或相关性建模问题。
>
## 关联卡片
- [[Historical Simulation Method]]
- [[VaR]]
- [[Observation Window]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
