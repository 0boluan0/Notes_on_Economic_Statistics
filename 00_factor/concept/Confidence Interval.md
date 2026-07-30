---
aliases:
- Confidence Interval
- CI
- 置信区间
tags:
- concept
- statistics
- econometrics
- probability
---

# Confidence Interval

## 先记一句话

置信区间用“估计值 ± 临界值 × 标准误”表达参数估计的不确定性。

## 它是什么

对由样本产生的估计量 $\hat\theta$，常见近似置信区间为：

$$
\hat\theta\pm c\cdot se(\hat\theta)
$$

其中 $c$ 是对应置信水平下的临界值。

## 解决什么判断

它是一条随机区间生成规则。若重复抽样并每次按同一规则构造区间，长期覆盖真实参数的比例等于所声明的 [[Confidence Level|置信水平]]。

## 最小例子

若 $\hat\beta_1=0.8$，$se=0.1$，95% 近似置信区间为：

$$
0.8\pm1.96\times0.1=[0.604,0.996]
$$

## 易混点

- 置信区间不是“真值有 95% 概率落在这个已算出的区间里”的贝叶斯说法。
- 区间是否可靠依赖标准误是否可靠。
- 若区间不含 0，通常对应双侧显著性检验拒绝 $H_0:\beta_j=0$。
- 6.042J 的抽样区间来自随机样本均值与偏差界；回归区间只是这一概念的一个应用。

## 来自课程位置

- [[02_一元线性回归]]

## 关联卡片

- [[P-value]]
- [[t Test]]
- [[White Robust Standard Errors]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[02_一元线性回归]]、[[P-value]]、[[t Test]]、[[White Robust Standard Errors]]。

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
