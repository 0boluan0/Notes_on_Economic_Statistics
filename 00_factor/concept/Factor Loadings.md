---
aliases:
- Factor Loadings
- Loading Matrix
- 因子载荷
- 因子载荷矩阵
tags:
- concept
- multivariate statistics
---

# Factor Loadings

>[!note] 一句话记忆
> 因子载荷是可观测变量对公共因子的敏感度，也是解释变量分组的主要依据。

## 它是什么

在因子模型
$$
X-\mu=LF+\epsilon
$$
中，$L=(l_{ij})$ 是因子载荷矩阵。

元素 $l_{ij}$ 表示第 $i$ 个变量与第 $j$ 个公共因子的联系强度。

## 解决什么判断

- 哪些变量主要由同一个因子解释。
- 某个因子应如何命名和解释。
- 某个变量是否在多个因子上都有较高载荷，导致解释不清。

## 最小例子

若变量 $X_1,X_2,X_3$ 在第一个因子上的载荷分别为 $0.85,0.78,0.10$，则 $X_1,X_2$ 更可能属于同一潜在维度。

## 易混点

- 载荷大不等于因果影响大，它首先是协方差结构中的解释系数。
- 载荷矩阵不唯一；旋转后更容易解释，但并不改变模型拟合的共同协方差。
- 金融中的因子暴露也是载荷思想，但课程中的重点是多元统计因子模型。

## 来自课程位置

- [[09_因子分析Factor Analysis and Inference for Structured#1.5.1. 主成分法（Principal Component Method）|第9章 3.1 主成分法]]

## 关联卡片

- [[Factor Analysis]]
- [[Communality]]
- [[Specific Variance]]
- [[Factor Analysis PC Method]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[09_因子分析Factor Analysis and Inference for Structured]]、[[Factor Analysis]]、[[Communality]]、[[Specific Variance]]、[[Factor Analysis PC Method]]。

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
