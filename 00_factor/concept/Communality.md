---
aliases:
- Communality
- Common Variance
- 公共度
- 共同度
tags:
- concept
- multivariate statistics
---
# Communality

>[!note] 一句话记忆
> 公共度是某个变量的方差中，被公共因子共同解释的那一部分。

## 它是什么

在因子分析中，
$$
\Sigma=LL'+\Psi.
$$

第 $i$ 个变量的公共度为
$$
h_i^2=\sum_{j=1}^m l_{ij}^2.
$$

它与特殊方差满足
$$
\sigma_{ii}=h_i^2+\psi_i.
$$

## 解决什么判断

- 某个变量是否适合被当前公共因子解释。
- 因子模型是否遗漏了重要因子。
- 变量的独特部分是否过大。

## 最小例子

若第 1 个变量在两个因子上的载荷为 $0.8$ 和 $0.2$，则
$$
h_1^2=0.8^2+0.2^2=0.68.
$$

## 易混点

- 公共度不是相关系数，而是方差分解中的被解释部分。
- 标准化变量时，$0\leq h_i^2\leq1$；非标准化变量时上界是 $\sigma_{ii}$。
- 公共度高说明因子模型能解释该变量，但不等于因子具有因果意义。

## 来自课程位置

- [[09_因子分析Factor Analysis and Inference for Structured#1.3. 公共度与特殊方差|第9章 1.3 公共度与特殊方差]]

## 关联卡片

- [[Factor Analysis]]
- [[Factor Loadings]]
- [[Specific Variance]]
- [[Factor Analysis PC Method]]

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
