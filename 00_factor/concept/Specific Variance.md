---
aliases:
- Specific Variance
- Unique Variance
- 特殊方差
- 独特方差
tags:
- concept
- multivariate statistics
---
# Specific Variance

>[!note] 一句话记忆
> 特殊方差是某个变量中没有被公共因子解释的那一部分方差。

## 它是什么

在因子分析中，
$$
\Sigma=LL'+\Psi.
$$

第 $i$ 个变量的方差可写成
$$
\sigma_{ii}=h_i^2+\psi_i,
$$
其中：

- $h_i^2$ 是 [[Communality]]；
- $\psi_i$ 是特殊方差；
- $\Psi=\operatorname{diag}(\psi_1,\ldots,\psi_p)$。

## 解决什么判断

- 某个变量是否主要被公共因子解释。
- 当前因子数量是否不足。
- 主成分法估计因子模型时是否出现 Heywood case，即 $\hat\psi_i<0$。

## 最小例子

如果 $\sigma_{ii}=1$，公共度 $h_i^2=0.72$，则
$$
\psi_i=1-0.72=0.28.
$$

## 易混点

- 特殊方差不是“误差方差”的普通回归含义；它是因子模型中变量独有部分的方差。
- 如果估计出负的特殊方差，通常说明模型或因子数设定有问题。

## 来自课程位置

- [[09_因子分析Factor Analysis and Inference for Structured#1.2. 正交因子模型（Orthogonal Factor Model）|第9章 2.2 正交因子模型]]

## 关联卡片

- [[Factor Analysis]]
- [[Factor Loadings]]
- [[Communality]]
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
