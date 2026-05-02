---
aliases:
- Factor Analysis
- FA
- 因子分析
tags:
- concept
- multivariate statistics
---
# Factor Analysis

>[!note] 一句话记忆
> 因子分析用少数不可观测的公共因子解释多个可观测变量之间的协方差结构。

## 它是什么

正交因子模型写作
$$
X-\mu=LF+\epsilon.
$$

其中：

- $L$ 是因子载荷矩阵；
- $F$ 是公共因子，满足 $E(F)=0,\ \operatorname{Cov}(F)=I_m$；
- $\epsilon$ 是特殊因子，满足 $E(\epsilon)=0,\ \operatorname{Cov}(\epsilon)=\Psi$，且 $\Psi$ 为对角矩阵；
- $\operatorname{Cov}(F,\epsilon)=0$。

因此协方差矩阵被拆成
$$
\Sigma=LL'+\Psi.
$$

## 解决什么判断

- 多个变量是否能由少数潜在维度解释。
- 每个变量有多少方差来自公共因子。
- 因子载荷矩阵是否给出了可解释的变量分组。

## 最小例子

如果问卷中“焦虑、紧张、担忧”高度相关，“愉悦、满意”也高度相关，因子分析会尝试用两个潜在因子分别解释这些变量组。

## 易混点

- 因子分析不是简单压缩变量；它关心共同协方差和特殊方差。
- [[PCA]] 的主成分由数据方差最大方向定义；因子分析有显式误差项 $\epsilon$。
- 因子载荷不唯一，正交旋转后 $L^*=LT$ 仍可给出相同的 $LL'$。

## 来自课程位置

- [[09_因子分析Factor Analysis and Inference for Structured#1. 第9章：因子分析（Factor Analysis）|第9章 因子分析]]

## 关联卡片

- [[Factor Loadings]]
- [[Communality]]
- [[Specific Variance]]
- [[Factor Analysis PC Method]]
- [[PCA vs Factor Analysis]]

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
