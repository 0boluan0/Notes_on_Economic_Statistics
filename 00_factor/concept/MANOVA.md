---
aliases:
- MANOVA
- Multivariate Analysis of Variance
- 多元方差分析
tags:
- concept
- multivariate statistics
---
# MANOVA

>[!note] 一句话记忆
> MANOVA 用一个联合检验比较多个组的均值向量是否相同。

## 它是什么

单因子 MANOVA 的常见原假设是
$$
H_0:\mu_1=\mu_2=\cdots=\mu_g.
$$

它把组间变异矩阵和组内误差变异矩阵进行比较，而不是逐个因变量做 ANOVA。

## 解决什么判断

- 多个总体均值向量是否整体相同。
- 组别差异是否体现在多个变量的联合模式上。
- 是否需要进一步做单变量检验或多重比较。

## 最小例子

比较三种教学方法对数学、阅读、写作三项成绩的均值向量是否不同。

## 易混点

- MANOVA 检验的是均值向量，不是协方差矩阵是否相等。
- 多个单变量 ANOVA 不能替代 MANOVA，因为它们忽略变量之间的协方差。
- 结果显著后，仍需解释哪些变量或线性组合贡献差异。

## 来自课程位置

- [[06_比较多个均值向量comparisons of multivariate mean vectors#1.5. 多个总体均值向量比较：单因子 MANOVA|第6章 4 单因子 MANOVA]]

## 关联卡片

- [[One-way MANOVA Procedure]]
- [[Wilks Lambda]]
- [[SSP Matrix]]
- [[Multivariate Mean Inference Map]]
