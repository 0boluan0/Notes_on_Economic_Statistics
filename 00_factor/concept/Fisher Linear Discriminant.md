---
aliases:
- Fisher Linear Discriminant
- Fisher Discriminant
- Linear Discriminant Analysis
- LDA
- Fisher 判别
- 线性判别分析
tags:
- concept
- multivariate statistics
---
# Fisher Linear Discriminant

>[!note] 一句话记忆
> Fisher 判别把多维观测投影到一条直线上，使两类投影均值尽量分开、类内变异尽量小。

## 它是什么

两总体情形中，常用判别方向为
$$
\hat a=S_{\text{pooled}}^{-1}(\bar x_1-\bar x_2).
$$

新观测 $x_0$ 投影为 $\hat a'x_0$，再与两类均值投影的中点比较。

## 解决什么判断

- 新观测应分到哪个已知类别。
- 哪个线性组合最能区分两组。
- 两类协方差近似相等时如何构造线性判别规则。

## 最小例子

用资产、保费、债券成本等变量把保险公司分为偿付能力正常与财务困境两类。

## 易混点

- Fisher 判别是投影思想；贝叶斯分类规则是密度比和成本思想。
- 线性判别通常隐含两组协方差相同或可 pooled 的条件。
- 判别方向不是回归系数，目标不是预测连续 $Y$。

## 来自课程位置

- [[11_分类与判别Discrimination and Classifications#1.4. Fisher 判别方法|第11章 4 Fisher 判别方法]]

## 关联卡片

- [[Fisher Discriminant Procedure]]
- [[Classification Rule Selection]]
- [[Expected Cost of Misclassification]]
