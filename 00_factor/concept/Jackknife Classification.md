---
aliases:
- Jackknife Classification
- Leave-one-out Classification
- Jackknife 分类
- 留一法分类
tags:
- concept
- multivariate statistics
---
# Jackknife Classification

>[!note] 一句话记忆
> Jackknife 分类每次留出一个观测，用其余样本训练分类器，再看被留出的观测是否被错分。

## 它是什么

流程是：

1. 去掉第 $j$ 个样本。
2. 用剩余样本估计分类规则。
3. 分类第 $j$ 个样本。
4. 对所有样本重复并汇总错分次数。

## 解决什么判断

- 训练集错误率是否过于乐观。
- 分类规则在新观测上的预计表现。
- 样本量较小时如何尽量利用数据做验证。

## 最小例子

50 个样本做 50 次留一分类，如果错分 6 次，则 Jackknife 错误率为 12%。

## 易混点

- Jackknife 不是重新抽样 bootstrap；它是系统性留一。
- 每次都要重新估计分类器，不能只用全样本分类结果替代。

## 来自课程位置

- [[11_分类与判别Discrimination and Classifications#1.6. 交叉验证方法|第11章 6 交叉验证方法]]

## 关联卡片

- [[Actual Error Rate]]
- [[Classification Rule Selection]]
