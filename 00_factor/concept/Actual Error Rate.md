---
aliases:
- Actual Error Rate
- AER
- 实际误差率
- 实际错误率
tags:
- concept
- multivariate statistics
---
# Actual Error Rate

>[!note] 一句话记忆
> 实际误差率是用训练样本或验证样本估计分类规则真实错误概率的指标。

## 它是什么

在分类问题中，AER 关注分类器在实际样本上的错分比例或其估计。

课程中常把它和理论误分类概率、Jackknife 估计联系起来。

## 解决什么判断

- 分类器在样本上的表现如何。
- 需要不需要交叉验证校正乐观偏差。
- 不同分类规则哪个更稳。

## 最小例子

100 个验证样本中错分 12 个，则验证集错误率为 12%。

## 易混点

- 训练集 AER 往往偏乐观。
- Jackknife 或交叉验证估计更接近新样本上的表现。

## 来自课程位置

- [[11_分类与判别Discrimination and Classifications#1.5. 分类性能评估|第11章 5 分类性能评估]]

## 关联卡片

- [[Jackknife Classification]]
- [[Total Probability of Misclassification]]
- [[Classification Rule Selection]]
