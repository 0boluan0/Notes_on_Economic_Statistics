---
aliases:
- Expected Cost of Misclassification
- ECM
- 期望误分类成本
tags:
- concept
- multivariate statistics
---
# Expected Cost of Misclassification

>[!note] 一句话记忆
> 期望误分类成本把先验概率、误分类概率和误分类损失合在一起，作为分类规则的优化目标。

## 它是什么

两类总体 $\pi_1,\pi_2$ 下，
$$
ECM=P_1P(2|1)c(2|1)+P_2P(1|2)c(1|2).
$$

其中 $P_i$ 是先验概率，$P(k|i)$ 是真实属于 $\pi_i$ 却被分到 $\pi_k$ 的概率，$c(k|i)$ 是相应成本。

## 解决什么判断

- 两种误分类错误成本不同时如何设定分类规则。
- 先验概率不等时如何调整阈值。
- 为什么不是简单选择密度更大的类别。

## 最小例子

把坏信用客户误判为好客户的成本，通常高于把好客户误判为坏客户。

## 易混点

- ECM 不是计量经济学里的 Error Correction Model。
- 成本相等且先验相等时，分类规则才简化为比较 $f_1(x)$ 和 $f_2(x)$。

## 来自课程位置

- [[11_分类与判别Discrimination and Classifications#1.3. 判别规则设定|第11章 3.2 判别规则与误分类成本]]

## 关联卡片

- [[Classification Rule Selection]]
- [[Total Probability of Misclassification]]
- [[Actual Error Rate]]
