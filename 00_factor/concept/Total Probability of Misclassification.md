---
aliases:
- Total Probability of Misclassification
- TPM
- 误分类总概率
tags:
- concept
- multivariate statistics
---
# Total Probability of Misclassification

>[!note] 一句话记忆
> 误分类总概率是在不考虑错误成本差异时，对两类误判概率按先验概率加权的总错误概率。

## 它是什么

两类总体下，
$$
TPM=P_1P(2|1)+P_2P(1|2).
$$

如果分类区域为 $R_1,R_2$，也可写成
$$
TPM=P_1\int_{R_2}f_1(x)\,dx+P_2\int_{R_1}f_2(x)\,dx.
$$

## 解决什么判断

- 分类规则整体错误率有多大。
- 成本相等时哪个规则更优。
- 理论误分类概率和样本估计误差率如何区分。

## 最小例子

若 $P_1=P_2=0.5$，且 $P(2|1)=0.1,\ P(1|2)=0.2$，则 $TPM=0.15$。

## 易混点

- TPM 不考虑不同错误的成本；成本不等时看 [[Expected Cost of Misclassification]]。
- TPM 是理论概率，实际样本上估计的是 [[Actual Error Rate]]。

## 来自课程位置

- [[11_分类与判别Discrimination and Classifications#1.5. 分类性能评估|第11章 5 分类性能评估]]

## 关联卡片

- [[Expected Cost of Misclassification]]
- [[Actual Error Rate]]
