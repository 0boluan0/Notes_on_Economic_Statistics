---
aliases:
- Volatility Clustering
- volatility clustering
- 波动率聚集
- 波动聚集
tags:
- concept
- 时间序列
- 波动建模
---
# Volatility Clustering

## 先记一句话

波动聚集就是：**大波动后面常跟大波动，小波动后面常跟小波动**。

这不是说收益率方向可预测，而是说波动大小有持续性。

## 它是什么

金融收益率常常满足：

- 收益率本身自相关不明显；
- 收益率平方或绝对值有明显自相关。

这说明均值可能不可预测，但风险大小可预测。

## 一个最小例子

危机日出现巨大负收益后，后面几天市场通常仍然剧烈波动。

这时你不一定知道明天涨还是跌，但知道明天“更可能很动荡”。

## 它在题里负责什么

波动聚集是选择 [[ARCH]] / [[GARCH]] 的直觉理由。

诊断上：

- 残差 ACF：看均值模型是否漏结构；
- 残差平方 ACF：看波动是否聚集；
- [[ARCH LM Test]] / [[McLeod-Li Test]]：正式检验 ARCH 效应。

## 常见误区

- 波动聚集不是收益率可预测，而是方差可预测。
- 不要只看原序列 ACF；要看平方残差。
- 波动聚集强时，用固定历史标准差可能反应太慢。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#1.1 为什么要进行波动建模|时间序列 04：波动聚集]]

## 关联卡片

- [[Conditional Heteroskedasticity]]
- [[ARCH]]
- [[GARCH]]
- [[EWMA]]
- [[ARCH LM Test]]

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
