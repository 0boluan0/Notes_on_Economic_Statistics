---
aliases:
- Marginal VaR
- marginal Value at Risk
- 边际VaR
tags:
- concept
- 金融风险
- VaR
---
# Marginal VaR

## 先记一句话

边际 VaR 就是：**某个头寸微小增加 1 单位时，组合 VaR 会增加多少**。

它是组合 VaR 对单个头寸的偏导数。

## 它是什么

若组合头寸向量为 $w$，则资产 $i$ 的 marginal VaR 是
$$
\frac{\partial VaR(w)}{\partial w_i}.
$$

它回答的是“边际风险贡献”，不是某个资产自己的 standalone VaR。

## 它解决什么判断

Marginal VaR 用来判断：

> 如果我再加一点这个资产，组合风险会怎么变？

如果某资产与组合负相关，边际 VaR 甚至可能为负，表示增加它可能降低组合 VaR。

## 和其他 VaR 分解的分工

- [[Marginal VaR]]：微小增加一单位的风险变化率。
- [[Incremental VaR]]：新增/删除一笔具体交易后的有限变化。
- [[Component VaR]]：把总 VaR 分摊到各资产。

## 常见误区

- 边际 VaR 不是资产自身 VaR。
- 边际 VaR 是局部概念，大幅调仓时要重新计算 incremental VaR。
- 在相关性强的组合里，单资产风险大小不等于边际贡献大小。

## 来自课程位置

- [[12_VAR风险|金融风险管理 12：VaR 分解]]

## 关联卡片

- [[VaR]]
- [[Incremental VaR]]
- [[Component VaR]]
- [[Variance-Covariance Method]]

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
