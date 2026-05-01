---
aliases:
- Stock Valuation Model
- Equity Valuation Model
- 股票估价模型
- 股票定价模型
tags:
- concept
- finance
---
# Stock Valuation Model

## 一句话记忆

股票价值来自未来能分给股东的现金流折现。

## 它是什么

Stock Valuation Model 是用未来股利、自由现金流或估值倍数判断股票内在价值的一组模型。最基础的股利贴现模型为：

$$
P_0=\sum_{t=1}^{\infty}\frac{D_t}{(1+r)^t}
$$

若股利从下一期开始以稳定增长率 $g$ 增长：

$$
P_0=\frac{D_1}{r-g}
$$

## 解决什么判断

- 当前股价是否高于或低于内在价值。
- 股利、增长率和必要报酬率哪个变量在驱动估值。
- 稳定增长模型、两阶段模型或估值倍数是否适用。

## 最小例子

下一期股利 $D_1=3$，必要报酬率 $r=10\%$，长期增长率 $g=4\%$：

$$
P_0=\frac{3}{0.10-0.04}=50
$$

## 易混点

- 稳定增长模型必须满足 $r>g$。
- 股价高低不能只看增长率，还要看资本成本。
- DDM 适合股利可预测的公司；不分红或高成长公司常要换成自由现金流或倍数法。
- 估值模型输出依赖假设，不是市场价格的机械替代品。

## 来自课程位置

- [[06_债券和股票估价|03_债券与股票估值]]

## 关联卡片

- [[Bond and Stock Valuation]]
- [[CAPM]]
- [[CAPM Estimation]]
- [[Present Value]]
- [[Price-to-Earnings Ratio]]

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
