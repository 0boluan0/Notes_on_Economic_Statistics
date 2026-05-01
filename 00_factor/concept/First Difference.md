---
aliases:
- First Difference
- first differencing
- Δ
- 一阶差分
tags:
- concept
- 时间序列
---
# First Difference

## 先记一句话

一阶差分就是：**把水平值改成相邻两期的变化量**。

公式是
$$
\Delta y_t=y_t-y_{t-1}.
$$

## 它解决什么判断

一阶差分常用于把单位根非平稳序列变成平稳序列。

如果 $y_t$ 是 $I(1)$，那么
$$
\Delta y_t
$$
通常是 $I(0)$。

## 一个最小例子

随机游走：
$$
y_t=y_{t-1}+\varepsilon_t.
$$

差分后：
$$
\Delta y_t=\varepsilon_t.
$$

这说明差分消掉了随机游走中的单位根。

## 它在题里负责什么

- ARIMA 中的 $d$ 阶差分。
- 单位根检验后对非平稳序列做平稳化。
- 把水平关系转换为变化率或增量关系。

## 常见误区

- 差分能处理单位根，但会丢掉水平上的长期关系。
- 如果变量之间协整，不应只做差分回归，还要看 [[Error Correction Model]]。
- 过度差分会引入不必要的噪声和 MA 结构。

## 来自课程位置

- [[03_平稳时间序列模型#1.1.3 ARIMA过程|时间序列 03：ARIMA 与差分]]
- [[07_协整和误差修正模型#2.3 协整与误差修正模型|时间序列 07：差分项与 ECM]]

## 关联卡片

- [[Random Walk]]
- [[ARIMA]]
- [[Unit Root Test]]
- [[Cointegration]]
- [[Error Correction Model]]

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
