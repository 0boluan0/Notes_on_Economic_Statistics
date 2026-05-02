---
aliases:
- Difference Operator
- Delta Operator
- Difference operator
- 差分算子
- Δ算子
tags:
- concept
- 时间序列
---
# Difference Operator

## 先记一句话

差分算子就是：**把水平值变成相邻时期的变化量**。

## 它是什么

一阶差分写作：
$$
\Delta y_t=y_t-y_{t-1}.
$$

二阶差分是对一阶差分再差分：
$$
\Delta^2 y_t=\Delta(\Delta y_t)=y_t-2y_{t-1}+y_{t-2}.
$$

用 [[Lag Operator]] 可以写成：
$$
\Delta=(1-L).
$$

## 它解决什么判断

- 序列有单位根时，差分能否把它变成 [[Stationarity|平稳]]。
- [[ARIMA]] 中的 $d$ 阶差分具体是什么意思。
- 趋势模型里水平关系和变化量关系是否应区分。

## 最小例子

随机游走：
$$
y_t=y_{t-1}+\varepsilon_t.
$$

差分后：
$$
\Delta y_t=\varepsilon_t.
$$

这说明差分消掉了单位根带来的随机趋势。

## 易混点

- 差分不是去掉所有趋势。确定性趋势和随机趋势要分开判断。
- 差分会丢掉水平关系。如果变量之间有 [[Cointegration|协整]]，不能只做差分回归。
- $\Delta y_t$ 和增长率不同。增长率通常还要除以基期水平或取对数差分。

## 来自课程位置

- [[02_差分方程Difference Equation#1.2. 术语：差分算子（Difference operator）|时间序列 02：差分算子]]
- [[06_含趋势的模型#3.3. 差分|时间序列 06：单位根过程的差分]]

## 关联卡片

- [[Lag Operator]]
- [[First Difference]]
- [[Unit Root Test]]
- [[ARIMA]]
- [[Random Walk]]

