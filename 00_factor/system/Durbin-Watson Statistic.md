---
aliases:
- Durbin-Watson Statistic
- Durbin-Watson统计量
- DW test
- DW检验
tags:
- system
- econometrics
---
# Durbin-Watson Statistic

## 诊断目标

Durbin-Watson 统计量用于检验线性回归残差的一阶自相关。

## 公式

$$
DW=\frac{\sum_{t=2}^n(\hat u_t-\hat u_{t-1})^2}{\sum_{t=1}^n\hat u_t^2}
$$

近似关系：

$$
DW\approx 2(1-\hat\rho)
$$

## 快速解释

- $DW\approx 2$：无明显一阶自相关。
- $DW<2$：可能正自相关。
- $DW>2$：可能负自相关。

正式判断应查 $d_L,d_U$ 临界值表。

## 适用边界

- 模型应含截距。
- 不适合含滞后因变量的模型。
- 主要检验一阶自相关，不适合复杂高阶结构。

## 失败模式

- 只根据 $DW<2$ 就下结论，忽略不确定区间。
- 动态模型仍用 DW，导致检验无效。
- 高阶自相关漏检，应改用 [[Breusch-Godfrey Test]]。

## 来自课程位置

- [[08_自相关]]

## 关联卡片

- [[Autocorrelation]]
- [[Autocorrelation Diagnosis]]
- [[Breusch-Godfrey Test]]
