---
aliases:
- Dynamic Panel Data Model
- Dynamic Panel Model
- 动态面板数据模型
tags:
- concept
- econometrics
- panel-data
---
# Dynamic Panel Data Model

## 先记一句话

动态面板模型在面板数据中加入滞后被解释变量，因此会重新引入内生性问题。

## 它是什么

典型形式：

$$
y_{it}=\alpha+\rho y_{i,t-1}+x_{it}'\beta+\mu_i+u_{it}
$$

其中 $\mu_i$ 是个体效应。

## 解决什么判断

它回答：“当前结果是否受到自身过去值影响，并且这种动态结构能否在面板中被一致估计？”

## 最小例子

今年企业投资可能受去年投资影响，同时企业固定管理能力 $\mu_i$ 也影响投资。

## 易混点

- 静态固定效应不能直接解决动态面板的全部内生性。
- 差分会消除 $\mu_i$，但 $\Delta y_{i,t-1}$ 与 $\Delta u_{it}$ 仍相关。
- 常用 [[GMM]]，尤其差分 GMM 或系统 GMM。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[Panel Data Model]]
- [[Fixed Effects Model]]
- [[GMM]]
- [[Hansen J Test]]
