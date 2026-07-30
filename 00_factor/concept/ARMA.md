---
aliases:
- ARMA
- ARMA Model
- Autoregressive Moving Average Model
- ARMA模型
- 自回归移动平均模型
tags:
- concept
- 时间序列
---

# ARMA

## 先记一句话

ARMA 是：**用过去的序列值和过去的冲击共同解释当前值的平稳时间序列模型**。

它把两种记忆放在一起：

- AR：过去的 $y$ 影响现在的 $y$；
- MA：过去的 shock 影响现在的 $y$。

## 它是什么

ARMA$(p,q)$ 常写成
$$
y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}
+\varepsilon_t+\sum_{j=1}^{q}\theta_j\varepsilon_{t-j}.
$$

其中：

- $p$ 是 AR 阶数；
- $q$ 是 MA 阶数；
- $\varepsilon_t$ 通常设为 [[White Noise]]。

## 它解决什么判断

ARMA 用来刻画已经平稳的序列里的短期动态结构。

题目出现这些信号时，想到 ARMA：

- ACF/PACF；
- model identification；
- white-noise residual；
- AIC/BIC 定阶；
- forecast stationary series。

## 平稳性和可逆性

ARMA 的平稳性由 AR 部分决定：

> AR characteristic roots 要在单位圆外。

最简单的参照是 [[AR(1) stationarity]]：
$$
y_t=a_0+a_1y_{t-1}+\varepsilon_t
$$
平稳要求
$$
|a_1|<1.
$$

MA 部分不决定平稳性，但决定可逆性。

## ACF/PACF 识别

| 模型 | ACF | PACF |
| --- | --- | --- |
| AR(p) | 拖尾 | p 阶后截尾 |
| MA(q) | q 阶后截尾 | 拖尾 |
| ARMA(p,q) | 拖尾 | 拖尾 |

实际建模见 [[ARMA Model Identification Steps]] 和 [[Box-Jenkins Method]]。

## 常见误区

- ARMA 默认用于平稳序列；非平稳要先看 [[ARIMA]]、差分或协整。
- ACF/PACF 是识别线索，不是机械答案；最终还要看 AIC/BIC 和残差白噪声检验。
- ARMA 解释条件均值，不负责条件方差；若残差平方有自相关，转向 [[ARCH]] / [[GARCH]]。

## 来自课程位置

- [[03_平稳时间序列模型#1.1 自回归移动平均模型ARMA(p,q) model|时间序列 03：ARMA 定义]]
- [[03_平稳时间序列模型#2.2.1 AR(1)的平稳性判断|时间序列 03：AR 平稳性]]
- [[03_平稳时间序列模型#3. ACF|时间序列 03：ACF/PACF 识别]]

## 关联卡片

- [[Stationarity]]
- [[Autoregressive Model]]
- [[Moving Average Model]]
- [[Autocorrelation Function]]
- [[Partial Autocorrelation Function]]
- [[ARMA Model Identification Steps]]
- [[White Noise]]
- [[ARCH]]
- [[GARCH]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[White Noise]]、[[AR(1) stationarity]]、[[ARMA Model Identification Steps]]、[[Box-Jenkins Method]]、[[ARIMA]]、[[ARCH]]、[[GARCH]]、[[03_平稳时间序列模型]]。

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
