---
aliases:
- 自回归积分移动平均模型
- ARIMA模型
tags:
- 时间序列
- concept
---
ARIMA 模型（Autoregressive Integrated Moving Average）在 ARMA 基础上引入差分阶数 $d$，用于描述经差分后平稳的非平稳序列。

## 形式

对原序列 $y_t$ 做 $d$ 阶差分 $\Delta^d y_t$，若 $\Delta^d y_t$ 可由 ARMA($p$,$q$) 刻画，则称 $y_t \sim \text{ARIMA}(p,d,q)$。

## 何时使用

- 特征根含 1（单位根）导致非平稳，需要差分以达成平稳再建模。

## Connections

- 相关：[[ARMA]]、[[00_factor/concept/Unit Root Test|单位根检验]]、[[00_factor/concept/Stationarity|平稳性]]

## source_notes

- [[03_平稳时间序列模型#1.1.3 ARIMA过程]]
