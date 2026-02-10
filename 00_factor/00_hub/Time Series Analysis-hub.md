---
aliases:
- 时间序列分析
- 时间序列分析-hub
- Time Series Analysis
- Time Series Analysis-hub
tags:
- hub
---
# 时间序列分析 Hub

## 核心概念

[[ARMA]]
[[Stationarity|平稳性]]
[[Random Walk|随机游走]]

## 基础模型

### 自回归移动平均模型
[[ARMA]]

### 向量自回归
[[VAR Model|VAR]]
[[VMA]]

## 平稳性与单位根

[[Stationarity|平稳性]]
[[Strict-Sense Stationary Process|严平稳过程]]
[[Wide-Sense Stationary Process|宽平稳过程]]
[[Random Walk|随机游走]]
[[Unit Root Test|单位根检验]]
[[Augmented Dickey-Fuller Test|ADF检验]]
[[Phillips-Perron Test|PP检验]]
[[Engle-Granger Two-Step Test|EG两步检验法]]
[[First Difference|一阶差分]]

## 相关函数分析

[[Autocorrelation Function|自相关函数]]
[[Partial Autocorrelation Function|偏自相关函数]]
[[Yule-Walker equations|Yule-Walker方程]]
[[White Noise|白噪声过程]]
[[Martingale Difference Sequence|鞅差序列]]
[[IID|独立同分布（i.i.d.）]]

## 波动率建模

### ARCH 类模型
[[ARCH]]
[[GARCH]]
[[TARCH]]
[[EGARCH]]

### 波动率估计
[[Historical Volatility|历史波动率]]
[[Implied Volatility|隐含波动率]]
[[Realized Volatility|已实现波动率]]
[[EWMA]]

## 检验方法

[[Box-Pierce Q statistic|Box-Pierce和Ljung-Box检验]]
[[Ljung-Box Test|Ljung-Box检验]]
[[White Noise Test|白噪声检验]]

## 模型识别与估计

[[Box-Jenkins Method|Box-Jenkins方法]]
[[ARMA Model Identification Steps|ARMA模型识别步骤]]
[[ARIMA|ARIMA模型]]

## 协整与长期关系

[[Cointegration|协整]]
[[Cointegration theorem|协整定理]]
[[Johansen Cointegration Test|Johansen检验]]
[[Error Correction Model|误差修正模型]]
[[Error Correction Model|误差纠正机制(ECM)]]
[[Spurious Regression|伪回归]]

## 课程笔记

- [[01_介绍时间序列]]
- [[03_平稳时间序列模型]]
- [[04_波动建模 Modeling Volatility]]

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
