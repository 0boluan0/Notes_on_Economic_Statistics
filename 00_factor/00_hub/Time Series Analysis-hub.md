---
aliases:
- 时间序列分析
- 时间序列分析-hub
- Time Series Analysis
- Time Series Analysis-hub
tags:
- hub
- 时间序列
---
# 时间序列分析 Hub

## 先从哪条主线进

### 主线 1：平稳性、单位根与差分

- 课程入口：[[03_平稳时间序列模型]]
- 先读这些卡：
  - [[Stationarity]]
  - [[Strict-Sense Stationary Process]]
  - [[Wide-Sense Stationary Process]]
  - [[Random Walk]]
  - [[Unit Root Test]]
  - [[ADF Test Steps]]
  - [[First Difference]]
  - [[ARIMA]]

### 主线 2：ARMA 均值模型

- 课程入口：[[03_平稳时间序列模型#1.1 自回归移动平均模型ARMA(p,q) model|ARMA 定义]]、[[03_平稳时间序列模型#3. ACF|ACF/PACF]]
- 先读这些卡：
  - [[ARMA]]
  - [[Autoregressive Model]]
  - [[Moving Average Model]]
  - [[AR(1) stationarity]]
  - [[Autocorrelation Function]]
  - [[Partial Autocorrelation Function]]
  - [[Yule-Walker equations]]
  - [[ARMA Model Identification Steps]]
  - [[Box-Jenkins Method]]

### 主线 3：残差诊断与波动建模

- 课程入口：[[04_波动建模 Modeling Volatility]]
- 先读这些卡：
  - [[White Noise]]
  - [[IID]]
  - [[Martingale Difference Sequence]]
  - [[Conditional Heteroskedasticity]]
  - [[Volatility Clustering]]
  - [[ARCH]]
  - [[GARCH]]
  - [[GARCH Model Estimation Steps]]

### 主线 4：协整与误差修正

- 课程入口：[[07_协整和误差修正模型]]
- 先读这些卡：
  - [[Cointegration]]
  - [[Spurious Regression]]
  - [[Engle-Granger Two-Step Test]]
  - [[Johansen Cointegration Test]]
  - [[Johansen Cointegration Test Steps]]
  - [[Error Correction Model]]
  - [[Cointegration theorem]]

## 按题型找入口

### 题型：判断序列能不能直接建 ARMA

- [[Stationarity]]
- [[Unit Root Test]]
- [[ADF Test Steps]]
- [[First Difference]]

### 题型：根据 ACF/PACF 定阶

- [[Autocorrelation Function]]
- [[Partial Autocorrelation Function]]
- [[ARMA Model Identification Steps]]
- [[Box-Jenkins Method]]

### 题型：残差均值没问题，但波动有结构

- [[White Noise Test]]
- [[ARCH LM Test]]
- [[McLeod-Li Test]]
- [[GARCH Model Estimation Steps]]

### 题型：非平稳变量之间是否有长期关系

- [[Spurious Regression]]
- [[Cointegration]]
- [[Engle-Granger Two-Step Test]]
- [[Johansen Cointegration Test]]
- [[Error Correction Model]]

## 常用模型和检验

### 均值模型

- [[ARMA]]
- [[ARIMA]]
- [[VAR Model]]
- [[VMA]]
- [[ADL]]

### 单位根和平稳性检验

- [[Augmented Dickey-Fuller Test]]
- [[Phillips-Perron Test]]
- [[Stationarity Tests Comparison]]

### 白噪声与残差诊断

- [[White Noise]]
- [[White Noise Test]]
- [[Ljung-Box Test]]
- [[Box-Pierce Q statistic]]

### 协整检验

- [[Engle-Granger Two-Step Test]]
- [[Johansen Cointegration Test]]
- [[Johansen Cointegration Test Steps]]

## 课程导航

- [[01_介绍时间序列]]
- [[03_平稳时间序列模型]]
- [[04_波动建模 Modeling Volatility]]
- [[05_多方程模型Multi-equation Time Series Models]]
- [[07_协整和误差修正模型]]

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
