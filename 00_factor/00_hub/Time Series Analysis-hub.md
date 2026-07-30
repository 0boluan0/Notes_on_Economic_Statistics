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

## 学习路线

### 1. 先补概率和统计入口

先看课程入口：[[00_导论与概率论复习]]、[[01_介绍时间序列]]。

这一段只负责把后面要用的语言准备好：随机变量、极限定理、i.i.d.、MDS、白噪声和 OLS。

核心卡片：

- [[IID]]
- [[Martingale Difference Sequence]]
- [[White Noise]]
- [[Law of Large Numbers]]
- [[Central Limit Theorem]]

### 2. 用差分方程理解动态系统

课程入口：[[02_差分方程Difference Equation]]。

先学会看一个递推系统会不会收敛，再进入 ARMA 的平稳性判断。

核心卡片：

- [[Difference Equation]]
- [[Difference Operator]]
- [[Lag Operator]]
- [[First Difference]]
- [[AR(1) stationarity]]
- [[Random Walk]]

### 3. 单变量平稳模型：ARMA 和预测

课程入口：[[03_平稳时间序列模型]]。

这条线回答：序列平稳以后，怎样识别、估计、检验和预测。

核心卡片：

- [[Stationarity]]
- [[Strict-Sense Stationary Process]]
- [[Wide-Sense Stationary Process]]
- [[ARMA]]
- [[Autoregressive Model]]
- [[Moving Average Model]]
- [[ARIMA]]
- [[Autocorrelation Function]]
- [[Partial Autocorrelation Function]]
- [[Yule-Walker equations]]
- [[ARMA Model Identification Steps]]
- [[Box-Jenkins Method]]
- [[Forecast Evaluation]]

### 4. 残差诊断与波动建模

课程入口：[[04_波动建模 Modeling Volatility]]。更细的路线见 [[Volatility Modeling-hub]]。

核心卡片：

- [[Conditional Heteroskedasticity]]
- [[Volatility Clustering]]
- [[ARCH]]
- [[GARCH]]
- [[ARCH LM Test]]
- [[McLeod-Li Test]]
- [[GARCH Model Estimation Steps]]
- [[IGARCH]]
- [[TARCH]]
- [[EGARCH]]

### 5. 多方程模型：ADL、传递函数和 VAR

课程入口：[[05_多方程模型Multi-equation Time Series Models]]。

这条线回答：多个变量一起动时，怎样描述滞后传导、冲击响应和预测因果。

核心卡片：

- [[Intervention Analysis]]
- [[ADL]]
- [[Distributed Lag Model]]
- [[Transfer Function Model]]
- [[Cross-Correlation Function]]
- [[Leading Indicator]]
- [[VAR Model]]
- [[Reduced Form VAR]]
- [[Structural VAR]]
- [[VMA]]
- [[Impulse Response Function]]
- [[Variance Decomposition]]
- [[Granger Causality Test]]

### 6. 非平稳、单位根和协整

课程入口：[[06_含趋势的模型]]、[[07_协整和误差修正模型]]。

这条线回答：序列不平稳时，什么时候差分，什么时候保留长期关系。

核心卡片：

- [[Unit Root Test]]
- [[Augmented Dickey-Fuller Test]]
- [[ADF Test Steps]]
- [[Phillips-Perron Test]]
- [[KPSS Test]]
- [[Stationarity Tests Comparison]]
- [[Spurious Regression]]
- [[Cointegration]]
- [[Common Stochastic Trend]]
- [[Engle-Granger Two-Step Test]]
- [[Johansen Cointegration Test]]
- [[Johansen Cointegration Test Steps]]
- [[Error Correction Model]]
- [[Cointegration theorem]]

## 按题型找入口

## 诊断决策顺序

先确认数据频率、缺失值和结构突变，再判断平稳性；平稳后识别均值动态，残差干净后才诊断 ARCH/GARCH。非平稳变量若存在长期关系，优先走协整/VECM 路线，不要把差分当作默认答案。

### 判断能不能直接建 ARMA

- [[Stationarity]]
- [[Unit Root Test]]
- [[ADF Test Steps]]
- [[First Difference]]
- [[ARIMA]]

### 根据 ACF/PACF 定阶

- [[Autocorrelation Function]]
- [[Partial Autocorrelation Function]]
- [[ARMA Model Identification Steps]]
- [[Box-Jenkins Method]]

### 残差本身干净，但残差平方有结构

- [[ARCH LM Test]]
- [[McLeod-Li Test]]
- [[ARCH LM Test Diagnostics]]
- [[GARCH Model Estimation Steps]]

### 比较两个预测模型谁更好

- [[Forecast Evaluation]]
- [[Granger-Newbold Test]]
- [[Diebold-Mariano Test]]

### 分析多变量动态传导

- [[Transfer Function Model]]
- [[Cross-Correlation Function]]
- [[VAR Model]]
- [[Impulse Response Function]]
- [[Variance Decomposition]]
- [[Granger Causality Test]]

### 非平稳变量之间是否有长期关系

- [[Spurious Regression]]
- [[Cointegration]]
- [[Engle-Granger Two-Step Test]]
- [[Johansen Cointegration Test]]
- [[Error Correction Model]]

## 课程导航

- [[00_导论与概率论复习]]
- [[01_介绍时间序列]]
- [[02_差分方程Difference Equation]]
- [[03_平稳时间序列模型]]
- [[04_波动建模 Modeling Volatility]]
- [[05_多方程模型Multi-equation Time Series Models]]
- [[06_含趋势的模型]]
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
