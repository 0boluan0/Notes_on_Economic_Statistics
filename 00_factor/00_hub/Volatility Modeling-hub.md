---
aliases:
- 波动建模
- 波动率建模
- 波动率建模-hub
- Volatility Modeling
- Volatility Modeling-hub
tags:
- hub
- 时间序列
- 金融风险
---
# 波动率建模 Hub

## 先从哪条主线进

### 主线 1：先判断是不是方差问题

- 课程入口：[[04_波动建模 Modeling Volatility#1.1 为什么要进行波动建模|波动建模动机]]
- 先读这些卡：
  - [[Conditional Heteroskedasticity]]
  - [[Volatility Clustering]]
  - [[White Noise]]
  - [[Martingale Difference Sequence]]

### 主线 2：ARCH/GARCH 家族

- 课程入口：[[04_波动建模 Modeling Volatility#2. ARCH,GARCH|ARCH/GARCH]]
- 先读这些卡：
  - [[ARCH]]
  - [[GARCH]]
  - [[GARCH Model Estimation Steps]]
  - [[ARCH LM Test]]
  - [[McLeod-Li Test]]

### 主线 3：持久性与非对称波动

- 课程入口：[[04_波动建模 Modeling Volatility#3.4 非对称模型:TARCH,EGARCH|非对称模型]]
- 先读这些卡：
  - [[IGARCH]]
  - [[EWMA]]
  - [[TARCH]]
  - [[EGARCH]]
  - [[ARCH-M]]

### 主线 4：波动率度量

- 课程入口：[[10_波动率]]
- 先读这些卡：
  - [[Historical Volatility]]
  - [[Implied Volatility]]
  - [[Realized Volatility]]
  - [[EWMA Volatility Estimation]]

## 按题型找入口

### 题型：残差是否需要 GARCH

- [[ARCH LM Test]]
- [[McLeod-Li Test]]
- [[ARCH LM Test Diagnostics]]
- [[GARCH Model Estimation Steps]]

### 题型：标准 GARCH 不够用

- 冲击极持久：[[IGARCH]] / [[EWMA]]
- 负冲击影响更强：[[TARCH]] / [[EGARCH]]
- 风险进入期望收益：[[ARCH-M]]

### 题型：风险管理中给 VaR 准备波动率

- [[Historical Volatility]]
- [[EWMA Volatility Estimation]]
- [[GARCH]]
- [[Implied Volatility]]

## 课程导航

- [[04_波动建模 Modeling Volatility]]
- [[10_波动率]]
- [[12_VAR风险]]
- [[14_VaR参数法和模拟法]]

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
