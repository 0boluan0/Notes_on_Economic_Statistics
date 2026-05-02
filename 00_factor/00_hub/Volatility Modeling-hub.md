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

## 学习路线

### 1. 先判断问题是不是方差问题

课程入口：[[04_波动建模 Modeling Volatility#1.1. 为什么要进行波动建模|波动建模动机]]。

先区分两件事：均值模型是否遗漏动态，还是残差条件方差本身有结构。

核心卡片：

- [[Conditional Heteroskedasticity]]
- [[Volatility Clustering]]
- [[White Noise]]
- [[Martingale Difference Sequence]]
- [[ARCH LM Test Diagnostics]]

### 2. ARCH/GARCH 主线

课程入口：[[04_波动建模 Modeling Volatility#2. ARCH,GARCH|ARCH/GARCH]]。

核心卡片：

- [[ARCH]]
- [[GARCH]]
- [[ARCH LM Test]]
- [[McLeod-Li Test]]
- [[GARCH Model Estimation Steps]]

### 3. 持久性与非对称波动

课程入口：[[04_波动建模 Modeling Volatility#3. 扩展模型|扩展波动模型]]。

核心卡片：

- [[IGARCH]]
- [[EWMA]]
- [[ARCH-M]]
- [[TARCH]]
- [[EGARCH]]

### 4. 风险管理中的波动率输入

这一段连接风险管理课程中的 VaR 和波动率估计。

核心卡片：

- [[Historical Volatility]]
- [[Implied Volatility]]
- [[Realized Volatility]]
- [[EWMA Volatility Estimation]]
- [[GARCH]]

## 按题型找入口

### 残差是否需要 GARCH

- [[ARCH LM Test]]
- [[McLeod-Li Test]]
- [[ARCH LM Test Diagnostics]]
- [[GARCH Model Estimation Steps]]

### 标准 GARCH 不够用

- 冲击极持久：[[IGARCH]] / [[EWMA]]
- 负冲击影响更强：[[TARCH]] / [[EGARCH]]
- 风险进入期望收益：[[ARCH-M]]

### 给 VaR 准备动态波动率

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
