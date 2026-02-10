---
aliases:
- Heteroskedasticity and Autocorrelation
- 异方差与自相关
tags:
- system
---
# Heteroskedasticity and Autocorrelation

## 诊断目标

识别残差方差不恒定或序列相关，避免标准误偏误导致的推断失真。

## 诊断信号

- 残差散点图呈“喇叭形”或随时间聚集。
- 统计检验显著：White、Breusch-Pagan、Durbin-Watson、BG、Ljung-Box。

## 稳健与修正策略

- 使用稳健标准误：White、Newey-West。
- 重新设定模型：加入遗漏变量、趋势、季节项或滞后项。
- 变换变量：对数、差分或标准化以稳定方差。
- 结构性方法：FGLS/GLS、ARMA 误差结构建模。

## 风险点

- 系数估计可能仍无偏，但标准误与显著性判断失真。
- 共线性与设定偏误可能同时存在，导致误判来源。

## 报告与复现

- 报告检验统计量与稳健标准误的选择理由。
- 保留残差图与诊断代码，保证可复现。

## 相关链接

- [[Heteroscedasticity Diagnosis|异方差诊断]]
- [[Autocorrelation Diagnosis|自相关诊断]]
- [[White Robust Standard Errors|White稳健标准误]]
- [[Newey-West]]

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
