---
aliases:
- 平稳性检验对比
- DF/ADF/KPSS 对比
- Stationarity Tests Comparison
tags:
- system
- 时间序列
---
# Stationarity Tests Comparison

DF/ADF 与 KPSS 检验相互补充：前者原假设为“存在单位根（非平稳）”，后者原假设为“平稳”。联合使用可提高判断稳健性。

## 速览

- DF/ADF：$H_0$ 非平稳；$H_1$ 平稳；ADF 通过加滞后差分消除自相关。
- KPSS：$H_0$ 平稳；$H_1$ 非平稳；按“水平/趋势”版本选择。

实践建议：
- 同时报告 DF/ADF 与 KPSS 结果；出现“拒绝/不拒绝”冲突时检查趋势设定、样本长度、结构突变。

## Connections

- 相关：[[Unit Root Test|单位根检验]]、[[Stationarity|平稳性]]、[[Random Walk|随机游走]]、[[ARIMA|ARIMA模型]]

## $source_notes$

- [[06_含趋势的模型#4.4 三检验比较]]
- [[12_非平稳时间序列#单位根检验和平稳性检验]]

## 课程笔记反链

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

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
