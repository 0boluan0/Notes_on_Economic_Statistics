---
aliases:
- ARCH LM Test diagnostics
- ARCH/GARCH effect diagnosis
- 侦测ARCH/GARCH效应
tags:
- system
- 时间序列
- 波动建模
---
# ARCH LM Test Diagnostics

## 这张卡回答什么问题

这张卡负责诊断流程：

> 均值模型已经拟合后，怎样判断是否还需要 ARCH/GARCH 方差模型？

概念定义见 [[ARCH LM Test]]。

## 诊断顺序

1. 先看残差 $\hat\varepsilon_t$ 的 ACF。
2. 如果残差有自相关，先修均值模型。
3. 再看残差平方 $\hat\varepsilon_t^2$ 的 ACF。
4. 对平方残差做 [[ARCH LM Test]] 或 [[McLeod-Li Test]]。
5. 若显著，进入 [[GARCH Model Estimation Steps]]。

## 结果解释

| 现象 | 解释 | 下一步 |
| --- | --- | --- |
| 残差有自相关 | 均值模型漏结构 | 重做 ARMA/均值模型 |
| 残差无自相关，平方残差有自相关 | 方差有结构 | 估计 ARCH/GARCH |
| 两者都不显著 | 均值和方差诊断暂时通过 | 做常规预测/风险计算 |

## 稳健性提醒

- 金融收益常有厚尾，正态 GARCH 可能低估尾部风险。
- 结构突变会让 ARCH 检验显著，但原因未必是稳定 GARCH 动态。
- 样本太短时，ARCH 效应诊断不稳。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#2.4 侦测ARCH/GARCH效应|时间序列 04：侦测 ARCH/GARCH 效应]]

## 关联卡片

- [[ARCH LM Test]]
- [[McLeod-Li Test]]
- [[GARCH Model Estimation Steps]]
- [[White Noise Test]]
- [[Ljung-Box Test]]

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
