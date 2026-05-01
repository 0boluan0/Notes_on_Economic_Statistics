---
aliases:
- DuPont Analysis
- DuPont
- 杜邦分析
- 杜邦分析法
tags:
- concept
- finance
---
# DuPont Analysis

## 一句话记忆

杜邦分析把 ROE 拆成盈利能力、资产效率和财务杠杆三部分。

## 它是什么

DuPont Analysis 是解释 [[Return on Equity]] 来源的分解方法：

$$
ROE=\frac{Net\ Income}{Sales}\times\frac{Sales}{Total\ Assets}\times\frac{Total\ Assets}{Equity}
$$

也就是：

$$
ROE=Net\ Profit\ Margin \times Asset\ Turnover \times Equity\ Multiplier
$$

## 解决什么判断

- ROE 高是因为利润率高、资产周转快，还是杠杆高。
- 企业盈利改善来自经营效率还是财务结构。
- 可持续增长率变化来自哪个驱动因素。

## 最小例子

若净利率 10%，资产周转率 1.2，权益乘数 2：

$$
ROE=10\%\times 1.2\times 2=24\%
$$

## 易混点

- ROE 高不一定健康，可能只是杠杆过高。
- 杜邦分解是诊断结构，不是单独的估值模型。
- 用同业比较时，要注意会计口径和行业资产结构差异。

## 来自课程位置

- [[02_财务报表分析和财务预测#六.杜邦分析体系|04_财务比率分析]]

## 关联卡片

- [[Financial Ratio Analysis]]
- [[Return on Equity]]
- [[Net Profit Margin]]
- [[Return on Assets]]
- [[Equity Multiplier]]
- [[Sustainable Growth Rate]]

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
