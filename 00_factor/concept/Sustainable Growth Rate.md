---
aliases:
- Sustainable Growth Rate
- SGR
- 可持续增长率
tags:
- concept
- finance
---
# Sustainable Growth Rate

## 一句话记忆

可持续增长率是在不增发新股、保持财务政策不变时，企业能长期承受的销售增长速度。

## 它是什么

Sustainable Growth Rate 衡量企业在维持经营效率、资本结构、股利政策不变的情况下，可以通过留存收益和同比例债务融资支持的增长率。

常用近似式：

$$
SGR=ROE \times b
$$

更严格的期末权益口径常写为：

$$
SGR=\frac{ROE \times b}{1-ROE \times b}
$$

其中 $b$ 是利润留存率。

## 解决什么判断

- 企业目标增长率是否超出自身融资能力。
- 增长过快时，是需要外部股权融资、提高盈利能力，还是调整股利政策。
- 增长质量是否与财务结构相匹配。

## 最小例子

若 $ROE=15\%$，利润留存率 $b=60\%$，近似可持续增长率为：

$$
SGR=15\%\times 60\%=9\%
$$

## 易混点

- [[Internal Growth Rate]] 假设没有外部融资；SGR 允许债务按既有资本结构同比例增长。
- SGR 不是企业想增长多少，而是在当前政策下能支撑多少。
- ROE、留存率、资产周转、利润率、杠杆任一变化，SGR 都会变。

## 来自课程位置

- [[02_财务报表分析和财务预测#二.可持续增长率的测算|04_财务比率分析]]

## 关联卡片

- [[Internal Growth Rate]]
- [[DuPont Analysis]]
- [[Return on Equity]]
- [[Financial Ratio Analysis]]
- [[Financial Management-hub]]

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
