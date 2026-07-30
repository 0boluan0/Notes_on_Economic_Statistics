---
aliases:
- Internal Growth Rate
- IGR
- 内部增长率
- 内含增长率
tags:
  - concept
  - finance
---
# Internal Growth Rate

## 一句话记忆

内部增长率是不依赖外部融资、只靠内部留存收益能支持的销售增长率。

## 它是什么

Internal Growth Rate 衡量企业在不新增外部资金的条件下，依靠经营留存和自然形成的经营负债能够支撑的增长速度。

一个常用口径是：

$$
IGR=\frac{ROA \times b}{1-ROA \times b}
$$

其中 $b$ 是利润留存率。

## 解决什么判断

- 企业如果不借新债、不发新股，最多能增长多快。
- 目标增长率是否必然带来外部融资需求。
- 增长受限来自盈利能力、资产效率还是股利支付。

## 最小例子

若 $ROA=8\%$，利润留存率 $b=50\%$：

$$
IGR=\frac{0.08\times 0.5}{1-0.08\times 0.5}=4.17\%
$$

## 易混点

- IGR 比 [[Sustainable Growth Rate]] 更保守，因为它不依赖外部债务融资。
- IGR 不是利润增长率，而是销售或经营规模可支持的增长率。
- 经营资产、经营负债和留存率的口径要一致。

## 来自课程位置

- [[02_财务报表分析和财务预测#一.内含增长率的测算|04_财务比率分析]]

## 关联卡片

- [[Sustainable Growth Rate]]
- [[Financial Ratio Analysis]]
- [[Return on Assets]]
- [[DuPont Analysis]]
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
