---
aliases:
- Observation Window
- Lookback Window
- 观察期
tags:
- concept
---
# Observation Window

>[!note] 它是什么
> - 「Observation Window」是指用于估计或监测的历史数据时间跨度。
>
>[!note] 最小可检索信息
> - 定义：用于估计或监测的历史数据时间跨度。
> - 符号/公式：无固定符号。
> - 最小例子：用最近250个交易日估计波动率。
>
## 关联卡片
- [[VaR-hub]]

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
