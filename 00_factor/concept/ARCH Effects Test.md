---
aliases:
- ARCH Effects Test
- ARCH效应检验
- ARCH
tags:
- concept
---
# ARCH Effects Test

>[!note] 它是什么
> - 「ARCH Effects Test」是指用于检验时间序列残差是否存在ARCH效应（条件异方差）的统计检验。
>
>[!note] 最小可检索信息
> - 定义：用于检验时间序列残差是否存在ARCH效应（条件异方差）的统计检验。
> - 符号/公式：常用LM检验：将残差平方对其滞后回归，统计量 $nR^2 \sim \chi^2(q)$。
> - 最小例子：对AR(1)残差做ARCH(4) LM检验判断是否需要GARCH。
>
## 关联卡片
- [[IGARCH]]

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
