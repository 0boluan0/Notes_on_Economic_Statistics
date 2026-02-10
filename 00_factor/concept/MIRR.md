---
aliases:
- Modified Internal Rate of Return
- MIRR
- 修正内部收益率
tags:
- concept
---
# Modified Internal Rate of Return (MIRR)

## 它是什么
- 「MIRR」是指将正现金流以再投资率复利、负现金流以融资率折现后求的内部收益率。

## 最小可检索信息
- 定义：将正现金流以再投资率复利、负现金流以融资率折现后求的内部收益率。
- 符号/公式：$MIRR=\left(\frac{FV_{pos}}{PV_{neg}}\right)^{1/n}-1。$
- 最小例子：再投资率10%的项目MIRR=12%。

## 关联卡片
- [[IRR Calculation]]

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
