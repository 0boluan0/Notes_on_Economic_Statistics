---
aliases:
- McLeod-Li Portmanteau Test
- McLeod-Li Test
- McLeod-Li检验
- McLeod
tags:
- concept
---
# McLeod-Li Test

## 它是什么
- 「McLeod-Li Test」是指对平方残差的自相关进行检验以发现ARCH效应。

## 最小可检索信息
- 定义：对平方残差的自相关进行检验以发现ARCH效应。
- 符号/公式：对 $\hat{\varepsilon}_t^2$ 做 Ljung-Box Q 检验。
- 最小例子：GARCH前对残差平方做Q检验。

## 关联卡片
- [[Volatility Modeling-hub]]

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
