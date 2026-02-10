---
aliases:
- Delta Approximation
- Delta近似法
tags:
- concept
---
# Delta Approximation

## 它是什么
- 「Delta Approximation」是指用一阶敏感度近似组合价值变化的线性方法。

## 最小可检索信息
- 定义：用一阶敏感度近似组合价值变化的线性方法。
- 符号/公式：$\Delta V\approx \Delta^T\Delta x$。
- 最小例子：用期权delta估计小幅标的变化下的P&L。

## 关联卡片
- [[Option Greeks-hub]]

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
