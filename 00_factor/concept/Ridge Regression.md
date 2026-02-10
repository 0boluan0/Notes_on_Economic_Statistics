---
aliases:
- Ridge Regression
- 岭回归
tags:
- concept
---
# Ridge Regression

## 它是什么
- 「Ridge Regression」是指在最小二乘中加入L2惩罚以缓解多重共线性。

## 最小可检索信息
- 定义：在最小二乘中加入L2惩罚以缓解多重共线性。
- 符号/公式：$\hat\beta=(X'X+\lambda I)^{-1}X'y。$
- 最小例子：高相关特征下稳定估计系数。

## 关联卡片
- [[Multicollinearity]]

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
